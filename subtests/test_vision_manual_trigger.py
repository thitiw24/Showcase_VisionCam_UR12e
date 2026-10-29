"""
final_test_01.py
===================
Test script for live object detection, UR pose reading, and calculating target 
object coordinates in the UR robot base frame.

Instead of establishing a TCP socket server to send data to UR, this script 
prints the target base XYZ payload data to stdout when the 'G' key is pressed.

Pipeline per frame:
    1. Grab color + depth from Orbbec Gemini 2 (hardware-aligned).
    2. Run YOLO on the color frame.
    3. Pick the SINGLE highest-confidence detection (across all classes).
    4. Sample depth at box center pixel.
    5. Back-project (pixel + depth) -> 3D point in CAMERA frame (mm).
    6. Transform camera-frame point -> GRIPPER frame, using T_cam_to_gripper.
    7. Transform gripper-frame point -> ROBOT BASE frame, using live TCP pose
       read from RTDE.
    8. Display live camera feed with annotations.
    9. When 'G' key is pressed, print the target base XYZ data formatted as it 
       would be sent to the UR.

Prerequisites:
    - Data/calibration.npz          (camera_matrix, dist_coeffs, ...)
    - Data/best.pt                  (your trained YOLO model)
    - T_cam_to_gripper.npz or Data/T_cam_to_gripper.npz
    - UR robot reachable over RTDE for TCP pose reading
    - Orbbec Gemini 2 connected
"""

import os
import argparse
import numpy as np
import cv2 as cv

from ultralytics import YOLO
from ultralytics.engine.results import Results

import rtde_receive                                                 # type: ignore[attr-defined]

from utils import frame_to_bgr_image                                # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode # type: ignore[attr-defined]


# =============================================================================
#  SETTINGS -- edit for your setup
# =============================================================================

ROBOT_IP = "10.42.5.151"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Data", "best.pt")
CALIB_PATH = os.path.join(BASE_DIR, "Data", "calibration.npz")

# Hand-eye result file path (checks Data/ first, then fallback to current folder)
HANDEYE_RESULT_FILE = os.path.join(BASE_DIR, "Data", "T_cam_to_gripper.npz")
if not os.path.exists(HANDEYE_RESULT_FILE):
    HANDEYE_RESULT_FILE = os.path.join(BASE_DIR, "T_cam_to_gripper.npz")

CONF_THRESHOLD = 0.25
DEPTH_SAMPLE_RADIUS = 3          # px radius for median depth sampling

POINT_COLOR = (0, 255, 255)


# =============================================================================
#  ROBOT (UR12e via ur-rtde)
# =============================================================================

def get_tcp_pose_mm(rtde_r):
    """T_gripper_to_base, 4x4 homogeneous transformation matrix, translation in mm."""
    pose = rtde_r.getActualTCPPose()
    R, _ = cv.Rodrigues(np.array(pose[3:6]))
    t = np.array(pose[0:3]) * 1000.0  # meters -> mm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# =============================================================================
#  DEPTH SAMPLING + BACKPROJECTION
# =============================================================================

def get_depth_at(depth_mm, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_mm.shape
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_camera_point(px, py, depth_mm, camera_matrix):
    """
    Pinhole back-projection: pixel (px,py) + depth (mm) -> 3D point in the
    CAMERA frame (mm). Standard OpenCV convention (X right, Y down, Z forward).
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx0, cy0 = camera_matrix[0, 2], camera_matrix[1, 2]

    x = (px - cx0) * depth_mm / fx
    y = (py - cy0) * depth_mm / fy
    z = depth_mm
    return np.array([x, y, z], dtype=np.float64)


def draw_label(img, text, x, y):
    (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), POINT_COLOR, -1)
    cv.putText(img, text, (x + 3, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# =============================================================================
#  SIMULATED UR TARGET TRANSMISSION
# =============================================================================

def print_target_send_data(xyz, label="target"):
    """
    Prints the target base XYZ coordinates in the exact format 
    that would be transmitted over TCP socket to the UR robot.
    Format: X,Y,Z\n (in mm)
    """
    x, y, z = [float(v) for v in xyz]
    msg = f"{x:.2f},{y:.2f},{z:.2f}\n"
    print("\n" + "=" * 60)
    print(f"[SIMULATED SEND TO UR] Class: '{label}'")
    print(f"  Target Base XYZ (mm) : X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
    print(f"  Payload String       : {repr(msg)}")
    print("=" * 60 + "\n")


# =============================================================================
#  MAIN LOOP
# =============================================================================

def main():
    global ROBOT_IP
    parser = argparse.ArgumentParser(description="Test Object Detection & UR Pose Reading (Simulated Send)")
    parser.add_argument("--ip", default=ROBOT_IP, help=f"UR IP address for RTDE (default: {ROBOT_IP})")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="YOLO confidence threshold")
    args = parser.parse_args()
    ROBOT_IP = args.ip

    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Loading intrinsic calibration from: {CALIB_PATH}")
    calib = np.load(CALIB_PATH)
    camera_matrix = calib["camera_matrix"]

    print(f"Loading hand-eye matrix from: {HANDEYE_RESULT_FILE}")
    handeye = np.load(HANDEYE_RESULT_FILE, allow_pickle=True)
    T_cam2gripper = handeye["T_cam_to_gripper"]

    print(f"Connecting to UR at {ROBOT_IP} for RTDE pose reading...")
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("RTDE connected successfully.")

    config = Config()
    pipeline = Pipeline()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    config.enable_stream(color_profiles.get_default_video_stream_profile())

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(depth_profiles.get_default_video_stream_profile())

    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.start(config)

    win_name = "final_test_01 | Detection & UR Pose Test"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    print("\nSystem running.")
    print("Controls:")
    print("  'g' : Print simulated target pose data to be sent to UR")
    print("  'q' / ESC : Quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            img = frame_to_bgr_image(color_frame)
            if img is None:
                continue

            d_width = depth_frame.get_width()
            d_height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_raw = depth_raw.reshape((d_height, d_width))
            depth_mm = depth_raw.astype(np.float32) * scale

            results = model.predict(source=img, conf=args.conf, device="cpu", verbose=False)
            r = results[0]                                                                      # type: ignore[index]
            assert isinstance(r, Results)
            assert r.boxes is not None

            annotated = img.copy()
            target_base = None
            best_label = None
            best_conf = None

            if len(r.boxes) > 0:
                # Pick the single highest-confidence detection
                confs = r.boxes.conf.tolist()
                best_idx = int(np.argmax(confs))
                best_box = r.boxes[best_idx]

                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                best_conf = float(best_box.conf[0])
                best_cls = int(best_box.cls[0])
                best_label = model.names[best_cls]

                cv.rectangle(annotated, (x1, y1), (x2, y2), POINT_COLOR, 2)
                cv.circle(annotated, (cx, cy), 5, POINT_COLOR, -1)

                depth_val = None
                if 0 <= cy < depth_mm.shape[0] and 0 <= cx < depth_mm.shape[1]:
                    depth_val = get_depth_at(depth_mm, cx, cy)

                if depth_val is not None:
                    # 1) pixel + depth -> camera frame (mm)
                    p_cam = pixel_to_camera_point(cx, cy, depth_val, camera_matrix)
                    p_cam_h = np.append(p_cam, 1.0)

                    # 2) camera frame -> gripper frame
                    p_gripper_h = T_cam2gripper @ p_cam_h

                    # 3) gripper frame -> base frame (read live TCP pose via RTDE)
                    T_gripper2base = get_tcp_pose_mm(rtde_r)
                    p_base_h = T_gripper2base @ p_gripper_h

                    target_base = p_base_h[:3]

                    label_text = f"{best_label} {best_conf:.2f} | base=({target_base[0]:.0f},{target_base[1]:.0f},{target_base[2]:.0f})mm"
                    draw_label(annotated, label_text, x1, y1 - 10)
                else:
                    draw_label(annotated, f"{best_label} {best_conf:.2f} | depth=N/A", x1, y1 - 10)

            status = f"Detections: {len(r.boxes)}"
            cv.putText(annotated, status, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv.putText(annotated, "G: Print Target Data | Q/ESC: Quit", (20, annotated.shape[0] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv.imshow(win_name, annotated)

            if target_base is not None:
                print(f"\r[LIVE TARGET] {best_label} (conf={best_conf:.2f}) -> "
                      f"base_xyz_mm = X:{target_base[0]:8.2f} Y:{target_base[1]:8.2f} Z:{target_base[2]:8.2f}", end="")

            key = cv.waitKey(1) & 0xFF

            if key == ord('g') or key == ord('G'):
                if target_base is not None:
                    print_target_send_data(target_base, best_label)                                 # type: ignore[index]
                else:
                    print("\n[SEND] No valid object target detected in current frame -- ignored.")

            if key == ord('q') or key == ord('Q') or key == 27:
                print("\nExiting...")
                break

    finally:
        pipeline.stop()
        rtde_r.disconnect()
        cv.destroyAllWindows()
        print("Stopped clean.")


if __name__ == "__main__":
    main()
