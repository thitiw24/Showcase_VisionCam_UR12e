"""
detect_to_robot_7.py
====================

Automatic Cycle version of detect_to_robot_6.py

System sequence:
    1. UR connects to Python TCP server.
    2. UR sends READY once after connection.
    3. Wait for UR -> START.
    4. START begins one cycle and starts cycle timer.
    5. Wait for UR -> EGG or CARTON.
    6. Python filters YOLO to that class only.
    7. Select the single highest-confidence detection.
    8. Read pixel + depth.
    9. Camera -> Gripper -> Robot Base transformation.
   10. Send X,Y,Z automatically to UR.
   11. Lock the target. No new target is selected in this cycle.
   12. Ignore intermediate UR status messages.
   13. Wait for UR -> DONE.
   14. DONE stops timer and saves one row to CSV.
   15. Reset cycle and wait for the next START.

CSV columns:
    Timestamp
    Cycle
    Target_Class
    Pixel_X
    Pixel_Y
    Robot_X_mm
    Robot_Y_mm
    Robot_Z_mm
    Cycle_Time_s

Important:
    CLASS_MAP below follows the current detect_to_robot_6.py:
        CARTON = 2
        EGG    = 0
        BLUE   = 1

    Verify this against:
        print(model.names)

Run:
    python detect_to_robot_7.py
    python detect_to_robot_7.py --ip 10.42.5.151
    python detect_to_robot_7.py --conf 0.5
"""

import os
import time
import argparse
import socket
import csv
from datetime import datetime

import numpy as np
import cv2 as cv

from ultralytics import YOLO
from ultralytics.engine.results import Results

import rtde_receive

from utils import frame_to_bgr_image                         # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode


# =============================================================================
# SETTINGS
# =============================================================================

ROBOT_IP = "10.42.5.151"
UR_PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Data", "best.pt")
CALIB_PATH = os.path.join(BASE_DIR, "Data", "calibration.npz")
HANDEYE_RESULT_FILE = os.path.join(BASE_DIR, "Data", "T_cam_to_gripper.npz")

# YOLO confidence threshold
CONF_THRESHOLD = 0.5

# Median depth patch radius
DEPTH_SAMPLE_RADIUS = 3

# CSV log file
CSV_FILE = os.path.join(BASE_DIR, "cycle_log.csv")

# Display
POINT_COLOR = (0, 255, 255)

COLOR_OK = (0, 255, 0)
COLOR_FAIL = (0, 0, 255)
COLOR_WARN = (0, 165, 255)
COLOR_INFO = (255, 255, 0)


# =============================================================================
# YOLO CLASS MAP
# =============================================================================
# Current mapping from detect_to_robot_6.py:
#
#     0 = EGG
#     1 = BLUE
#     2 = CARTON
#
# IMPORTANT:
# Check model.names when using a new best.pt.

CLASS_MAP = {
    "CARTON": 2,
    "BLUE": 1,
    "EGG": 0,
}

CLASS_NAME_LOOKUP = {
    2: "CARTON",
    1: "BLUE",
    0: "EGG",
}


# =============================================================================
# CYCLE STATES
# =============================================================================

WAIT_START = "WAIT_START"
WAIT_CLASS = "WAIT_CLASS"
WAIT_TARGET = "WAIT_TARGET"
WAIT_DONE = "WAIT_DONE"


# =============================================================================
# TCP BUFFER
# =============================================================================

_tcp_buffers = {}


# =============================================================================
# ROBOT TCP POSE
# =============================================================================

def get_tcp_pose_mm(rtde_r):
    """
    Read actual UR TCP pose.

    RTDE:
        position = meters
        rotation = axis-angle Rodrigues vector

    Return:
        T_gripper_to_base, 4x4 homogeneous matrix
        translation is in millimeters.
    """

    pose = rtde_r.getActualTCPPose()

    R, _ = cv.Rodrigues(np.array(pose[3:6]))

    t = np.array(pose[0:3]) * 1000.0

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = t

    return T


# =============================================================================
# DEPTH + BACKPROJECTION
# =============================================================================

def get_depth_at(depth_mm, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    """
    Read depth around (cx, cy).

    Uses median of valid (>0) depth pixels.
    Return depth in mm, or None if no valid depth exists.
    """

    h, w = depth_mm.shape

    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, w)

    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, h)

    patch = depth_mm[y0:y1, x0:x1]

    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def pixel_to_camera_point(px, py, depth_mm, camera_matrix):
    """
    Pinhole back-projection.

    Input:
        pixel = (px, py)
        depth = mm

    Output:
        [X, Y, Z] in CAMERA frame, mm.

    OpenCV convention:
        X = right
        Y = down
        Z = forward
    """

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]

    x = (px - cx0) * depth_mm / fx
    y = (py - cy0) * depth_mm / fy
    z = depth_mm

    return np.array([x, y, z], dtype=np.float64)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def draw_label(img, text, x, y):
    (tw, th), _ = cv.getTextSize(
        text,
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv.rectangle(
        img,
        (x, y - th - 8),
        (x + tw + 6, y + 4),
        POINT_COLOR,
        -1
    )

    cv.putText(
        img,
        text,
        (x + 3, y - 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


def draw_banner(img, text, y, color):
    (tw, th), _ = cv.getTextSize(
        text,
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        2
    )

    x = 20

    cv.rectangle(
        img,
        (x - 6, y - th - 10),
        (x + tw + 6, y + 6),
        (30, 30, 30),
        -1
    )

    cv.rectangle(
        img,
        (x - 6, y - th - 10),
        (x + tw + 6, y + 6),
        color,
        2
    )

    cv.putText(
        img,
        text,
        (x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# =============================================================================
# CSV LOGGING
# =============================================================================

CSV_HEADER = [
    "Timestamp",
    "Cycle",
    "Target_Class",
    "Pixel_X",
    "Pixel_Y",
    "Robot_X_mm",
    "Robot_Y_mm",
    "Robot_Z_mm",
    "Cycle_Time_s",
]


def init_csv():
    """
    Create CSV and header if the file does not exist or is empty.
    Existing data is preserved (file is never overwritten).
    """

    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(CSV_HEADER)

        print(f"[CSV] Created: {CSV_FILE}", flush=True)

    else:
        print(f"[CSV] Using existing file: {CSV_FILE}", flush=True)


def get_last_cycle_number():
    """
    Read the last cycle number from existing CSV so cycle numbers increment continuously across runs.
    """

    if not os.path.exists(CSV_FILE):
        return 0

    try:
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            lines = [row for row in csv.reader(f) if row]
            if len(lines) <= 1:
                return 0

            # Scan from the bottom for the last data row
            for row in reversed(lines[1:]):
                # New format (9 cols): Timestamp is col 0, Cycle is col 1
                if len(row) >= 9:
                    try:
                        return int(row[1])
                    except ValueError:
                        continue
                # Old format (8 cols): Cycle is col 0
                elif len(row) == 8:
                    try:
                        return int(row[0])
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[CSV] Warning reading last cycle number: {e}", flush=True)

    return 0


def save_cycle(
    cycle_number,
    target_class,
    pixel_x,
    pixel_y,
    robot_xyz,
    cycle_time
):
    """
    Save one completed cycle.

    This is called ONLY after DONE. Appends row to CSV.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    x, y, z = [float(v) for v in robot_xyz]

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            cycle_number,
            target_class,
            int(pixel_x),
            int(pixel_y),
            f"{x:.2f}",
            f"{y:.2f}",
            f"{z:.2f}",
            f"{cycle_time:.4f}",
        ])

    print(
        f"[CSV] Cycle {cycle_number} saved | "
        f"timestamp={timestamp} | "
        f"{target_class} | "
        f"pixel=({pixel_x},{pixel_y}) | "
        f"XYZ=({x:.2f},{y:.2f},{z:.2f}) mm | "
        f"time={cycle_time:.4f} s",
        flush=True
    )


# =============================================================================
# TCP SERVER
# =============================================================================

def start_tcp_server(host="0.0.0.0", port=UR_PORT):
    """
    Python = TCP SERVER
    UR = TCP CLIENT

    UR -> Python:
        READY
        START
        EGG
        BLUE
        CARTON
        status messages
        DONE

    Python -> UR:
        X,Y,Z
    """

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((host, port))

    server.listen(1)

    print(
        f"[TCP] Waiting for UR connection on port {port}...",
        flush=True
    )

    conn, addr = server.accept()

    print(
        f"[TCP] UR connected from {addr[0]}:{addr[1]}",
        flush=True
    )

    return server, conn


def send_message(conn, msg):
    """
    Send ASCII text to UR.
    """

    data = (msg).encode("ascii")

    conn.sendall(data)

    print(
        f"[TCP] PC -> UR: '{msg}'",
        flush=True
    )


def send_xyz(conn, xyz):
    """
    Send robot BASE-frame XYZ in millimeters.
    """

    x, y, z = [float(v) for v in xyz]

    msg = f"{x:.2f},{y:.2f},{z:.2f}"

    send_message(conn, msg)

    return msg


def receive_messages(conn):
    """
    Receive messages from UR.

    Handles:
        - newline messages
        - CR/LF
        - multiple messages in one TCP packet
        - partial messages
        - socket timeout
        - connection loss

    Return:
        list[str] = messages
        None = connection lost
    """

    global _tcp_buffers

    if conn not in _tcp_buffers:
        _tcp_buffers[conn] = ""

    try:

        data = conn.recv(1024)

        if not data:
            print(
                "[TCP] UR closed connection.",
                flush=True
            )

            remaining = _tcp_buffers.pop(
                conn,
                ""
            ).strip()

            if remaining:
                return [remaining]

            return None

        text = data.decode(
            "ascii",
            errors="replace"
        )

        _tcp_buffers[conn] += text

        messages = []

        # Normalize line endings
        buffer = _tcp_buffers[conn]
        buffer = buffer.replace("\r\n", "\n")
        buffer = buffer.replace("\r", "\n")

        _tcp_buffers[conn] = buffer

        # Complete line(s)
        if "\n" in _tcp_buffers[conn]:

            lines = _tcp_buffers[conn].split("\n")

            _tcp_buffers[conn] = lines[-1]

            for line in lines[:-1]:

                msg = line.strip()

                if msg:
                    messages.append(msg)

        # Some UR messages may arrive without newline.
        # Only treat residual buffer as a message when recv timed out,
        # so normal fragmented TCP packets are not incorrectly split.

        for msg in messages:
            print(
                f"[TCP] UR -> PC: '{msg}'",
                flush=True
            )

        return messages

    except socket.timeout:

        # If a timeout happens and there is residual text,
        # treat it as one message.
        if (
            conn in _tcp_buffers
            and _tcp_buffers[conn].strip()
        ):

            msg = _tcp_buffers[conn].strip()

            _tcp_buffers[conn] = ""

            print(
                f"[TCP] UR -> PC: '{msg}'",
                flush=True
            )

            return [msg]

        return []

    except (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
        OSError
    ) as e:

        print(
            f"[TCP] Receive error: {e}",
            flush=True
        )

        remaining = _tcp_buffers.pop(
            conn,
            ""
        ).strip()

        if remaining:
            return [remaining]

        return None


def close_tcp(server, conn):
    """
    Safely close TCP.
    """

    if conn is not None:

        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            conn.close()
        except OSError:
            pass

    if server is not None:

        try:
            server.close()
        except OSError:
            pass


# =============================================================================
# CLASS MESSAGE
# =============================================================================

def get_class_from_message(msg):
    """
    Convert UR class message into:
        ("CARTON", 2)
    or
        ("EGG", 0)
    or
        ("BLUE", 1)

    Return:
        (None, None) if not a class message.
    """

    if not msg:
        return None, None

    text = msg.upper().strip()

    if text == "CARTON":
        return "CARTON", CLASS_MAP["CARTON"]

    if text == "EGG":
        return "EGG", CLASS_MAP["EGG"]

    if text == "BLUE":
            return "BLUE", CLASS_MAP["BLUE"]

    return None, None


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic YOLO detection -> "
            "Robot Base XYZ -> UR TCP -> Cycle CSV logging"
        )
    )

    parser.add_argument(
        "--ip",
        default=ROBOT_IP,
        help=f"UR12e IP for RTDE pose reading (default: {ROBOT_IP})"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"YOLO confidence threshold (default: {CONF_THRESHOLD})"
    )

    args = parser.parse_args()

    robot_ip = args.ip
    conf_threshold = args.conf

    # -------------------------------------------------------------------------
    # State variables
    # -------------------------------------------------------------------------

    state = WAIT_START

    cycle_number = 0

    cycle_start_time = None

    cycle_target_class = None
    cycle_target_class_id = None

    cycle_pixel_x = None
    cycle_pixel_y = None

    cycle_robot_xyz = None

    target_locked = False

    # -------------------------------------------------------------------------
    # Display status
    # -------------------------------------------------------------------------

    last_status_text = "WAITING FOR START"
    last_status_color = COLOR_INFO

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    print("=" * 70, flush=True)
    print("detect_to_robot_7.py", flush=True)
    print("=" * 70, flush=True)

    print(
        f"[CONFIG] YOLO confidence = {conf_threshold}",
        flush=True
    )

    print(
        "[MODEL] Loading model...",
        flush=True
    )

    model = YOLO(MODEL_PATH)

    print(
        f"[MODEL] Class names: {model.names}",
        flush=True
    )

    print(
        "[MODEL] Expected mapping:",
        flush=True
    )

    print(
        f"        EGG    = {CLASS_MAP['EGG']}",
        flush=True
    )

    print(
        f"        CARTON = {CLASS_MAP['CARTON']}",
        flush=True
    )

    print(
        f"        BLUE = {CLASS_MAP['BLUE']}",
        flush=True
    )

    # -------------------------------------------------------------------------
    # Load camera intrinsic calibration
    # -------------------------------------------------------------------------

    print(
        "[CALIB] Loading intrinsic calibration...",
        flush=True
    )

    calib = np.load(CALIB_PATH)

    camera_matrix = calib["camera_matrix"]

    print(
        "[CALIB] Camera matrix loaded.",
        flush=True
    )

    # -------------------------------------------------------------------------
    # Load hand-eye
    # -------------------------------------------------------------------------

    print(
        "[HANDEYE] Loading T_cam_to_gripper...",
        flush=True
    )

    handeye = np.load(
        HANDEYE_RESULT_FILE,
        allow_pickle=True
    )

    T_cam2gripper = handeye["T_cam_to_gripper"]

    print(
        "[HANDEYE] T_cam_to_gripper loaded.",
        flush=True
    )

    # -------------------------------------------------------------------------
    # RTDE
    # -------------------------------------------------------------------------

    print(
        f"[RTDE] Connecting to UR12e at {robot_ip}...",
        flush=True
    )

    rtde_r = rtde_receive.RTDEReceiveInterface(
        robot_ip
    )

    print(
        "[RTDE] Connected.",
        flush=True
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    init_csv()
    cycle_number = get_last_cycle_number()
    if cycle_number > 0:
        print(
            f"[CSV] Resuming cycle count from cycle {cycle_number}.",
            flush=True
        )

    # -------------------------------------------------------------------------
    # TCP
    # -------------------------------------------------------------------------

    tcp_server = None
    tcp_conn = None

    try:

        tcp_server, tcp_conn = start_tcp_server(
            port=UR_PORT
        )

        # TCP socket must be non-blocking-ish for the main loop.
        tcp_conn.settimeout(0.01)

        print(
            "[TCP] Waiting for UR READY...",
            flush=True
        )

        # READY is expected exactly once after TCP connection.
        ready_received = False

        while not ready_received:

            msgs = receive_messages(tcp_conn)

            if msgs is None:
                raise RuntimeError(
                    "UR disconnected before READY."
                )

            for msg in msgs:

                if msg.upper().strip() == "READY":

                    ready_received = True

                    print(
                        "[TCP] READY received. "
                        "Connection initialization complete.",
                        flush=True
                    )

                else:

                    print(
                        f"[TCP] Ignoring message before READY: '{msg}'",
                        flush=True
                    )

        # ---------------------------------------------------------------------
        # Camera
        # ---------------------------------------------------------------------

        config = Config()

        pipeline = Pipeline()

        color_profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )

        config.enable_stream(
            color_profiles.get_default_video_stream_profile()
        )

        depth_profiles = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )

        config.enable_stream(
            depth_profiles.get_default_video_stream_profile()
        )

        config.set_align_mode(
            OBAlignMode.HW_MODE
        )

        pipeline.start(config)

        win_name = (
            "detect_to_robot_7 | "
            "Automatic Cycle"
        )

        cv.namedWindow(
            win_name,
            cv.WINDOW_NORMAL
        )

        print("", flush=True)
        print("=" * 70, flush=True)
        print("SYSTEM READY", flush=True)
        print("=" * 70, flush=True)
        print("WAIT_START -> WAIT_CLASS -> WAIT_TARGET -> WAIT_DONE", flush=True)
        print("Q/ESC = Quit", flush=True)
        print("", flush=True)

        # ---------------------------------------------------------------------
        # Main loop
        # ---------------------------------------------------------------------

        while True:

            # =================================================================
            # 1. RECEIVE UR MESSAGES
            # =================================================================

            msgs = receive_messages(tcp_conn)

            if msgs is None:

                print(
                    "[TCP] UR connection lost.",
                    flush=True
                )

                break

            for msg in msgs:

                text = msg.upper().strip()

                # -------------------------------------------------------------
                # START
                # -------------------------------------------------------------

                if text == "START":

                    if state == WAIT_START:

                        cycle_number += 1

                        cycle_start_time = time.perf_counter()

                        cycle_target_class = None
                        cycle_target_class_id = None

                        cycle_pixel_x = None
                        cycle_pixel_y = None

                        cycle_robot_xyz = None

                        target_locked = False

                        state = WAIT_CLASS

                        last_status_text = (
                            f"CYCLE {cycle_number}: WAITING FOR CLASS"
                        )

                        last_status_color = COLOR_INFO

                        print("", flush=True)

                        print(
                            f"[CYCLE {cycle_number}] START received.",
                            flush=True
                        )

                        print(
                            "[TIMER] Cycle timer started.",
                            flush=True
                        )

                    else:

                        print(
                            f"[TCP] START ignored. "
                            f"Current state = {state}",
                            flush=True
                        )

                # -------------------------------------------------------------
                # EGG / CARTON / BLUE
                # -------------------------------------------------------------

                elif text in ("EGG", "CARTON", "BLUE"):

                    class_name, class_id = get_class_from_message(
                        text
                    )

                    if state == WAIT_CLASS:

                        cycle_target_class = class_name
                        cycle_target_class_id = class_id

                        target_locked = False

                        state = WAIT_TARGET

                        last_status_text = (
                            f"CYCLE {cycle_number}: "
                            f"LOOKING FOR {class_name}"
                        )

                        last_status_color = COLOR_INFO

                        print(
                            f"[CLASS] Target = {class_name} "
                            f"(ID={class_id})",
                            flush=True
                        )

                        print(
                            f"[YOLO] Filtering only class ID {class_id}.",
                            flush=True
                        )

                    else:

                        print(
                            f"[TCP] Class '{text}' ignored. "
                            f"Current state = {state}",
                            flush=True
                        )

                # -------------------------------------------------------------
                # DONE
                # -------------------------------------------------------------

                elif text == "DONE":

                    if state == WAIT_DONE:

                        if cycle_start_time is None:

                            print(
                                "[CYCLE] ERROR: DONE received "
                                "without start timer.",
                                flush=True
                            )

                        else:

                            cycle_time = (
                                time.perf_counter()
                                - cycle_start_time
                            )

                            print("", flush=True)

                            print(
                                f"[CYCLE {cycle_number}] DONE received.",
                                flush=True
                            )

                            print(
                                f"[CYCLE {cycle_number}] "
                                f"Cycle time = {cycle_time:.4f} s",
                                flush=True
                            )

                            # -------------------------------------------------
                            # Save ONLY after DONE
                            # -------------------------------------------------

                            if (
                                cycle_target_class is not None
                                and cycle_pixel_x is not None
                                and cycle_pixel_y is not None
                                and cycle_robot_xyz is not None
                            ):

                                save_cycle(
                                    cycle_number,
                                    cycle_target_class,
                                    cycle_pixel_x,
                                    cycle_pixel_y,
                                    cycle_robot_xyz,
                                    cycle_time
                                )

                            else:

                                print(
                                    "[CSV] Cycle data incomplete. "
                                    "Nothing saved.",
                                    flush=True
                                )

                        # -----------------------------------------------------
                        # Reset cycle
                        # -----------------------------------------------------

                        cycle_start_time = None

                        cycle_target_class = None
                        cycle_target_class_id = None

                        cycle_pixel_x = None
                        cycle_pixel_y = None

                        cycle_robot_xyz = None

                        target_locked = False

                        state = WAIT_START

                        last_status_text = "WAITING FOR START"
                        last_status_color = COLOR_INFO

                        print(
                            "[CYCLE] Reset complete. "
                            "Waiting for next START.",
                            flush=True
                        )

                    else:

                        print(
                            f"[TCP] DONE ignored. "
                            f"Current state = {state}",
                            flush=True
                        )

                # -------------------------------------------------------------
                # READY after initial READY
                # -------------------------------------------------------------

                elif text == "READY":

                    print(
                        "[TCP] READY received again. "
                        "Ignored because READY is only for connection startup.",
                        flush=True
                    )

                # -------------------------------------------------------------
                # Other UR status
                # -------------------------------------------------------------

                else:

                    print(
                        f"[TCP] UR status: '{msg}'",
                        flush=True
                    )

            # =================================================================
            # 2. CAMERA FRAME
            # =================================================================

            frames = pipeline.wait_for_frames(100)

            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            img = frame_to_bgr_image(
                color_frame
            )

            if img is None:
                continue

            # =================================================================
            # 3. DEPTH
            # =================================================================

            d_width = depth_frame.get_width()
            d_height = depth_frame.get_height()

            scale = depth_frame.get_depth_scale()

            depth_raw = np.frombuffer(
                depth_frame.get_data(),
                dtype=np.uint16
            )

            depth_raw = depth_raw.reshape(
                (d_height, d_width)
            )

            depth_mm = (
                depth_raw.astype(np.float32)
                * scale
            )

            # =================================================================
            # 4. YOLO
            # =================================================================

            annotated = img.copy()

            target_base = None

            best_label = None
            best_conf = None

            best_pixel_x = None
            best_pixel_y = None

            # Only run target detection after START + EGG/CARTON/BLUE
            if (
                state == WAIT_TARGET
                and cycle_target_class_id is not None
                and not target_locked
            ):

                predict_kwargs = {
                    "source": img,
                    "conf": conf_threshold,
                    "device": "cpu",
                    "verbose": False,
                    "classes": [
                        cycle_target_class_id
                    ],
                }

                results = model.predict(
                    **predict_kwargs
                )

                r = results[0]                  # type: ignore[index]

                assert isinstance(r, Results)
                assert r.boxes is not None

                if len(r.boxes) > 0:

                    # ---------------------------------------------------------
                    # Pick ONE highest-confidence target
                    # ---------------------------------------------------------

                    confs = r.boxes.conf.tolist()

                    best_idx = int(
                        np.argmax(confs)
                    )

                    best_box = r.boxes[best_idx]

                    x1, y1, x2, y2 = map(
                        int,
                        best_box.xyxy[0].tolist()
                    )

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    best_conf = float(
                        best_box.conf[0]
                    )

                    best_cls = int(
                        best_box.cls[0]
                    )

                    best_label = model.names[
                        best_cls
                    ]

                    best_pixel_x = cx
                    best_pixel_y = cy

                    cv.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        POINT_COLOR,
                        2
                    )

                    cv.circle(
                        annotated,
                        (cx, cy),
                        5,
                        POINT_COLOR,
                        -1
                    )

                    # ---------------------------------------------------------
                    # Depth
                    # ---------------------------------------------------------

                    depth_val = None

                    if (
                        0 <= cy < depth_mm.shape[0]
                        and
                        0 <= cx < depth_mm.shape[1]
                    ):

                        depth_val = get_depth_at(
                            depth_mm,
                            cx,
                            cy
                        )

                    # ---------------------------------------------------------
                    # Coordinate transformation
                    # ---------------------------------------------------------

                    if depth_val is not None:

                        # Pixel + depth
                        p_cam = pixel_to_camera_point(
                            cx,
                            cy,
                            depth_val,
                            camera_matrix
                        )

                        p_cam_h = np.append(
                            p_cam,
                            1.0
                        )

                        # Camera -> Gripper
                        p_gripper_h = (
                            T_cam2gripper
                            @
                            p_cam_h
                        )

                        # Read current TCP pose
                        T_gripper2base = get_tcp_pose_mm(
                            rtde_r
                        )

                        # Gripper -> Robot Base
                        p_base_h = (
                            T_gripper2base
                            @
                            p_gripper_h
                        )

                        target_base = p_base_h[:3]

                        label = (
                            f"{best_label} "
                            f"{best_conf:.2f} "
                            f"base=("
                            f"{target_base[0]:.0f},"
                            f"{target_base[1]:.0f},"
                            f"{target_base[2]:.0f}"
                            f")mm"
                        )

                        draw_label(
                            annotated,
                            label,
                            x1,
                            y1 - 10
                        )

                        # -----------------------------------------------------
                        # FIRST VALID TARGET
                        # -----------------------------------------------------

                        if (
                            target_base is not None
                            and
                            not target_locked
                        ):

                            if tcp_conn is None:

                                print(
                                    "[SEND] UR disconnected. "
                                    "Cannot send target.",
                                    flush=True
                                )

                            else:

                                try:

                                    # Save cycle data BEFORE sending.
                                    cycle_pixel_x = cx
                                    cycle_pixel_y = cy

                                    cycle_robot_xyz = (
                                        target_base.copy()
                                    )

                                    # -------------------------------------------------
                                    # Automatic send
                                    # -------------------------------------------------

                                    sent_msg = send_xyz(
                                        tcp_conn,
                                        target_base
                                    )

                                    # Lock target immediately after successful send
                                    target_locked = True

                                    state = WAIT_DONE

                                    last_status_text = (
                                        f"SENT {sent_msg} mm"
                                    )

                                    last_status_color = COLOR_OK

                                    print("", flush=True)

                                    print(
                                        f"[TARGET] "
                                        f"{best_label} "
                                        f"(conf={best_conf:.3f})",
                                        flush=True
                                    )

                                    print(
                                        f"[PIXEL] "
                                        f"X={cx}, Y={cy}",
                                        flush=True
                                    )

                                    print(
                                        f"[ROBOT BASE] "
                                        f"X={target_base[0]:.2f}, "
                                        f"Y={target_base[1]:.2f}, "
                                        f"Z={target_base[2]:.2f} mm",
                                        flush=True
                                    )

                                    print(
                                        "[SEND] First valid target "
                                        "sent automatically.",
                                        flush=True
                                    )

                                    print(
                                        "[TARGET] Target LOCKED. "
                                        "Waiting for DONE...",
                                        flush=True
                                    )

                                except (
                                    BrokenPipeError,
                                    ConnectionResetError,
                                    ConnectionAbortedError,
                                    OSError
                                ) as e:

                                    print(
                                        f"[TCP] Send failed: {e}",
                                        flush=True
                                    )

                                    last_status_text = (
                                        "SEND FAILED"
                                    )

                                    last_status_color = COLOR_FAIL

                    else:

                        draw_label(
                            annotated,
                            (
                                f"{best_label} "
                                f"{best_conf:.2f} "
                                f"depth=N/A"
                            ),
                            x1,
                            y1 - 10
                        )

            # =================================================================
            # 5. DISPLAY
            # =================================================================

            if state == WAIT_START:

                state_text = "WAITING: START"

            elif state == WAIT_CLASS:

                state_text = (
                    f"CYCLE {cycle_number}: "
                    "WAITING: EGG/CARTON/BLUE"
                )

            elif state == WAIT_TARGET:

                state_text = (
                    f"CYCLE {cycle_number}: "
                    f"DETECTING {cycle_target_class}"
                )

            elif state == WAIT_DONE:

                state_text = (
                    f"CYCLE {cycle_number}: "
                    "TARGET SENT / WAITING DONE"
                )

            else:

                state_text = state

            cv.putText(
                annotated,
                state_text,
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_INFO,
                2
            )

            # Class filter
            if cycle_target_class_id is not None:

                class_status = (
                    f"Filter: {cycle_target_class} "
                    f"[{cycle_target_class_id}]"
                )

            else:

                class_status = "Filter: NONE"

            cv.putText(
                annotated,
                class_status,
                (20, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.60,
                COLOR_INFO,
                2
            )

            # Target lock
            lock_text = (
                "TARGET: LOCKED"
                if target_locked
                else "TARGET: NOT LOCKED"
            )

            lock_color = (
                COLOR_OK
                if target_locked
                else COLOR_WARN
            )

            cv.putText(
                annotated,
                lock_text,
                (20, 88),
                cv.FONT_HERSHEY_SIMPLEX,
                0.60,
                lock_color,
                2
            )

            # Connection
            conn_text = (
                "UR: CONNECTED"
                if tcp_conn is not None
                else "UR: DISCONNECTED"
            )

            conn_color = (
                COLOR_OK
                if tcp_conn is not None
                else COLOR_FAIL
            )

            (ctw, cth), _ = cv.getTextSize(
                conn_text,
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv.putText(
                annotated,
                conn_text,
                (
                    annotated.shape[1] - ctw - 20,
                    30
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                conn_color,
                2
            )

            # Cycle number
            cv.putText(
                annotated,
                f"Cycle: {cycle_number}",
                (20, annotated.shape[0] - 45),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv.putText(
                annotated,
                "Q/ESC: Quit",
                (20, annotated.shape[0] - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            # Current target display
            if target_base is not None:

                cv.putText(
                    annotated,
                    (
                        f"XYZ: "
                        f"{target_base[0]:.1f}, "
                        f"{target_base[1]:.1f}, "
                        f"{target_base[2]:.1f} mm"
                    ),
                    (20, 120),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_OK,
                    2
                )

            # Current cycle timer
            if cycle_start_time is not None:

                elapsed = (
                    time.perf_counter()
                    - cycle_start_time
                )

                cv.putText(
                    annotated,
                    f"Cycle time: {elapsed:.2f} s",
                    (20, 150),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_INFO,
                    2
                )

            # Status banner
            draw_banner(
                annotated,
                last_status_text,
                190,
                last_status_color
            )

            cv.imshow(
                win_name,
                annotated
            )

            # =================================================================
            # 6. KEYBOARD
            # =================================================================

            key = cv.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:

                print(
                    "[SYSTEM] Quit requested.",
                    flush=True
                )

                break

    finally:

        print(
            "[SYSTEM] Shutting down...",
            flush=True
        )

        try:
            pipeline.stop()                 # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            rtde_r.disconnect()
        except Exception:
            pass

        close_tcp(
            tcp_server,
            tcp_conn
        )

        cv.destroyAllWindows()

        print(
            f"[CSV] Log file: {CSV_FILE}",
            flush=True
        )

        print(
            "Done.",
            flush=True
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
