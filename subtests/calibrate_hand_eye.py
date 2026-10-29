"""
eye-in-hand-07.py
==========================
Eye-in-Hand Calibration
Orbbec Gemini 2 + UR12e

Updates in eye-in-hand-06:
- Unlimited dynamic target pose generation in collect_plus (get_calibration_target).
- Non-blocking target error pose saving: target errors are displayed on UI and printed, but poses are saved without rejection.
- Automatic RTDE connection recovery (auto-reconnect on socket loss or robot error state).
- Carried over from detect_to_robot_4.py: RTDE and the camera are now both
  managed as independent, non-blocking background links (RTDELink /
  try_open_camera). Either can be absent at startup, drop out mid-session,
  or come back later -- collect() / collect_plus() keep running, show a
  "waiting" placeholder / indicator, and resume automatically once the
  link is back. Nothing here blocks the GUI loop waiting on either link.
"""

import os
import time
import threading
import cv2
import numpy as np
import json
import glob
import argparse
from datetime import datetime

import rtde_receive

from utils import frame_to_bgr_image                    # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType


# =============================================================================
# SETTINGS
# =============================================================================

ROBOT_IP = "10.42.5.151"

PATTERN_SIZE = (9, 7)
SQUARE_SIZE_MM = 20.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROJECT_DATA_DIR = os.path.join(PROJECT_DIR, "Data") if os.path.exists(os.path.join(PROJECT_DIR, "Data")) else os.path.join(BASE_DIR, "Data")

CAMERA_CALIB_FILE = os.path.join(
    PROJECT_DATA_DIR, "calibration.npz"
)

DATA_DIR = os.path.join(
    BASE_DIR, "handeye_eih_data"
)

# Save output directly to project Data folder to update main production system immediately
RESULT_FILE = os.path.join(
    PROJECT_DATA_DIR, "T_cam_to_gripper.npz"
)

REPORT_FILE = os.path.join(
    PROJECT_DATA_DIR, "T_cam_to_gripper_report.json"
)

EXPORT_FILE = os.path.join(
    PROJECT_DATA_DIR, "T_cam_to_gripper_manual.npy"
)


# -------------------------------------------------------------------------
# Collection rules
# -------------------------------------------------------------------------

MIN_POSES = 20

# If robot moves less than this, do not save
MIN_TRANSLATION_MM = 100.0
MIN_ROTATION_DEG = 15.0

# Used for checking duplicates during calibration
DUP_TRANSLATION_MM = 5.0
DUP_ROTATION_DEG = 1.0

# Combined data must have sufficient rotation diversity
MIN_ROTATION_SPAN_DEG = 30.0

CRITERIA = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# --- reconnect / liveness timing (see detect_to_robot_4.py) ---
CAMERA_FRAME_TIMEOUT = 1.5
CAMERA_RETRY_INTERVAL = 2.0      # sec between camera (re)connect attempts
RTDE_RETRY_INTERVAL = 2.0        # sec between RTDE (re)connect attempts

# =============================================================================
# ROBOT
# =============================================================================

def get_tcp_pose_mm(rtde_r):
    """
    Read UR TCP pose with automatic reconnection fallback.

    UR:
        x,y,z   = meters
        rx,ry,rz = rotation vector in radians

    Return:
        4x4 T_gripper_to_base
        translation in mm
    """

    try:
        if hasattr(rtde_r, "isConnected") and not rtde_r.isConnected():
            rtde_r.reconnect()
    except Exception:
        pass

    try:
        pose = rtde_r.getActualTCPPose()
    except Exception:
        pose = None

    if pose is None or len(pose) != 6:
        try:
            rtde_r.reconnect()
            pose = rtde_r.getActualTCPPose()
        except Exception:
            pass

    if pose is None or len(pose) != 6:
        raise RuntimeError(
            "Invalid TCP pose received from UR (RTDE connection dropped or robot in error state)"
        )

    pose = np.asarray(pose, dtype=np.float64)

    if not np.all(np.isfinite(pose)):
        raise RuntimeError(
            "TCP pose contains NaN or Inf"
        )

    R, _ = cv2.Rodrigues(pose[3:6])

    t = pose[:3] * 1000.0

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def rotation_difference_deg(R1, R2):
    """
    Rotation difference between two rotation matrices.
    """

    R_rel = R2 @ R1.T

    value = (np.trace(R_rel) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def pose_difference(T1, T2):
    """
    Difference between two robot poses.

    Returns:
        translation_mm
        rotation_deg
    """

    translation = np.linalg.norm(
        T2[:3, 3] - T1[:3, 3]
    )

    rotation = rotation_difference_deg(
        T1[:3, :3],
        T2[:3, :3]
    )

    return float(translation), float(rotation)


# =============================================================================
# RTDE LINK -- fully backgrounded: a worker thread owns the connection AND
# does every getActualTCPPose() read. The GUI thread only ever calls
# get_pose_mm(), which returns a cached copy instantly and never touches
# the network -- so a stalled/slow RTDE read can never freeze the window.
# =============================================================================

class RTDELink:
    def __init__(self, ip, retry_interval=RTDE_RETRY_INTERVAL, poll_interval=0.05):
        self.ip = ip
        self.retry_interval = retry_interval
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._r = None
        self._pose = None
        self._stop = False
        threading.Thread(target=self._loop, daemon=True, name="RTDELink").start()

    def _loop(self):
        while not self._stop:
            with self._lock:
                r = self._r
            if r is None:
                try:
                    new_r = rtde_receive.RTDEReceiveInterface(self.ip)
                    if new_r.isConnected():
                        with self._lock:
                            self._r = new_r
                        print("[RTDE] Connected to UR.")
                    else:
                        time.sleep(self.retry_interval)
                except Exception as e:
                    print(f"[RTDE] Connect failed, will retry: {e}")
                    time.sleep(self.retry_interval)
                continue

            # This getActualTCPPose() call happens ONLY on this worker
            # thread -- it can stall as long as it wants without ever
            # touching the GUI loop.
            try:
                T = get_tcp_pose_mm(r)
                with self._lock:
                    self._pose = T
            except Exception as e:
                print(f"[RTDE] Read failed, will reconnect: {e}")
                self._drop()
                continue

            time.sleep(self.poll_interval)

    def get_pose_mm(self):
        """Returns a cached 4x4 T_gripper_to_base (mm), or None. Instant, never blocks."""
        with self._lock:
            return None if self._pose is None else self._pose.copy()

    def _drop(self):
        with self._lock:
            r = self._r
            self._r = None
            self._pose = None
        if r is not None:
            try:
                r.disconnect()
            except Exception:
                pass

    def is_connected(self):
        with self._lock:
            return self._r is not None

    def close(self):
        self._stop = True
        self._drop()


# =============================================================================
# CAMERA LINK
# =============================================================================

# How often to try opening the camera again
CAMERA_RETRY_INTERVAL = 2.0

# If no valid COLOR frame arrives for this long,
# consider the camera stream unhealthy and restart the pipeline.
CAMERA_FRAME_TIMEOUT = 1.5


def try_open_camera():

    pipeline = None

    try:

        config = Config()
        pipeline = Pipeline()

        color_profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )

        color_profile = (
            color_profiles.get_default_video_stream_profile()
        )

        config.enable_stream(
            color_profile
        )

        pipeline.start(config)

        return pipeline

    except Exception as e:

        print(
            f"[CAM] Failed to start pipeline: {e}"
        )

        if pipeline is not None:

            try:
                pipeline.stop()

            except Exception:
                pass

        return None


def safe_stop_pipeline(pipeline):

    if pipeline is None:
        return

    try:

        pipeline.stop()

        print(
            "[CAM] Pipeline stopped."
        )

    except Exception as e:

        msg = str(e).lower()

        if (
            "deactivated" in msg
            or
            "disconnected" in msg
            or
            "no longer present" in msg
        ):

            print(
                "[CAM] Pipeline already disconnected."
            )

        else:

            print(
                f"[CAM] Pipeline stop warning: {e}"
            )


def get_camera_frame(pipeline):

    """
    Read ONE color frame.

    IMPORTANT:
    This function is called ONLY by CameraLink's
    background thread.

    It must NEVER be called directly by the GUI.
    """

    frames = pipeline.wait_for_frames(100)

    if frames is None:
        return None

    color_frame = (
        frames.get_color_frame()
    )

    if color_frame is None:
        return None

    img = frame_to_bgr_image(
        color_frame
    )

    if img is None:
        return None

    return img


class CameraLink:

    def __init__(
        self,
        retry_interval=CAMERA_RETRY_INTERVAL,
        frame_timeout=CAMERA_FRAME_TIMEOUT
    ):

        self.retry_interval = retry_interval
        self.frame_timeout = frame_timeout

        self._lock = threading.Lock()

        self._pipeline = None
        self._frame = None

        self._connected = False

        # Time when the most recent valid frame arrived
        self._last_frame_time = 0.0

        self._stop = False

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="CameraLink"
        )

        self._thread.start()

    # -------------------------------------------------------------------------
    # DROP CURRENT PIPELINE
    # -------------------------------------------------------------------------

    def _drop_pipeline(
        self,
        pipeline
    ):

        with self._lock:

            # Another thread may already have
            # replaced this pipeline.
            if self._pipeline is not pipeline:
                return

            self._pipeline = None
            self._frame = None

            self._connected = False

            self._last_frame_time = 0.0

        print(
            "[CAM] Camera stream marked unhealthy."
        )

        safe_stop_pipeline(
            pipeline
        )

        print(
            "[CAM] Waiting for camera connecting..."
        )

    # -------------------------------------------------------------------------
    # CAMERA WORKER
    # -------------------------------------------------------------------------

    def _loop(self):

        next_retry = 0.0

        while not self._stop:

            # =================================================================
            # GET CURRENT PIPELINE
            # =================================================================

            with self._lock:

                pipeline = self._pipeline

            # =================================================================
            # NO PIPELINE
            #
            # Try to create a new one.
            # =================================================================

            if pipeline is None:

                now = time.time()

                if now < next_retry:

                    time.sleep(0.05)

                    continue

                next_retry = (
                    now +
                    self.retry_interval
                )

                print(
                    "[CAM] Trying to open camera ..."
                )

                new_pipeline = (
                    try_open_camera()
                )

                if new_pipeline is not None:

                    with self._lock:

                        if self._stop:

                            safe_stop_pipeline(
                                new_pipeline
                            )

                            break

                        self._pipeline = (
                            new_pipeline
                        )

                        self._frame = None

                        self._connected = True

                        # Start timeout timer.
                        # It will be refreshed when the
                        # first valid frame arrives.
                        self._last_frame_time = (
                            time.time()
                        )

                    print(
                        "[CAM] Camera connected."
                    )

                continue

            # =================================================================
            # GET FRAME
            # =================================================================

            try:

                img = get_camera_frame(
                    pipeline
                )

            except Exception as e:

                print()
                print(
                    "[CAM] Frame grab exception:"
                )
                print(
                    f"       {e}"
                )

                self._drop_pipeline(
                    pipeline
                )

                next_retry = (
                    time.time()
                    +
                    self.retry_interval
                )

                continue

            # =================================================================
            # VALID FRAME
            # =================================================================

            if img is not None:

                with self._lock:

                    # Make sure this pipeline is still
                    # the active pipeline.

                    if (
                        self._pipeline
                        is pipeline
                    ):

                        self._frame = img

                        self._last_frame_time = (
                            time.time()
                        )

                        self._connected = True

                continue

            # =================================================================
            # NO FRAME
            #
            # This is the important part.
            #
            # wait_for_frames() can return None without
            # throwing an exception.
            #
            # Therefore an exception-only reconnect
            # mechanism is not enough.
            # =================================================================

            now = time.time()

            with self._lock:

                last_frame_time = (
                    self._last_frame_time
                )

            if (
                last_frame_time > 0.0
                and
                (
                    now -
                    last_frame_time
                ) > self.frame_timeout
            ):

                print()
                print(
                    "[CAM] Camera stream timeout!"
                )

                print(
                    "[CAM] No valid color frame for "
                    f"{now - last_frame_time:.2f} sec."
                )

                self._drop_pipeline(
                    pipeline
                )

                next_retry = (
                    time.time()
                    +
                    self.retry_interval
                )

                continue

            time.sleep(
                0.01
            )

    # -------------------------------------------------------------------------
    # GET LATEST FRAME
    # -------------------------------------------------------------------------

    def get_frame(self):

        """
        Return the latest cached image.

        This function NEVER waits for Orbbec.
        """

        with self._lock:

            if self._frame is None:
                return None

            return self._frame.copy()

    # -------------------------------------------------------------------------
    # CAMERA STATUS
    # -------------------------------------------------------------------------

    def is_connected(self):

        with self._lock:

            return (
                self._connected
                and
                self._pipeline is not None
            )

    # -------------------------------------------------------------------------
    # FACTORY INTRINSICS
    # -------------------------------------------------------------------------

    def get_intrinsics(self):
        """
        Extract Factory Intrinsic Matrix and Distortion Coefficients directly from Orbbec camera
        """
        with self._lock:
            if self._pipeline is not None:
                try:
                    param = self._pipeline.get_camera_param()
                    rgb = param.rgb_intrinsic
                    dist = param.rgb_distortion
                    k = np.array([
                        [rgb.fx, 0.0, rgb.cx],
                        [0.0, rgb.fy, rgb.cy],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    d = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float64)
                    return k, d
                except Exception as e:
                    print(f"[CAM] Failed to read Factory Intrinsic: {e}")
        return None, None

    # -------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------

    def close(self):

        self._stop = True

        with self._lock:

            pipeline = self._pipeline

            self._pipeline = None
            self._frame = None

            self._connected = False

            self._last_frame_time = 0.0

        safe_stop_pipeline(
            pipeline
        )

def make_placeholder(width, height, lines):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y = height // 2 - (len(lines) - 1) * 18
    for line, color in lines:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        y += 36
    cv2.putText(img, "Q/ESC: Quit", (40, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def draw_rtde_indicator(img, rtde_link):
    connected = rtde_link.is_connected()
    text = "RTDE: CONNECTED" if connected else "RTDE: WAITING"
    color = (0, 255, 0) if connected else (0, 165, 255)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = img.shape[1] - tw - 20
    cv2.putText(img, text, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_cam_indicator(img, cam_link):
    connected = cam_link.is_connected()
    text = "CAM: CONNECTED" if connected else "CAM: WAITING"
    color = (0, 255, 0) if connected else (0, 165, 255)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = img.shape[1] - tw - 20
    cv2.putText(img, text, (x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# =============================================================================
# CHECKERBOARD
# =============================================================================

def build_objp():

    objp = np.zeros(
        (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    return objp


def detect_board_pose(
    gray,
    objp,
    camera_matrix,
    dist_coeffs
):

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        None
    )

    if not found:
        return False, None, None, None

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CRITERIA
    )

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        camera_matrix,
        dist_coeffs
    )

    if not ok:
        return False, None, None, None

    return True, rvec, tvec, corners


def rvec_tvec_to_T(rvec, tvec):

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


# =============================================================================
# FILE
# =============================================================================

def imwrite_unicode(path, img):

    ext = os.path.splitext(path)[1]

    success, encoded = cv2.imencode(
        ext,
        img
    )

    if success:
        encoded.tofile(path)

    return success


def create_session_dir():

    os.makedirs(DATA_DIR, exist_ok=True)

    ts = datetime.now().strftime(
        "session_%Y%m%d_%H%M%S"
    )

    session_dir = os.path.join(
        DATA_DIR,
        ts
    )

    os.makedirs(session_dir, exist_ok=True)

    return session_dir


def get_latest_session():

    if not os.path.exists(DATA_DIR):
        return None

    sessions = []

    for name in os.listdir(DATA_DIR):

        path = os.path.join(DATA_DIR, name)

        if os.path.isdir(path) and name.startswith("session_"):
            sessions.append(path)

    if not sessions:
        return None

    sessions.sort()

    return sessions[-1]


# =============================================================================
# COLLECT PLUS - HELPER FUNCTIONS
# =============================================================================

def ur_pose_to_T(pose):
    """
    Convert UR pose:

        [x, y, z, rx, ry, rz]

    to:

        4x4 homogeneous transformation matrix

    Input:
        x,y,z  = mm
        rx,ry,rz = radians

    Output:
        T = 4x4
    """

    pose = np.asarray(
        pose,
        dtype=np.float64
    )

    if len(pose) != 6:
        raise ValueError(
            "UR pose must contain 6 values."
        )

    R, _ = cv2.Rodrigues(
        pose[3:6]
    )

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = pose[:3]

    return T


def T_to_ur_pose(T):
    """
    Convert 4x4 transformation matrix
    back to UR pose:

        [x, y, z, rx, ry, rz]

    Output:
        x,y,z  = mm
        rx,ry,rz = radians
    """

    T = np.asarray(
        T,
        dtype=np.float64
    )

    if T.shape != (4, 4):
        raise ValueError(
            "T must be a 4x4 matrix."
        )

    rvec, _ = cv2.Rodrigues(
        T[:3, :3]
    )

    pose = np.zeros(
        6,
        dtype=np.float64
    )

    pose[:3] = T[:3, 3]
    pose[3:6] = rvec.flatten()

    return pose


def get_calibration_target(T_reference, index):
    """
    Generate target pose matrix for any target index (0-indexed).
    Index 0 to 19: Baseline 20 calibration offsets.
    Index >= 20: Programmatically generated dynamic target offsets (unlimited target poses).
    """

    baseline_offsets = [
        # Translation offsets (mm) - 14 poses
        {"type": "trans", "offset": np.array([+50.0,   0.0,   0.0])},
        {"type": "trans", "offset": np.array([-50.0,   0.0,   0.0])},
        {"type": "trans", "offset": np.array([  0.0, +50.0,   0.0])},
        {"type": "trans", "offset": np.array([  0.0, -50.0,   0.0])},
        {"type": "trans", "offset": np.array([  0.0,   0.0, +50.0])},
        {"type": "trans", "offset": np.array([  0.0,   0.0, -50.0])},
        {"type": "trans", "offset": np.array([+40.0, +40.0,  0.0])},
        {"type": "trans", "offset": np.array([-40.0, +40.0,  0.0])},
        {"type": "trans", "offset": np.array([+40.0, -40.0,  0.0])},
        {"type": "trans", "offset": np.array([-40.0, -40.0,  0.0])},
        {"type": "trans", "offset": np.array([+40.0,  0.0, +40.0])},
        {"type": "trans", "offset": np.array([-40.0,  0.0, +40.0])},
        {"type": "trans", "offset": np.array([ 0.0, +40.0, +40.0])},
        {"type": "trans", "offset": np.array([ 0.0, -40.0, +40.0])},

        # Rotation offsets (deg) - 6 poses
        {"type": "rot", "axis": np.array([1.0, 0.0, 0.0]), "angle_deg": +20.0},
        {"type": "rot", "axis": np.array([1.0, 0.0, 0.0]), "angle_deg": -20.0},
        {"type": "rot", "axis": np.array([0.0, 1.0, 0.0]), "angle_deg": +20.0},
        {"type": "rot", "axis": np.array([0.0, 1.0, 0.0]), "angle_deg": -20.0},
        {"type": "rot", "axis": np.array([0.0, 0.0, 1.0]), "angle_deg": +20.0},
        {"type": "rot", "axis": np.array([0.0, 0.0, 1.0]), "angle_deg": -20.0},
    ]

    T = T_reference.copy()

    if index < len(baseline_offsets):
        item = baseline_offsets[index]
        if item["type"] == "trans":
            T[:3, 3] = T_reference[:3, 3] + item["offset"]
        else:
            angle_rad = np.radians(item["angle_deg"])
            rvec = item["axis"] * angle_rad
            R_delta, _ = cv2.Rodrigues(rvec)
            T[:3, :3] = R_delta @ T_reference[:3, :3]
    else:
        # Dynamic procedural targets for index >= 20
        k = index - len(baseline_offsets)

        trans_mag = 30.0 + (k % 4) * 10.0  # 30, 40, 50, 60 mm
        angles_t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angle_t = angles_t[k % 8]
        z_offset = ((k // 8) % 3 - 1) * 25.0  # -25, 0, +25 mm

        dx = trans_mag * np.cos(angle_t)
        dy = trans_mag * np.sin(angle_t)
        dz = z_offset

        rx_deg = 15.0 * np.sin(k * 1.1)
        ry_deg = 15.0 * np.cos(k * 1.3)
        rz_deg = 10.0 * np.sin(k * 1.7)

        rvec = np.radians(np.array([rx_deg, ry_deg, rz_deg]))
        R_delta, _ = cv2.Rodrigues(rvec)

        T[:3, :3] = R_delta @ T_reference[:3, :3]
        T[:3, 3] = T_reference[:3, 3] + np.array([dx, dy, dz])

    return T


def build_calibration_targets(T_reference, count=20):
    return [get_calibration_target(T_reference, i) for i in range(count)]

def get_camera_calibration_data(cam_link=None, timeout_sec=3.0):
    """
    Load Intrinsic parameters:
    1. Attempt to extract Factory Intrinsic from Orbbec camera via cam_link.get_intrinsics()
    2. If camera not available, fall back to CAMERA_CALIB_FILE
    """
    if cam_link is not None:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            k, d = cam_link.get_intrinsics()
            if k is not None:
                print("\n[CALIB] Factory Intrinsic successfully loaded directly from Orbbec camera")
                print(f"        fx={k[0,0]:.1f}, fy={k[1,1]:.1f}, cx={k[0,2]:.1f}, cy={k[1,2]:.1f}")
                return k, d
            time.sleep(0.1)

    if os.path.exists(CAMERA_CALIB_FILE):
        calib = np.load(CAMERA_CALIB_FILE)
        print(f"\n[CALIB] Using fallback Intrinsic file: {CAMERA_CALIB_FILE}")
        return calib["camera_matrix"], calib["dist_coeffs"]

    raise FileNotFoundError(f"[ERROR] Neither Factory Intrinsic from camera nor {CAMERA_CALIB_FILE} could be found")


# =============================================================================
# COLLECT
# =============================================================================

def collect():

    os.makedirs(DATA_DIR, exist_ok=True)

    session_dir = create_session_dir()

    print()
    print("=" * 70)
    print("  NEW EYE-IN-HAND COLLECTION SESSION")
    print("=" * 70)

    print(f"\n  Session:")
    print(f"  {session_dir}")

    # Start camera and robot links
    cam_link = CameraLink()
    rtde_link = RTDELink(ROBOT_IP)

    camera_matrix, dist_coeffs = get_camera_calibration_data(cam_link)

    objp = build_objp()

    print("\n  Checklist:")
    print("  1. Checkerboard MUST stay stationary")
    print("  2. Move the UR12e, NOT the checkerboard")
    print("  3. Change both position and wrist orientation")
    print("  4. Wait until robot is completely still")
    print("  5. Press S to save")
    print()
    print("  Minimum:")
    print(f"    Translation > {MIN_TRANSLATION_MM:.1f} mm")
    print(f"    Rotation    > {MIN_ROTATION_DEG:.1f} deg")
    print()

    d_trans = None
    d_rot = None

    saved = 0

    previous_T_robot = None

    win_name = (
        "Eye-in-Hand Collect | "
        "Orbbec Gemini 2 + UR12e"
    )

    cv2.namedWindow(
        win_name,
        cv2.WINDOW_NORMAL
    )

    try:

        while True:

            # ---------------- read cached camera frame ----------------
            # (CameraLink's worker thread owns the pipeline and its
            # retries entirely -- this call is instant either way.)
            img = cam_link.get_frame()

            if img is None:
                placeholder = make_placeholder(
                    960, 540,
                    [
                        (
                            "CAMERA STREAM UNAVAILABLE",
                            (0, 0, 255)
                        ),
                        (
                            "Waiting for camera connecting...",
                            (0, 165, 255)
                        ),
                        (
                            "Please reconnect Orbbec Gemini 2",
                            (255, 255, 255)
                        ),
                    ],
                )
                draw_rtde_indicator(placeholder, rtde_link)
                draw_cam_indicator(placeholder, cam_link)
                cv2.imshow(win_name, placeholder)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):

                    print()
                    print(
                        "\n  [ESC] Collection stopped."
                    )

                    break
                continue

            # -------------------------------------------------------------
            # Read robot pose LIVE (cached -- instant, never blocks)
            # -------------------------------------------------------------

            T_robot = rtde_link.get_pose_mm()
            robot_ok = T_robot is not None
            robot_error = "" if robot_ok else "RTDE not connected"


            # -------------------------------------------------------------
            # Camera
            # -------------------------------------------------------------

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            found, rvec, tvec, corners = detect_board_pose(
                gray,
                objp,
                camera_matrix,
                dist_coeffs
            )

            display = img.copy()

            # -------------------------------------------------------------
            # Checkerboard display
            # -------------------------------------------------------------

            if found:

                assert (
                    corners is not None
                    and rvec is not None
                    and tvec is not None
                )

                cv2.drawChessboardCorners(
                    display,
                    PATTERN_SIZE,
                    corners,
                    found
                )

                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    50
                )

                dist_mm = float(
                    np.linalg.norm(tvec)
                )

                board_text = (
                    f"BOARD FOUND  dist={dist_mm:.0f}mm"
                )

                cv2.putText(
                    display,
                    board_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                dist_mm = None

                cv2.putText(
                    display,
                    "BOARD NOT DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # -------------------------------------------------------------
            # Robot information
            # -------------------------------------------------------------

            if robot_ok and T_robot is not None:

                x, y, z = T_robot[:3, 3]

                cv2.putText(
                    display,
                    f"TCP X={x:.1f} Y={y:.1f} Z={z:.1f} mm",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                if previous_T_robot is not None:

                    d_trans, d_rot = pose_difference(
                        previous_T_robot,
                        T_robot
                    )

                    cv2.putText(
                        display,
                        f"Delta: {d_trans:.1f}mm  "
                        f"{d_rot:.1f}deg",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                else:

                    d_trans = None
                    d_rot = None

                    cv2.putText(
                        display,
                        "Delta: FIRST POSE",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

            else:

                cv2.putText(
                    display,
                    "RTDE ERROR",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # -------------------------------------------------------------
            # Save information
            # -------------------------------------------------------------

            cv2.putText(
                display,
                f"Saved: {saved}/{MIN_POSES}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display,
                "[S] Save   [Q] Quit",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            draw_rtde_indicator(display, rtde_link)
            draw_cam_indicator(display, cam_link)

            cv2.imshow(
                win_name,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            # =============================================================
            # SAVE
            # =============================================================

            elif key == ord("s"):

                # ---------------------------------------------------------
                # Robot must be available
                # ---------------------------------------------------------

                if not robot_ok:

                    print(
                        "\n  [REJECTED]"
                    )
                    print(
                        "  RTDE robot pose is not available."
                    )
                    print(
                        f"  Reason: {robot_error}"
                    )
                    continue

                # ---------------------------------------------------------
                # Board must exist
                # ---------------------------------------------------------

                if not found:

                    print(
                        "\n  [REJECTED]"
                    )
                    print(
                        "  Checkerboard not detected."
                    )
                    continue

                # ---------------------------------------------------------
                # Robot movement check
                # ---------------------------------------------------------

                if previous_T_robot is not None:

                    d_trans, d_rot = pose_difference(
                        previous_T_robot,
                        T_robot
                    )

                    print()
                    print("  ----------------------------------------")
                    print("  Robot movement from previous saved pose")
                    print("  ----------------------------------------")
                    print(
                        f"  Translation : {d_trans:.2f} mm"
                    )
                    print(
                        f"  Rotation    : {d_rot:.2f} deg"
                    )

                    if (
                        d_trans < MIN_TRANSLATION_MM
                        and
                        d_rot < MIN_ROTATION_DEG
                    ):

                        print()
                        print(
                            "  [REJECTED] Robot movement is too small."
                        )
                        print(
                            "  Move the robot to a new pose."
                        )

                        continue

                # ---------------------------------------------------------
                # Save
                # ---------------------------------------------------------

                assert rvec is not None
                assert tvec is not None

                ts = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                img_path = os.path.join(
                    session_dir,
                    f"eih_{saved:03d}_{ts}.jpg"
                )

                imwrite_unicode(
                    img_path,
                    img
                )

                T_target2cam = rvec_tvec_to_T(
                    rvec,
                    tvec
                )

                if T_robot is None:
                    print("[REJECTED] Robot pose is None.")
                    continue

                # IMPORTANT:
                # This is the actual robot pose read at capture time.
                T_gripper2base = T_robot.copy()

                pose_path = os.path.join(
                    session_dir,
                    f"eih_{saved:03d}_{ts}.json"
                )

                data = {
                    "image_path": img_path,

                    "T_target2cam":
                        T_target2cam.tolist(),

                    "T_gripper2base":
                        T_gripper2base.tolist(),

                    "rvec":
                        rvec.tolist(),

                    "tvec":
                        tvec.tolist(),

                    "robot_tcp_pose_mm":
                        T_gripper2base[:3, 3].tolist(),

                    "board_distance_mm":
                        float(np.linalg.norm(tvec)),

                    "timestamp":
                        datetime.now().isoformat()
                }

                if previous_T_robot is not None:

                    data[
                        "delta_translation_mm"
                    ] = d_trans

                    data[
                        "delta_rotation_deg"
                    ] = d_rot

                with open(
                    pose_path,
                    "w"
                ) as f:

                    json.dump(
                        data,
                        f,
                        indent=2
                    )

                saved += 1

                previous_T_robot = T_gripper2base.copy()

                print()
                print(
                    f"  [SAVED] Pose {saved}"
                )

                print(
                    f"  Robot:"
                )

                print(
                    f"    X={T_robot[0,3]:.2f} mm"
                )
                print(
                    f"    Y={T_robot[1,3]:.2f} mm"
                )
                print(
                    f"    Z={T_robot[2,3]:.2f} mm"
                )

                print(
                    f"  Board distance:"
                    f" {np.linalg.norm(tvec):.2f} mm"
                )

    finally:

        cam_link.close()
        rtde_link.close()

        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(
        f"[COLLECT] {saved} poses saved."
    )
    print(
        f"Session: {session_dir}"
    )
    print("=" * 60)

    if saved < MIN_POSES:

        print(
            f"WARNING: only {saved} poses."
        )

# =============================================================================
# COLLECT PLUS
# =============================================================================

def collect_plus():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    session_dir = create_session_dir()

    print()
    print("=" * 70)
    print("  NEW EYE-IN-HAND COLLECTION PLUS SESSION")
    print("=" * 70)

    print(
        f"\n  Session:"
    )

    print(
        f"  {session_dir}"
    )

    # -----------------------------------------------------------------
    # Open camera / connect RTDE -- both fully backgrounded. Neither the
    # camera pipeline nor the RTDE socket is ever touched from the GUI
    # thread; each worker thread just keeps a cached latest value.
    # -----------------------------------------------------------------

    cam_link = CameraLink()
    rtde_link = RTDELink(ROBOT_IP)

    camera_matrix, dist_coeffs = get_camera_calibration_data(cam_link)

    objp = build_objp()

    print()
    print("=" * 70)
    print("  COLLECT PLUS WORKFLOW")
    print("=" * 70)

    print()
    print("  1. Move robot to the first REFERENCE pose.")
    print("  2. Make sure checkerboard is visible.")
    print("  3. Press S.")
    print("  4. Program creates the next recommended pose.")
    print("  5. Move robot manually to that pose.")
    print("  6. Press S again.")
    print("  7. Repeat.")
    print("  8. Press ESC to stop.")
    print()

    print("  IMPORTANT:")
    print("  All target poses are generated from")
    print("  the FIRST reference pose.")
    print()

    print("  No robot movement command is sent.")
    print("  RTDE is READ ONLY.")
    print()

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    reference_T = None

    target_T_list = []

    target_index = 0

    saved = 0

    win_name = (
        "Eye-in-Hand Collect PLUS | "
        "Orbbec Gemini 2 + UR"
    )

    cv2.namedWindow(
        win_name,
        cv2.WINDOW_NORMAL
    )

    try:

        while True:

            # ---------------- read cached camera frame ----------------
            img = cam_link.get_frame()

            if img is None:
                placeholder = make_placeholder(
                    960, 540,
                    [("Waiting for camera ...", (0, 165, 255))],
                )
                draw_rtde_indicator(placeholder, rtde_link)
                draw_cam_indicator(placeholder, cam_link)
                cv2.imshow(win_name, placeholder)
                key = cv2.waitKey(50) & 0xFF
                if key == 27:
                    print()
                    print("  [ESC] Collection stopped.")
                    break
                continue

            # =========================================================
            # READ ROBOT POSE (cached -- instant, never blocks)
            # =========================================================

            T_robot = rtde_link.get_pose_mm()
            robot_ok = T_robot is not None
            robot_error = "" if robot_ok else "RTDE not connected"

            # =========================================================
            # CHECKERBOARD
            # =========================================================

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            found, rvec, tvec, corners = (
                detect_board_pose(
                    gray,
                    objp,
                    camera_matrix,
                    dist_coeffs
                )
            )

            display = img.copy()

            # =========================================================
            # DRAW CHECKERBOARD
            # =========================================================

            if found:

                assert (
                    corners is not None
                    and rvec is not None
                    and tvec is not None
                )

                cv2.drawChessboardCorners(
                    display,
                    PATTERN_SIZE,
                    corners,
                    found
                )

                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    50
                )

                dist_mm = float(
                    np.linalg.norm(
                        tvec
                    )
                )

                cv2.putText(
                    display,
                    f"BOARD FOUND  "
                    f"dist={dist_mm:.0f}mm",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                dist_mm = None

                cv2.putText(
                    display,
                    "BOARD NOT DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # =========================================================
            # ROBOT CURRENT POSE
            # =========================================================

            if robot_ok and T_robot is not None:

                current_pose = T_to_ur_pose(
                    T_robot
                )

                x = current_pose[0]
                y = current_pose[1]
                z = current_pose[2]

                rx = float(current_pose[3])
                ry = float(current_pose[4])
                rz = float(current_pose[5])

                rx_deg = float(np.degrees(rx))
                ry_deg = float(np.degrees(ry))
                rz_deg = float(np.degrees(rz))

                cv2.putText(
                    display,
                    f"CURRENT "
                    f"X={x:.1f} "
                    f"Y={y:.1f} "
                    f"Z={z:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"RX={rx_deg:.3f} "
                    f"RY={ry_deg:.3f} "
                    f"RZ={rz_deg:.3f}",
                    (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

            else:

                current_pose = None

                cv2.putText(
                    display,
                    "RTDE ERROR",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display,
                    robot_error[:70],
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

            # =========================================================
            # TARGET DISPLAY
            # =========================================================

            if reference_T is None:

                # -----------------------------------------------------
                # FIRST POINT
                # -----------------------------------------------------

                cv2.putText(
                    display,
                    "REFERENCE POINT",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    "Move robot to desired "
                    "reference position",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    "[S] Save Reference",
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            else:

                # -----------------------------------------------------
                # NEXT TARGET
                # -----------------------------------------------------

                target_T = get_calibration_target(
                    reference_T,
                    target_index
                )

                target_pose = T_to_ur_pose(
                    target_T
                )

                tx = target_pose[0]
                ty = target_pose[1]
                tz = target_pose[2]

                trx = target_pose[3]
                try_ = target_pose[4]
                trz = target_pose[5]

                trx_deg = np.degrees(trx)
                try_deg = np.degrees(try_)
                trz_deg = np.degrees(trz)

                cv2.putText(
                    display,
                    f"TARGET {target_index + 1}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"X={tx:.1f} Y={ty:.1f} Z={tz:.1f}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"RX={trx_deg:.3f} RY={try_deg:.3f} RZ={trz_deg:.3f}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # -------------------------------------------------
                # Difference between actual and target
                # -------------------------------------------------

                if T_robot is not None:

                    target_d_trans, target_d_rot = pose_difference(
                        T_robot,
                        target_T
                    )

                    cv2.putText(
                        display,
                        f"TARGET ERROR: {target_d_trans:.1f}mm {target_d_rot:.1f}deg",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2
                    )

            # =========================================================
            # SAVE COUNTER
            # =========================================================

            cv2.putText(
                display,
                f"SAVED: {saved}/{MIN_POSES}",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "[S] Save    [ESC] Stop",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            draw_rtde_indicator(display, rtde_link)
            draw_cam_indicator(display, cam_link)

            cv2.imshow(
                win_name,
                display
            )

            # =========================================================
            # KEY
            # =========================================================

            key = cv2.waitKey(1) & 0xFF

            # ---------------------------------------------------------
            # ESC
            # ---------------------------------------------------------

            if key == 27:

                print()
                print(
                    "  [ESC] Collection stopped."
                )

                break

            # ---------------------------------------------------------
            # SAVE
            # ---------------------------------------------------------

            elif key == ord("s"):

                # =====================================================
                # ROBOT CHECK
                # =====================================================

                if not robot_ok or T_robot is None:
                    # RTDELink reconnects automatically in the background --
                    # just check again in case a connection just came up.
                    T_robot = rtde_link.get_pose_mm()
                    robot_ok = T_robot is not None
                    if not robot_ok:
                        robot_error = "RTDE not connected"

                if not robot_ok or T_robot is None:

                    print()
                    print(
                        "  [REJECTED]"
                    )

                    print(
                        "  Robot pose unavailable."
                    )

                    if robot_error:
                        print(
                            f"  Reason: {robot_error}"
                        )

                    continue

                # =====================================================
                # CHECKERBOARD CHECK
                # =====================================================

                if not found:

                    print()
                    print(
                        "  [REJECTED]"
                    )

                    print(
                        "  Checkerboard not detected."
                    )

                    continue

                # =====================================================
                # FIRST REFERENCE
                # =====================================================

                if reference_T is None:

                    reference_T = T_robot.copy()

                    target_index = 0

                    # -------------------------------------------------
                    # Save first point
                    # -------------------------------------------------

                    assert rvec is not None
                    assert tvec is not None

                    ts = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    img_path = os.path.join(
                        session_dir,
                        f"eih_plus_{saved:03d}_"
                        f"{ts}.jpg"
                    )

                    imwrite_unicode(
                        img_path,
                        img
                    )

                    T_target2cam = (
                        rvec_tvec_to_T(
                            rvec,
                            tvec
                        )
                    )

                    pose_path = os.path.join(
                        session_dir,
                        f"eih_plus_{saved:03d}_"
                        f"{ts}.json"
                    )

                    robot_pose = T_to_ur_pose(
                        T_robot
                    )

                    data = {

                        "point_type":
                            "REFERENCE",

                        "point_index":
                            saved,

                        "image_path":
                            img_path,

                        "T_target2cam":
                            T_target2cam.tolist(),

                        "T_gripper2base":
                            T_robot.tolist(),

                        "rvec":
                            rvec.tolist(),

                        "tvec":
                            tvec.tolist(),

                        "robot_tcp_pose_mm":
                            T_robot[:3, 3].tolist(),

                        "robot_tcp_pose_ur":
                            robot_pose.tolist(),

                        "board_distance_mm":
                            float(
                                np.linalg.norm(
                                    tvec
                                )
                            ),

                        "timestamp":
                            datetime.now().isoformat()
                    }

                    with open(
                        pose_path,
                        "w"
                    ) as f:

                        json.dump(
                            data,
                            f,
                            indent=2
                        )

                    saved += 1

                    print()
                    print("=" * 60)
                    print(
                        "  [REFERENCE SAVED]"
                    )
                    print("=" * 60)

                    print(
                        f"  X  = {robot_pose[0]:.2f} mm"
                    )

                    print(
                        f"  Y  = {robot_pose[1]:.2f} mm"
                    )

                    print(
                        f"  Z  = {robot_pose[2]:.2f} mm"
                    )

                    print(
                        f"  RX = {np.degrees(robot_pose[3]):.2f} deg"
                    )

                    print(
                        f"  RY = {np.degrees(robot_pose[4]):.2f} deg"
                    )

                    print(
                        f"  RZ = {np.degrees(robot_pose[5]):.2f} deg"
                    )

                    print()
                    print(
                        "  Generating unlimited dynamic target poses..."
                    )

                    print()

                    first_target_T = get_calibration_target(
                        reference_T,
                        0
                    )

                    next_pose = T_to_ur_pose(first_target_T)

                    print(
                        "  NEXT TARGET 1:"
                    )

                    print(
                        f"    X  = {next_pose[0]:.2f} mm"
                    )

                    print(
                        f"    Y  = {next_pose[1]:.2f} mm"
                    )

                    print(
                        f"    Z  = {next_pose[2]:.2f} mm"
                    )

                    print(
                        f"    RX = {np.degrees(next_pose[3]):.2f} deg"
                    )

                    print(
                        f"    RY = {np.degrees(next_pose[4]):.2f} deg"
                    )

                    print(
                        f"    RZ = {np.degrees(next_pose[5]):.2f} deg"
                    )

                    continue

                # =====================================================
                # CHECK ACTUAL POSITION & TARGET ERROR
                # =====================================================

                target_T = get_calibration_target(
                    reference_T,
                    target_index
                )

                d_trans, d_rot = pose_difference(
                    T_robot,
                    target_T
                )

                print()
                print(
                    "  ----------------------------------------"
                )

                print(
                    f"  Target {target_index + 1} verification"
                )

                print(
                    "  ----------------------------------------"
                )

                print(
                    f"  Position error : "
                    f"{d_trans:.2f} mm"
                )

                print(
                    f"  Rotation error : "
                    f"{d_rot:.2f} deg"
                )

                TARGET_TRANSLATION_TOL_MM = 15.0
                TARGET_ROTATION_TOL_DEG = 5.0

                if (
                    d_trans > TARGET_TRANSLATION_TOL_MM
                    or
                    d_rot > TARGET_ROTATION_TOL_DEG
                ):
                    print()
                    print(
                        "  [NOTE] Target error exceeds standard tolerance, but saving anyway."
                    )
                    print(
                        f"  Tolerance: Trans <= {TARGET_TRANSLATION_TOL_MM:.1f} mm, Rot <= {TARGET_ROTATION_TOL_DEG:.1f} deg"
                    )

                # =====================================================
                # SAVE IMAGE
                # =====================================================

                assert rvec is not None
                assert tvec is not None

                ts = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                img_path = os.path.join(
                    session_dir,
                    f"eih_plus_{saved:03d}_"
                    f"{ts}.jpg"
                )

                imwrite_unicode(
                    img_path,
                    img
                )

                # =====================================================
                # SAVE TRANSFORM
                # =====================================================

                T_target2cam = (
                    rvec_tvec_to_T(
                        rvec,
                        tvec
                    )
                )

                # =====================================================
                # SAVE JSON
                # =====================================================

                pose_path = os.path.join(
                    session_dir,
                    f"eih_plus_{saved:03d}_"
                    f"{ts}.json"
                )

                robot_pose = T_to_ur_pose(
                    T_robot
                )

                target_pose_val = T_to_ur_pose(target_T).tolist()

                data = {

                    "point_type":
                        "CALIBRATION",

                    "point_index":
                        saved,

                    "target_index":
                        target_index + 1,

                    "image_path":
                        img_path,

                    "T_target2cam":
                        T_target2cam.tolist(),

                    "T_gripper2base":
                        T_robot.tolist(),

                    "rvec":
                        rvec.tolist(),

                    "tvec":
                        tvec.tolist(),

                    "robot_tcp_pose_mm":
                        T_robot[:3, 3].tolist(),

                    "robot_tcp_pose_ur":
                        robot_pose.tolist(),

                    "target_pose_ur":
                        target_pose_val,

                    "target_error_translation_mm":
                        float(d_trans),

                    "target_error_rotation_deg":
                        float(d_rot),

                    "board_distance_mm":
                        float(
                            np.linalg.norm(
                                tvec
                            )
                        ),

                    "timestamp":
                        datetime.now().isoformat()
                }

                with open(
                    pose_path,
                    "w"
                ) as f:

                    json.dump(
                        data,
                        f,
                        indent=2
                    )

                # =====================================================
                # NEXT
                # =====================================================

                saved += 1

                target_index += 1

                print()
                print(
                    "  [SAVED]"
                )

                print(
                    f"  Calibration target point "
                    f"{target_index}"
                )

                print(
                    f"  Actual:"
                )

                print(
                    f"    X  = "
                    f"{robot_pose[0]:.2f} mm"
                )

                print(
                    f"    Y  = "
                    f"{robot_pose[1]:.2f} mm"
                )

                print(
                    f"    Z  = "
                    f"{robot_pose[2]:.2f} mm"
                )

                print(
                    f"    RX = "
                    f"{np.degrees(robot_pose[3]):.2f} deg"
                )

                print(
                    f"    RY = "
                    f"{np.degrees(robot_pose[4]):.2f} deg"
                )

                print(
                    f"    RZ = "
                    f"{np.degrees(robot_pose[5]):.2f} deg"
                )

                # -----------------------------------------------------
                # Show next target in console
                # -----------------------------------------------------

                next_T = get_calibration_target(
                    reference_T,
                    target_index
                )

                next_pose = T_to_ur_pose(
                    next_T
                )

                print()
                print(
                    f"  NEXT TARGET {target_index + 1}:"
                )

                print(
                    f"    X  = "
                    f"{next_pose[0]:.2f} mm"
                )

                print(
                    f"    Y  = "
                    f"{next_pose[1]:.2f} mm"
                )

                print(
                    f"    Z  = "
                    f"{next_pose[2]:.2f} mm"
                )

                print(
                    f"    RX = {np.degrees(next_pose[3]):.2f} deg"
                )

                print(
                    f"    RY = {np.degrees(next_pose[4]):.2f} deg"
                )

                print(
                    f"    RZ = {np.degrees(next_pose[5]):.2f} deg"
                )

    finally:

        cam_link.close()

        rtde_link.close()

        cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print(
        f"  [COLLECT PLUS] "
        f"{saved} poses saved."
    )

    print(
        f"  Session:"
        f" {session_dir}"
    )

    print("=" * 70)

    return session_dir


# =============================================================================
# VALIDATE POSES
# =============================================================================

def validate_and_filter_poses(
    T_g2b_list,
    T_t2c_list
):

    print()
    print("=" * 70)
    print("  POSE DATA VALIDATION")
    print("=" * 70)

    valid_indices = []

    for i in range(len(T_g2b_list)):

        Rg = T_g2b_list[i][:3, :3]
        Rt = T_t2c_list[i][:3, :3]

        det_g = np.linalg.det(Rg)
        det_t = np.linalg.det(Rt)

        if abs(det_g - 1.0) > 0.01:

            print(
                f"  Pose {i+1}: BAD robot rotation"
            )

            continue

        if abs(det_t - 1.0) > 0.01:

            print(
                f"  Pose {i+1}: BAD target rotation"
            )

            continue

        valid_indices.append(i)

    # ---------------------------------------------------------------------
    # Remove duplicate robot poses
    # ---------------------------------------------------------------------

    unique_indices = []

    for idx in valid_indices:

        if not unique_indices:

            unique_indices.append(idx)
            continue

        last_idx = unique_indices[-1]

        d_trans, d_rot = pose_difference(
            T_g2b_list[last_idx],
            T_g2b_list[idx]
        )

        if (
            d_trans < DUP_TRANSLATION_MM
            and
            d_rot < DUP_ROTATION_DEG
        ):

            print(
                f"  Pose {idx+1}: DUPLICATE "
                f"({d_trans:.2f} mm, {d_rot:.2f} deg)"
            )

        else:

            unique_indices.append(idx)

    print()
    print(
        f"  Original poses : {len(T_g2b_list)}"
    )

    print(
        f"  Valid poses    : {len(valid_indices)}"
    )

    print(
        f"  Unique poses   : {len(unique_indices)}"
    )

    # ---------------------------------------------------------------------
    # Calculate rotational span
    # ---------------------------------------------------------------------

    max_rotation = 0.0

    for a in range(len(unique_indices)):

        for b in range(a + 1, len(unique_indices)):

            i = unique_indices[a]
            j = unique_indices[b]

            angle = rotation_difference_deg(
                T_g2b_list[i][:3, :3],
                T_g2b_list[j][:3, :3]
            )

            max_rotation = max(
                max_rotation,
                angle
            )

    print(
        f"  Rotation span  : {max_rotation:.2f} deg"
    )

    # ---------------------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------------------

    if len(unique_indices) < MIN_POSES:

        print()
        print(
            "  WARNING:"
        )
        print(
            f"  Only {len(unique_indices)} unique poses."
        )
        print(
            f"  Recommended: at least {MIN_POSES}."
        )

    if max_rotation < MIN_ROTATION_SPAN_DEG:

        print()
        print(
            "  WARNING:"
        )
        print(
            f"  Rotation span is only "
            f"{max_rotation:.2f} deg."
        )

        print(
            f"  Recommended: > "
            f"{MIN_ROTATION_SPAN_DEG:.1f} deg."
        )

    return unique_indices, max_rotation


# =============================================================================
# CALIBRATE
# =============================================================================

def calibrate():

    session_dir = get_latest_session()

    if session_dir is None:

        raise FileNotFoundError(
            "No collection session found. "
            "Run collect first."
        )

    pose_files = sorted(
        glob.glob(
            os.path.join(
                session_dir,
                "*.json"
            )
        )
    )

    if len(pose_files) < 3:

        raise RuntimeError(
            f"Only {len(pose_files)} poses found."
        )

    print()
    print("=" * 70)
    print("  CALIBRATE LATEST SESSION")
    print("=" * 70)

    print(
        f"\n  Session: {session_dir}"
    )

    print(
        f"  Found: {len(pose_files)} poses"
    )

    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []

    T_g2b_list = []
    T_t2c_list = []

    for path in pose_files:

        with open(path) as f:
            d = json.load(f)

        T_g2b = np.asarray(
            d["T_gripper2base"],
            dtype=np.float64
        )

        T_t2c = np.asarray(
            d["T_target2cam"],
            dtype=np.float64
        )

        R_gripper2base.append(
            T_g2b[:3, :3]
        )

        t_gripper2base.append(
            T_g2b[:3, 3].reshape(3, 1)
        )

        R_target2cam.append(
            T_t2c[:3, :3]
        )

        t_target2cam.append(
            T_t2c[:3, 3].reshape(3, 1)
        )

        T_g2b_list.append(T_g2b)
        T_t2c_list.append(T_t2c)

    # ---------------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------------

    unique_indices, rotation_span = (
        validate_and_filter_poses(
            T_g2b_list,
            T_t2c_list
        )
    )

    if len(unique_indices) < 10:

        raise RuntimeError(
            "Too few unique robot poses. "
            "Collect new data."
        )

    if rotation_span < MIN_ROTATION_SPAN_DEG:

        raise RuntimeError(
            "Insufficient rotational diversity. "
            "Move the robot wrist with larger orientation changes."
        )

    # ---------------------------------------------------------------------
    # Apply filter
    # ---------------------------------------------------------------------

    Rg = [
        R_gripper2base[i]
        for i in unique_indices
    ]

    tg = [
        t_gripper2base[i]
        for i in unique_indices
    ]

    Rt = [
        R_target2cam[i]
        for i in unique_indices
    ]

    tt = [
        t_target2cam[i]
        for i in unique_indices
    ]

    Tg = [
        T_g2b_list[i]
        for i in unique_indices
    ]

    Tt = [
        T_t2c_list[i]
        for i in unique_indices
    ]

    # ---------------------------------------------------------------------
    # Save raw arrays
    # ---------------------------------------------------------------------

    np.save(
        os.path.join(
            session_dir,
            "R_gripper2base.npy"
        ),
        np.asarray(Rg)
    )

    np.save(
        os.path.join(
            session_dir,
            "t_gripper2base.npy"
        ),
        np.asarray(tg)
    )

    np.save(
        os.path.join(
            session_dir,
            "R_target2cam.npy"
        ),
        np.asarray(Rt)
    )

    np.save(
        os.path.join(
            session_dir,
            "t_target2cam.npy"
        ),
        np.asarray(tt)
    )

    # ---------------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------------

    methods = {

        "TSAI":
            cv2.CALIB_HAND_EYE_TSAI,

        "PARK":
            cv2.CALIB_HAND_EYE_PARK,

        "HORAUD":
            cv2.CALIB_HAND_EYE_HORAUD,

        "ANDREFF":
            cv2.CALIB_HAND_EYE_ANDREFF,

        "DANIILIDIS":
            cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best_method = None
    best_error = float("inf")
    best_T = None

    results = {}

    print()
    print(
        "  Method         Rotation err (deg) "
        " Translation err (mm)"
    )

    print(
        "  " + "-" * 55
    )

    for name, method in methods.items():

        try:

            R_c2g, t_c2g = cv2.calibrateHandEye(
                Rg,
                tg,
                Rt,
                tt,
                method=method
            )

            T_cam2gripper = np.eye(4)

            T_cam2gripper[:3, :3] = R_c2g
            T_cam2gripper[:3, 3] = t_c2g.flatten()

            rot_err, trans_err = _compute_error(
                T_cam2gripper,
                Tg,
                Tt
            )

            total = (
                rot_err +
                trans_err * 0.01
            )

            results[name] = {
                "T_cam_to_gripper":
                    T_cam2gripper,

                "rot_err_deg":
                    rot_err,

                "trans_err_mm":
                    trans_err
            }

            print(
                f"  {name:<16}"
                f"{rot_err:>8.4f} deg"
                f"      {trans_err:>8.4f} mm"
            )

            if total < best_error:

                best_error = total
                best_method = name
                best_T = T_cam2gripper

        except Exception as e:

            print(
                f"  {name:<16} FAILED: {e}"
            )

    if best_T is None:

        raise RuntimeError(
            "All calibration methods failed."
        )

    print(
        f"\n  Best method: {best_method}"
    )

    _print_result(
        best_T,
        best_method,
        results[best_method]["rot_err_deg"],
        results[best_method]["trans_err_mm"]
    )

    # ---------------------------------------------------------------------
    # Save result
    # ---------------------------------------------------------------------

    np.savez(
        RESULT_FILE,

        T_cam_to_gripper=best_T,

        best_method=np.array(
            [best_method]
        ),

        rot_error_deg=np.array([
            results[best_method]["rot_err_deg"]
        ]),

        trans_error_mm=np.array([
            results[best_method]["trans_err_mm"]
        ]),

        n_poses=np.array([
            len(unique_indices)
        ]),

        rotation_span_deg=np.array([
            rotation_span
        ])
    )

    report = {

        "camera_model":
            "Orbbec Gemini 2",

        "robot_model":
            "UR12e",

        "type":
            "Eye-in-Hand",

        "session":
            session_dir,

        "date":
            datetime.now().isoformat(),

        "original_poses":
            len(pose_files),

        "unique_poses":
            len(unique_indices),

        "rotation_span_deg":
            rotation_span,

        "best_method":
            best_method,

        "rot_error_deg":
            results[best_method]["rot_err_deg"],

        "trans_error_mm":
            results[best_method]["trans_err_mm"],

        "T_cam_to_gripper":
            best_T.tolist()
    }

    with open(
        REPORT_FILE,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"\n  Saved: {RESULT_FILE}"
    )

    print(
        f"  Saved: {REPORT_FILE}"
    )

    print(
        f"  Raw arrays saved in: {session_dir}"
    )

    return best_T


# =============================================================================
# ERROR
# =============================================================================

def _compute_error(
    X,
    T_g2b_list,
    T_t2c_list
):

    rot_errs = []
    trans_errs = []

    n = len(T_g2b_list)

    for i in range(n):

        for j in range(i + 1, n):

            A = (
                np.linalg.inv(T_g2b_list[j])
                @
                T_g2b_list[i]
            )

            B = (
                T_t2c_list[j]
                @
                np.linalg.inv(T_t2c_list[i])
            )

            AX = A @ X
            XB = X @ B

            R_err = (
                AX[:3, :3]
                @
                XB[:3, :3].T
            )

            value = (
                (np.trace(R_err) - 1)
                / 2
            )

            value = np.clip(
                value,
                -1,
                1
            )

            ang = np.degrees(
                np.arccos(value)
            )

            rot_errs.append(
                abs(ang)
            )

            trans_errs.append(
                np.linalg.norm(
                    AX[:3, 3]
                    -
                    XB[:3, 3]
                )
            )

    return (
        float(np.mean(rot_errs)),
        float(np.mean(trans_errs))
    )


# =============================================================================
# PRINT RESULT
# =============================================================================

def _print_result(
    T,
    method,
    rot_err,
    trans_err
):

    t = T[:3, 3]

    print()
    print("=" * 62)

    print(
        "  EYE-IN-HAND RESULT"
    )

    print(
        "  T_cam_to_gripper"
    )

    print(
        "  Camera: Orbbec Gemini 2"
        "   Robot: UR12e"
    )

    print("=" * 62)

    print(
        f"\n  Method: {method}\n"
    )

    print(
        "  Matrix:"
    )

    for row in T:

        print(
            "    "
            +
            "  ".join(
                f"{v:+10.5f}"
                for v in row
            )
        )

    print(
        f"\n  Translation (mm): "
        f"X={t[0]:.2f} "
        f"Y={t[1]:.2f} "
        f"Z={t[2]:.2f}"
    )

    print(
        "\n  Consistency error:"
    )

    print(
        f"    Rotation    : "
        f"{rot_err:.4f} deg"
    )

    print(
        f"    Translation : "
        f"{trans_err:.4f} mm"
    )

    qual = (
        "Very good"
        if rot_err < 1.0
        and trans_err < 2.0

        else "Good"
        if rot_err < 2.0
        and trans_err < 5.0

        else "Usable"
        if rot_err < 3.0

        else "Redo — error too high"
    )

    print(
        f"\n  Assessment: {qual}"
    )

    print("=" * 62)


# =============================================================================
# VERIFY
# =============================================================================

def verify():

    if not os.path.exists(
        RESULT_FILE
    ):

        raise FileNotFoundError(
            "Run calibrate first."
        )

    data = np.load(
        RESULT_FILE,
        allow_pickle=True
    )

    T_cam2gripper = data[
        "T_cam_to_gripper"
    ]

    method = str(
        data["best_method"][0]
    )

    rot_err = float(
        data["rot_error_deg"][0]
    )

    trans_err = float(
        data["trans_error_mm"][0]
    )

    _print_result(
        T_cam2gripper,
        method,
        rot_err,
        trans_err
    )

    session_dir = get_latest_session()

    if session_dir is None:
        return

    pose_files = sorted(
        glob.glob(
            os.path.join(
                session_dir,
                "*.json"
            )
        )
    )

    positions = []

    print()
    print(
        "  Board position in BASE frame:"
    )

    for i, path in enumerate(
        pose_files
    ):

        with open(path) as f:
            d = json.load(f)

        T_g2b = np.asarray(
            d["T_gripper2base"]
        )

        T_t2c = np.asarray(
            d["T_target2cam"]
        )

        p_target_cam = np.append(
            T_t2c[:3, 3],
            1.0
        )

        p_target_base = (
            T_g2b
            @
            T_cam2gripper
            @
            p_target_cam
        )[:3]

        positions.append(
            p_target_base
        )

        print(
            f"  {i+1:02d}: "
            f"X={p_target_base[0]:8.1f} "
            f"Y={p_target_base[1]:8.1f} "
            f"Z={p_target_base[2]:8.1f}"
        )

    positions = np.asarray(
        positions
    )

    std = positions.std(
        axis=0
    )

    total_std = np.linalg.norm(
        std
    )

    print()
    print(
        f"  Std X = {std[0]:.2f} mm"
    )

    print(
        f"  Std Y = {std[1]:.2f} mm"
    )

    print(
        f"  Std Z = {std[2]:.2f} mm"
    )

    print(
        f"  Total = {total_std:.2f} mm"
    )

    if total_std < 5:

        print(
            "  -> GOOD"
        )

    elif total_std < 15:

        print(
            "  -> ACCEPTABLE"
        )

    else:

        print(
            "  -> REDO CALIBRATION"
        )


# =============================================================================
# EXPORT
# =============================================================================

def export():

    if not os.path.exists(
        RESULT_FILE
    ):

        raise FileNotFoundError(
            "Run calibrate first."
        )

    data = np.load(
        RESULT_FILE,
        allow_pickle=True
    )

    T_cam2gripper = data[
        "T_cam_to_gripper"
    ]

    np.save(
        EXPORT_FILE,
        T_cam2gripper
    )

    print(
        f"\n[EXPORT] Saved: "
        f"{EXPORT_FILE}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    global ROBOT_IP

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ip",
        default=ROBOT_IP
    )

    args = parser.parse_args()

    ROBOT_IP = args.ip

    commands = {
        "collect": collect,
        "collect_plus": collect_plus,
        "calibrate": calibrate,
        "verify": verify,
        "export": export
    }

    print()
    print("=" * 62)

    print(
        "  Eye-in-Hand Calibration"
    )

    print(
        "  Robot IP:",
        ROBOT_IP
    )

    print("=" * 62)

    print(
        "\n  Commands:"
    )

    print(
        "  collect"
    )

    print(
    "  collect_plus"
    )

    print(
        "  calibrate"
    )

    print(
        "  verify"
    )

    print(
        "  export"
    )

    print(
        "  all"
    )

    print(
        "  quit\n"
    )

    while True:

        cmd = input(
            "Enter command > "
        ).strip().lower()

        if cmd in (
            "quit",
            "exit",
            "q"
        ):

            break

        elif cmd == "all":

            collect()
            calibrate()
            verify()
            export()

        elif cmd in commands:

            try:

                commands[cmd]()

            except Exception as e:

                print(
                    f"\n[ERROR] {e}"
                )

        elif cmd == "":

            continue

        else:

            print(
                "Unknown command."
            )


if __name__ == "__main__":
    main()