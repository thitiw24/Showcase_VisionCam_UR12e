#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Knom_Buabarn - Main Industrial Robot Vision & Cobot Pick-and-Place Service
=============================================================================
Unified Production Release (Poom_Windows GUI + Industrial Hardened Safety)

Engineered Features:
- Case 1: Fail-Closed RTDE Binding (Never return identity matrix),
          3D Safety Envelope Interlock (Z: -60.0mm to +80.0mm),
          Universal Blackbox Rotating File Logger.
- Case 2: Corrupted Depth Buffer Guard (Byte verification W*H*2), Skip frame policy A.
- Case 5: Camera Auto-Recovery (Probe every 3s), Firmware Soft-Reboot, Live GUI Reconnect.
- Case 6: YOLO Model Selection (.pt) via GUI FileDialog, Default Data/best.pt,
          Automatic Fallback Guard, Thread-Safe Hot-Swap, Dynamic Class Resolver.
- Extension Ready (Case 3 & 4): Clean Single TCP Send with explicit hooks and
          guidance for activating 2-Way Handshake and OS Keep-Alive in the future.
"""

from __future__ import annotations

import os
import sys
import time
import json
import queue
import socket
import threading
import subprocess
import gc
from datetime import datetime
from collections import deque
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import cv2 as cv

# GUI Libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# YOLO Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    Results = None
    YOLO_AVAILABLE = False

# Robot RTDE
try:
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    rtde_receive = None
    RTDE_AVAILABLE = False

# Orbbec 3D Camera SDK
try:
    from pyorbbecsdk import (
        Context,
        Pipeline,
        Config,
        OBSensorType,
        OBAlignMode,
    )
    from utils import (
        safe_frame_to_bgr_image,
        get_valid_depth_image,
        get_depth_median_at,
    )
    ORBBEC_AVAILABLE = True
except ImportError:
    Context = Pipeline = Config = OBSensorType = OBAlignMode = None
    safe_frame_to_bgr_image = None
    get_valid_depth_image = None
    get_depth_median_at = None
    ORBBEC_AVAILABLE = False


# =============================================================================
# 1. SYSTEM CONFIGURATION & SAFETY BOUNDARIES (PROGRAMMER ADJUSTABLE)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
LOGS_DIR = BASE_DIR / "logs"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
CONFIG_FILE = BASE_DIR / "gui_config.json"
CSV_FILE = BASE_DIR / "cycle_log.csv"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_FILE = DATA_DIR / "best.pt"
CALIB_PATH = DATA_DIR / "calibration.npz"
HANDEYE_PATH = DATA_DIR / "T_cam_to_gripper.npz"

# -----------------------------------------------------------------------------
# CASE 1: ROBOT SAFETY & WORKSPACE BOUNDARY CONFIG
# -----------------------------------------------------------------------------
ROBOT_SAFETY_CONFIG = {
    # Robot Base Frame Z Limits (mm) - อ้างอิงจากสถิติชิ้นงานจริง 76 รอบ (-44mm ถึง +30mm)
    "Z_MIN_MM": -60.0,   # ขีดจำกัดล่างสุด: ต่ำกว่าผิวสายพาน (ห้ามลงต่ำกว่านี้เด็ดขาด ป้องกันชนสายพาน)
    "Z_MAX_MM": 80.0,    # ขีดจำกัดบนสุด: ห้ามหยิบสูงเกินจริง (ป้องกันดูดลมหรือพิกัดลอย)

    # Robot Base Frame X, Y Limits (mm) - กรอบขอบเขตสายพานลำเลียง
    "X_MIN_MM": -320.0,  # แกน X ฝั่งไกล
    "X_MAX_MM": 20.0,    # แกน X ฝั่งใกล้
    "Y_MIN_MM": -1020.0, # แกน Y ฝั่งซ้าย (โซน EGG ~ -950mm)
    "Y_MAX_MM": -400.0,  # แกน Y ฝั่งขวา (โซน CARTON ~ -480mm)

    # Strict IP Binding: ห้ามเชื่อมต่อไปยัง IP อื่นเองโดยเด็ดขาด
    "TARGET_ROBOT_IP": "10.42.5.151",
    "PC_IP": "10.42.5.50",
    "RTDE_PORT": 30004,
    "RTDE_RETRY_INTERVAL_S": 3.0,
}

# -----------------------------------------------------------------------------
# CASE 2 & 5: CAMERA STREAM & AUTO-RECOVERY CONFIG
# -----------------------------------------------------------------------------
CAMERA_CONFIG = {
    "DEPTH_WIDTH": 1280,
    "DEPTH_HEIGHT": 800,
    "COLOR_WIDTH": 1280,
    "COLOR_HEIGHT": 720,
    "RETRY_INTERVAL_S": 3.0,
    "FRAME_STALL_TIMEOUT_S": 3.0,  # No valid frame this long -> treat camera as disconnected
    "MEDIAN_SAMPLE_RADIUS": 4,
}

# -----------------------------------------------------------------------------
# TCP NETWORK SETTINGS & FUTURE EXTENSION (CASE 3 & 4 HOOKS)
# -----------------------------------------------------------------------------
DEFAULT_TCP_PORT = 5000

# หมายเหตุสำหรับโปรแกรมเมอร์ในอนาคต:
# หากต้องการเปิดระบบ 2-Way Handshake (TARGET -> ACK -> EXECUTE) และ Keep-Alive:
# ให้ปรับ ENABLE_2WAY_HANDSHAKE = True และดูฟังก์ชัน _handle_tcp_handshake_execute()
ENABLE_2WAY_HANDSHAKE = False
ENABLE_TCP_KEEPALIVE = False

# -----------------------------------------------------------------------------
# CASE 6: WORKPIECE TARGET CLASSES (ปรับเปลี่ยนเป้าหมายกล่องอาหาร/ไข่ไก่ได้ที่นี่)
# -----------------------------------------------------------------------------
# โครงสร้างคลาสในโมเดล YOLO (best.pt):
#   ID 0: Egg       (บรรจุภัณฑ์ไข่ไก่สด)
#   ID 1: BlueBox   (กล่องอาหารสำเร็จรูปสีฟ้า)
#   ID 2: RedBox    (กล่องอาหารสำเร็จรูปสีแดง)
#
# การตั้งค่าสำหรับคำสั่ง "CARTON" จากหุ่นยนต์:
# - ต้องการจับทั้งกล่องฟ้าและกล่องแดง: [1, 2]  (ค่ามาตรฐานเริ่มต้น)
# - ต้องการจับเฉพาะกล่องแดง (RedBox): [2]      <--- ปรับตรงนี้หากต้องการเฉพาะกล่องแดง
# - ต้องการจับเฉพาะกล่องฟ้า (BlueBox): [1]
CARTON_TARGET_CLASSES = [1, 2]
EGG_TARGET_CLASSES = [0]

# UI Colors & Theme (Modern Poom_Windows Dark)
BG_MAIN       = "#09090b"
BG_SURFACE    = "#18181b"
BG_SURFACE2   = "#27272a"
BORDER_COL    = "#3f3f46"
TEXT_MAIN     = "#f4f4f5"
TEXT_MUTED    = "#a1a1aa"
ACCENT_GREEN  = "#22c55e"
ACCENT_RED    = "#ef4444"
ACCENT_BLUE   = "#3b82f6"
ACCENT_WARN   = "#f59e0b"
ACCENT_PURPLE = "#a855f7"

# Operational States
WAIT_START  = "WAIT_START"
WAIT_CLASS  = "WAIT_CLASS"
WAIT_TARGET = "WAIT_TARGET"
WAIT_DONE   = "WAIT_DONE"


# =============================================================================
# 2. UNIVERSAL BLACKBOX ROTATING LOGGER (CASE 1)
# =============================================================================
def setup_blackbox_logger(log_dir: Path) -> logging.Logger:
    """Industrial-Grade Rotating Logger: logs everything to disk with milliseconds."""
    logger = logging.getLogger("KnomBuabarn")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File Handler: 10MB per file, keeps 10 backups
        log_file = log_dir / f"blackbox_{datetime.now().strftime('%Y%m%d')}.log"
        file_h = RotatingFileHandler(str(log_file), maxBytes=10*1024*1024, backupCount=10, encoding="utf-8")
        file_fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_h.setFormatter(file_fmt)
        logger.addHandler(file_h)

        # Console Handler
        console_h = logging.StreamHandler(sys.stdout)
        console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        console_h.setFormatter(console_fmt)
        logger.addHandler(console_h)

    return logger

logger = setup_blackbox_logger(LOGS_DIR)


# =============================================================================
# 3. SAFETY ENVELOPE & FAIL-CLOSED RTDE (CASE 1)
# =============================================================================
def check_safety_envelope(target_xyz: np.ndarray | list) -> tuple[bool, str]:
    """
    ตรวจจับขอบเขตพิกัด 3 มิติทางกายภาพ ป้องกันหุ่นยนต์ชนสายพาน 100%
    Returns: (is_safe: bool, reason: str)
    """
    if target_xyz is None or len(target_xyz) < 3:
        return False, "Target coordinate is None"

    x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    cfg = ROBOT_SAFETY_CONFIG

    # ตรวจสอบขอบเขตความสูง Z
    if not (cfg["Z_MIN_MM"] <= z <= cfg["Z_MAX_MM"]):
        return False, f"Z ละเมิดขอบเขตความปลอดภัย ({z:.1f}mm ไม่อยู่ในช่วง [{cfg['Z_MIN_MM']}, {cfg['Z_MAX_MM']}])"

    # ตรวจสอบขอบเขตระนาบสายพาน X, Y
    if not (cfg["X_MIN_MM"] <= x <= cfg["X_MAX_MM"]):
        return False, f"X ละเมิดขอบเขตสายพาน ({x:.1f}mm ไม่อยู่ในช่วง [{cfg['X_MIN_MM']}, {cfg['X_MAX_MM']}])"

    if not (cfg["Y_MIN_MM"] <= y <= cfg["Y_MAX_MM"]):
        return False, f"Y ละเมิดขอบเขตสายพาน ({y:.1f}mm ไม่อยู่ในช่วง [{cfg['Y_MIN_MM']}, {cfg['Y_MAX_MM']}])"

    return True, "SAFE"


def get_tcp_pose_mm(rtde_r) -> Optional[np.ndarray]:
    """
    อ่าน TCP Pose จริงจากหุ่นยนต์ผ่าน RTDE (Fail-Closed)
    ห้ามคืนค่า np.eye(4) เด็ดขาด เพื่อป้องกันการเอาพิกัดกริปเปอร์ไปสั่งขยับ Base
    """
    if rtde_r is None:
        return None
    try:
        if hasattr(rtde_r, "isConnected") and not rtde_r.isConnected():
            return None
        pose = rtde_r.getActualTCPPose()
        if not pose or len(pose) < 6:
            return None
        R, _ = cv.Rodrigues(np.array(pose[3:6]))
        t = np.array(pose[0:3]) * 1000.0  # เมตร -> มิลลิเมตร
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    except Exception as e:
        logger.error(f"[RTDE ERROR] Failed to read actual TCP Pose: {e}")
        return None


# =============================================================================
# 4. TEMPORAL AVERAGING FILTER (MODULE 07)
# =============================================================================
class TemporalSmoother:
    """Rolling buffer of 3D coordinates to ensure stability and reject jitter."""
    def __init__(self, buffer_size: int = 4, max_jitter_mm: float = 8.0):
        self.buffer = deque(maxlen=buffer_size)
        self.max_jitter = max_jitter_mm

    def add(self, xyz: np.ndarray):
        if xyz is not None:
            self.buffer.append(np.array(xyz, dtype=np.float64))

    def get_smoothed(self) -> tuple[Optional[np.ndarray], float, bool]:
        if len(self.buffer) < 2:
            return (self.buffer[-1].copy() if len(self.buffer) == 1 else None, 0.0, False)
        arr = np.array(self.buffer)
        mean_xyz = np.mean(arr, axis=0)
        std_xyz = np.std(arr, axis=0)
        jitter = float(np.linalg.norm(std_xyz))
        is_stable = (jitter <= self.max_jitter) and (len(self.buffer) >= self.buffer.maxlen)
        return mean_xyz, jitter, is_stable

    def reset(self):
        self.buffer.clear()


# =============================================================================
# 5. RESILIENT CAMERA MANAGER (CASE 2 & CASE 5)
# =============================================================================
class ResilientCameraManager:
    """Manages Orbbec Gemini 2 camera lifecycle, safe depth parsing, and firmware reset."""
    def __init__(self):
        self.ctx = None
        self.pipeline = None
        self.device = None
        self.is_connected = False
        self.last_retry = 0.0
        self.consecutive_drops = 0
        self.last_good_frame_time = 0.0

    def connect(self) -> tuple[bool, str]:
        """Soft-reset and initialize Orbbec camera pipeline safely."""
        if not ORBBEC_AVAILABLE:
            self.is_connected = False
            return False, "pyorbbecsdk not installed"

        self.stop()
        gc.collect()
        time.sleep(0.2)

        try:
            self.ctx = Context()
            dev_list = self.ctx.query_devices()
            if dev_list.get_count() == 0:
                self.is_connected = False
                return False, "ไม่พบอุปกรณ์ Orbbec บนพอร์ต USB"

            self.device = dev_list.get_device_by_index(0)
            self.pipeline = Pipeline(self.device)
            config = Config()

            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            config.enable_stream(color_profiles.get_default_video_stream_profile())

            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())

            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.start(config)

            self.is_connected = True
            self.consecutive_drops = 0
            self.last_good_frame_time = time.time()
            logger.info("[CAMERA] Orbbec Gemini 2 pipeline connected successfully.")
            return True, "เชื่อมต่อกล้อง Orbbec สำเร็จสมบูรณ์"

        except Exception as e:
            self.is_connected = False
            self.stop()
            return False, f"เปิดกล้องล้มเหลว: {e}"

    def get_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads Color and Depth frames safely.
        Applies Case 2 Byte Validation Guard. Drops corrupted frames silently.
        """
        if not self.is_connected or self.pipeline is None:
            return None, None
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                self._check_frame_watchdog()
                return None, None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                self._check_frame_watchdog()
                return None, None

            # 1. Safe Color Conversion
            img = safe_frame_to_bgr_image(color_frame)

            # 2. Case 2: Safe Depth Extraction with Byte Verification
            depth_raw = get_valid_depth_image(
                depth_frame,
                expected_w=CAMERA_CONFIG["DEPTH_WIDTH"],
                expected_h=CAMERA_CONFIG["DEPTH_HEIGHT"]
            )

            if img is not None and depth_raw is not None:
                scale = depth_frame.get_depth_scale()
                depth_mm = depth_raw.astype(np.float32) * scale
                self.consecutive_drops = 0
                self.last_good_frame_time = time.time()
                return img, depth_mm
            else:
                self.consecutive_drops += 1
                self._check_frame_watchdog()
                return None, None

        except Exception:
            self.is_connected = False
            return None, None

    def _check_frame_watchdog(self):
        """No valid frame for too long (e.g. USB drop) -> mark disconnected so auto-recovery reconnects."""
        if time.time() - self.last_good_frame_time > CAMERA_CONFIG["FRAME_STALL_TIMEOUT_S"]:
            logger.warning(f"[CAMERA] No valid frames for {CAMERA_CONFIG['FRAME_STALL_TIMEOUT_S']:.0f} s -> marking disconnected for auto-reconnect")
            self.stop()

    def trigger_firmware_soft_reboot(self):
        """Firmware Soft-Reboot via SDK (100% safe, no power surges)."""
        try:
            if self.device is not None and hasattr(self.device, "reboot"):
                logger.info("[CAMERA] ส่งคำสั่ง Firmware Soft-Reboot สู่กล้อง...")
                self.device.reboot()
                time.sleep(2.0)
                return True
        except Exception as e:
            logger.warning(f"[CAMERA] Firmware reboot warning: {e}")
        return False

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.device = None
        self.ctx = None
        self.is_connected = False


# =============================================================================
# 6. UI STATUS PILL COMPONENT
# =============================================================================
class StatusPill(tk.Frame):
    def __init__(self, parent, text: str, status: str = "offline", **kwargs):
        super().__init__(parent, bg=BG_SURFACE2, padx=8, pady=3, **kwargs)
        self.status = status
        self.text_str = text

        self.dot = tk.Label(self, text="●", font=("Leelawadee UI", 8), bg=BG_SURFACE2, fg=self._get_color(status))
        self.dot.pack(side=tk.LEFT, padx=(0, 4))

        self.lbl = tk.Label(self, text=text, font=("Leelawadee UI", 8, "bold"), bg=BG_SURFACE2, fg=TEXT_MAIN)
        self.lbl.pack(side=tk.LEFT)

    def _get_color(self, status: str) -> str:
        if status == "ok":
            return ACCENT_GREEN
        elif status == "warn":
            return ACCENT_WARN
        elif status == "info":
            return ACCENT_BLUE
        return ACCENT_RED

    def set_status(self, status: str, custom_text: Optional[str] = None):
        self.status = status
        self.dot.config(fg=self._get_color(status))
        if custom_text:
            self.lbl.config(text=custom_text)


# =============================================================================
# 7. MAIN APPLICATION (POOM_WINDOWS GUI STYLE + HARDENED WORKER)
# =============================================================================
class KnomBuabarnApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Knom_Buabarn | Unified Industrial Cobot Vision (UR12e)")
        self.geometry("1480x880")
        self.minsize(1280, 760)
        self.configure(bg=BG_MAIN)

        # Worker & Model Threading State
        self.is_running = True
        self.model_lock = threading.Lock()
        self.model = None
        self.current_model_path = DEFAULT_MODEL_FILE
        self.active_classes = {0: "Egg", 1: "BlueBox", 2: "RedBox"}
        self.command_class_map = {
            "CARTON": list(CARTON_TARGET_CLASSES),
            "EGG": list(EGG_TARGET_CLASSES)
        }

        # Quick Target Override & Snapshot Buffers
        self.var_target_override = tk.StringVar(value="Auto (Robot)")
        self.manual_test_requested = False
        self.latest_raw_bgr = None
        self.latest_depth_raw = None

        # Hardware Managers
        self.camera_mgr = ResilientCameraManager()
        self.camera_reconnect_requested = False  # Set by UI button, handled by worker thread
        self.rtde_r = None
        self.last_rtde_try = 0.0

        # Calibration Matrices
        self.camera_matrix = None
        self.T_cam2gripper = None
        self._load_calibrations()

        # TCP Server State
        self.tcp_server = None
        self.tcp_conn = None
        self.server_active = True
        self._tcp_buffers = {}

        # Telemetry & Queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.log_queue = queue.Queue()
        self.telemetry = {
            "cam_connected": False,
            "ur_connected": False,
            "rtde_connected": False,
            "fps": 0.0,
            "target_class": "NONE",
            "conf": 0.0,
            "xyz": [0.0, 0.0, 0.0],
            "pixel": (0, 0),
            "jitter": 0.0,
            "is_stable": False,
            "cycle": 0,
            "cycle_time": 0.0,
            "safety_status": "WAITING",
            "safety_reason": "Waiting for Target",
            "banner_text": "SYSTEM READY - WAITING FOR UR",
            "banner_color": ACCENT_BLUE,
        }

        # Hotkey bindings
        self.bind("<F12>", lambda e: self._save_snapshot())

        # Tunable Settings (Interactive GUI)
        self.var_conf = tk.DoubleVar(value=0.25)
        self.var_strategy = tk.StringVar(value="MAX_Y")  # MAX_Y or MAX_CONF
        self.var_roi_enabled = tk.BooleanVar(value=True)
        self.var_temporal_enabled = tk.BooleanVar(value=True)
        self.var_safety_z_enabled = tk.BooleanVar(value=True)

        # Default Conveyor ROI (x1, y1, x2, y2)
        self.roi = {"x1": 200, "y1": 80, "x2": 1100, "y2": 660}
        self.drag_start = None

        # Build UI & Load Initial YOLO Model
        self._build_ui()
        self._load_initial_model()

        # Start Background Worker Thread
        self.worker_thread = threading.Thread(target=self._worker_thread_loop, daemon=True)
        self.worker_thread.start()

        # UI Periodic Polling
        self.after(33, self._ui_poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------------
    # CALIBRATION & MODEL LOADERS
    # -------------------------------------------------------------------------
    def _load_calibrations(self):
        """Load intrinsic camera matrix and hand-eye calibration matrix."""
        try:
            if CALIB_PATH.exists():
                calib = np.load(CALIB_PATH)
                self.camera_matrix = calib["camera_matrix"]
                logger.info("[INIT] Loaded Camera Intrinsic Matrix successfully.")
            else:
                logger.error(f"[INIT ERROR] Calibration file not found: {CALIB_PATH}")

            if HANDEYE_PATH.exists():
                handeye = np.load(HANDEYE_PATH, allow_pickle=True)
                self.T_cam2gripper = handeye["T_cam_to_gripper"]
                logger.info("[INIT] Loaded Hand-Eye Matrix successfully.")
            else:
                logger.error(f"[INIT ERROR] Hand-Eye file not found: {HANDEYE_PATH}")
        except Exception as e:
            logger.error(f"[INIT ERROR] Error loading calibrations: {e}")

    def _load_initial_model(self):
        """Case 6: Load initial model with fallback guard."""
        config_path = self._read_config_model_path()
        target = config_path if (config_path and Path(config_path).exists()) else DEFAULT_MODEL_FILE
        self.load_yolo_model(str(target))

    def _read_config_model_path(self) -> Optional[str]:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("yolo_model_path")
            except Exception:
                pass
        return None

    def _save_config(self):
        try:
            data = {
                "yolo_model_path": str(self.current_model_path),
                "confidence": self.var_conf.get(),
                "strategy": self.var_strategy.get(),
                "roi": self.roi,
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._log_msg("[CONFIG] Saved settings to gui_config.json")
        except Exception as e:
            logger.error(f"[CONFIG ERROR] Failed to save config: {e}")

    # -------------------------------------------------------------------------
    # CASE 6: YOLO MODEL SELECTION & SMART DYNAMIC CLASS MAPPER
    # -------------------------------------------------------------------------
    def _browse_custom_model(self):
        """Case 6: Browse .pt weights starting in Data/ folder."""
        initial_dir = DATA_DIR if DATA_DIR.exists() else BASE_DIR
        selected = filedialog.askopenfilename(
            title="เลือกไฟล์ YOLO Weights (.pt)",
            initialdir=str(initial_dir),
            filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")]
        )
        if selected:
            success, msg = self.load_yolo_model(selected)
            if success:
                messagebox.showinfo("YOLO Model", msg)
            else:
                messagebox.showerror("YOLO Error", msg)

    def _reset_default_model(self):
        success, msg = self.load_yolo_model(str(DEFAULT_MODEL_FILE))
        if success:
            messagebox.showinfo("YOLO Reset", "Reset model to default Data/best.pt successfully.")

    def load_yolo_model(self, model_path_str: str) -> tuple[bool, str]:
        """Case 6: Thread-Safe Hot-Swap Model with Fallback Guard."""
        target_path = Path(model_path_str)

        # Fallback Guard: If selected file missing, fall back to Data/best.pt
        if not target_path.exists():
            logger.warning(f"[MODEL WARN] File {target_path.name} not found -> Falling back to Data/best.pt")
            target_path = DEFAULT_MODEL_FILE
            if not target_path.exists():
                return False, "ไม่พบทั้งโมเดลที่เลือกและไฟล์ Default best.pt"

        try:
            if not YOLO_AVAILABLE:
                return False, "Ultralytics YOLO is not installed"

            loaded_model = YOLO(str(target_path))
            class_dict = loaded_model.names

            with self.model_lock:
                self.model = loaded_model
                self.current_model_path = target_path
                self.active_classes = class_dict
                self._update_dynamic_class_mapping(class_dict)

            self._save_config()

            # Update UI labels
            if hasattr(self, "lbl_model_name"):
                self.lbl_model_name.config(text=f"Model: {target_path.name}")
                class_str = ", ".join([f"{k}:{v}" for k, v in class_dict.items()])
                self.lbl_model_classes.config(text=f"Classes: [{class_str}]")

            msg = f"Loaded {target_path.name} (Classes: {list(class_dict.values())})"
            logger.info(f"[MODEL] {msg}")
            self._log_msg(f"[MODEL] {msg}")
            return True, msg

        except Exception as e:
            logger.error(f"[MODEL ERROR] Error loading {target_path.name}: {e}")
            return False, f"โหลดโมเดลล้มเหลว: {e}"

    def _update_dynamic_class_mapping(self, class_dict: dict):
        """Smart Dynamic Class Resolver: Maps robot commands to class IDs automatically."""
        carton_ids = []
        egg_ids = []
        mapping = {}

        for cid, name in class_dict.items():
            name_lower = name.lower()
            mapping[name.upper()] = [cid]

            if any(k in name_lower for k in ["box", "carton", "blue", "red"]):
                carton_ids.append(cid)
            if any(k in name_lower for k in ["egg", "kai"]):
                egg_ids.append(cid)

        if CARTON_TARGET_CLASSES is not None:
            mapping["CARTON"] = list(CARTON_TARGET_CLASSES)
        else:
            mapping["CARTON"] = carton_ids if carton_ids else [0]

        if EGG_TARGET_CLASSES is not None:
            mapping["EGG"] = list(EGG_TARGET_CLASSES)
        else:
            mapping["EGG"] = egg_ids if egg_ids else [0]

        mapping["ALL"] = list(class_dict.keys())

        self.command_class_map = mapping
        logger.info(f"[MODEL] Dynamic Class Map updated: {self.command_class_map}")

    # -------------------------------------------------------------------------
    # CASE 5: RECONNECT CAMERA BUTTON ACTION
    # -------------------------------------------------------------------------
    def _trigger_reconnect_camera(self):
        """Case 5: Live non-blocking camera reconnect action.
        Only sets a flag; the worker thread performs the reboot/reconnect so the
        camera is never opened by two threads at once."""
        self._log_msg("[CAMERA] User triggered manual camera reconnect...")
        self.camera_reconnect_requested = True

    def _manual_cam_reconnect_worker(self):
        """Runs inside the worker thread (see _worker_thread_loop step A)."""
        self.camera_mgr.trigger_firmware_soft_reboot()
        success, msg = self.camera_mgr.connect()
        self._log_msg(f"[CAMERA] {msg}")

    # -------------------------------------------------------------------------
    # QUICK TARGET OVERRIDE & MANUAL TEST ACTIONS
    # -------------------------------------------------------------------------
    def _on_target_override_changed(self, event=None):
        """Interactive Workpiece Override from GUI Combobox."""
        val = self.var_target_override.get()
        if val == "RedBox Only":
            self.command_class_map["CARTON"] = [2]
            self._log_msg("[TARGET OVERRIDE] Target locked to RedBox Only (Class ID: 2)")
        elif val == "BlueBox Only":
            self.command_class_map["CARTON"] = [1]
            self._log_msg("[TARGET OVERRIDE] Target locked to BlueBox Only (Class ID: 1)")
        elif val == "Cartons (Blue + Red)":
            self.command_class_map["CARTON"] = [1, 2]
            self._log_msg("[TARGET OVERRIDE] Target set to Cartons (Blue + Red: [1, 2])")
        elif val == "Egg Only":
            self.command_class_map["EGG"] = [0]
            self._log_msg("[TARGET OVERRIDE] Target locked to Egg Only (Class ID: 0)")
        else:
            self.command_class_map["CARTON"] = list(CARTON_TARGET_CLASSES)
            self.command_class_map["EGG"] = list(EGG_TARGET_CLASSES)
            self._log_msg("[TARGET OVERRIDE] Restored Auto (Controlled by Robot Commands)")

    def _trigger_manual_test(self):
        """1-Shot Manual Test: triggers detection and 3D calculation without UR."""
        self.manual_test_requested = True
        self._log_msg("[MANUAL TEST] One-shot detection triggered from GUI...")

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION (MODERN POOM_WINDOWS STYLE)
    # -------------------------------------------------------------------------
    def _build_ui(self):
        pad_x = 10

        # Header Bar: Title + Status Pills
        header = tk.Frame(self, bg=BG_SURFACE2, height=44, padx=12)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(header, text="KNOM BUABARN", font=("Leelawadee UI", 12, "bold"), fg=ACCENT_GREEN, bg=BG_SURFACE2).pack(side=tk.LEFT)
        tk.Label(header, text=" | Cobot Vision Service", font=("Leelawadee UI", 10), fg=TEXT_MUTED, bg=BG_SURFACE2).pack(side=tk.LEFT)

        pills = tk.Frame(header, bg=BG_SURFACE2)
        pills.pack(side=tk.RIGHT)

        self.pill_cam = StatusPill(pills, text="CAM", status="offline")
        self.pill_cam.pack(side=tk.LEFT, padx=3)
        self.pill_ur = StatusPill(pills, text="UR TCP", status="offline")
        self.pill_ur.pack(side=tk.LEFT, padx=3)
        self.pill_rtde = StatusPill(pills, text="RTDE", status="offline")
        self.pill_rtde.pack(side=tk.LEFT, padx=3)
        self.pill_fps = StatusPill(pills, text="FPS: 0", status="info")
        self.pill_fps.pack(side=tk.LEFT, padx=3)

        # Body: Left Controls + Right Canvas
        body = tk.Frame(self, bg=BG_MAIN, padx=10, pady=8)
        body.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL
        left_panel = tk.Frame(body, bg=BG_SURFACE, width=470)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Scrollable Control Area
        ctrl_canvas = tk.Canvas(left_panel, bg=BG_SURFACE, highlightthickness=0)
        ctrl_scroll = tk.Scrollbar(left_panel, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        scroll_frame = tk.Frame(ctrl_canvas, bg=BG_SURFACE)

        scroll_frame.bind("<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=450)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_control_cards(scroll_frame)

        # RIGHT PANEL (Live Canvas & Status Banner)
        right_panel = tk.Frame(body, bg=BG_SURFACE)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_bar = tk.Frame(right_panel, bg=BG_SURFACE2, height=32, padx=8)
        canvas_bar.pack(fill=tk.X, side=tk.TOP)
        tk.Label(canvas_bar, text="LIVE 3D VISION FEED (1280x720)", font=("Leelawadee UI", 9, "bold"), fg=TEXT_MAIN, bg=BG_SURFACE2).pack(side=tk.LEFT)
        tk.Label(canvas_bar, text="Mouse Drag on canvas to adjust Conveyor ROI", font=("Leelawadee UI", 9), fg=TEXT_MUTED, bg=BG_SURFACE2).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(right_panel, bg="#000000", highlightthickness=1, highlightbackground=BORDER_COL)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        banner_frame = tk.Frame(right_panel, bg=BG_SURFACE2, height=36, padx=10)
        banner_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))
        self.lbl_banner = tk.Label(banner_frame, text="SYSTEM READY | WAITING FOR START", font=("Leelawadee UI", 11, "bold"), fg=ACCENT_GREEN, bg=BG_SURFACE2)
        self.lbl_banner.pack(side=tk.LEFT, fill=tk.Y)

    def _build_control_cards(self, parent):
        pad_x = 10

        # CARD 1: Core Controls & Case 5 Reconnect
        sec_core = tk.LabelFrame(parent, text=" 1. Core Controls ", bg=BG_SURFACE, fg=ACCENT_BLUE, font=("Leelawadee UI", 9, "bold"), padx=8, pady=6)
        sec_core.pack(fill=tk.X, padx=pad_x, pady=(6, 4))

        r1 = tk.Frame(sec_core, bg=BG_SURFACE)
        r1.pack(fill=tk.X, pady=2)
        self.btn_server = tk.Button(r1, text="⏹ Stop Server", font=("Leelawadee UI", 9, "bold"), bg=ACCENT_RED, fg="#ffffff", relief=tk.FLAT, command=self._toggle_server)
        self.btn_server.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_reconnect = tk.Button(r1, text="🔄 Cam Reconnect", font=("Leelawadee UI", 9), bg=BG_SURFACE2, fg=TEXT_MAIN, relief=tk.FLAT, command=self._trigger_reconnect_camera)
        self.btn_reconnect.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        r2 = tk.Frame(sec_core, bg=BG_SURFACE)
        r2.pack(fill=tk.X, pady=(4, 2))
        self.btn_manual_test = tk.Button(r2, text="🎯 1-Shot Test", font=("Leelawadee UI", 9, "bold"), bg=BG_SURFACE2, fg=ACCENT_BLUE, relief=tk.FLAT, command=self._trigger_manual_test)
        self.btn_manual_test.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_quick_snap = tk.Button(r2, text="📸 Snapshot (F12)", font=("Leelawadee UI", 9), bg=BG_SURFACE2, fg=TEXT_MAIN, relief=tk.FLAT, command=self._save_snapshot)
        self.btn_quick_snap.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # CARD 2: Case 6 YOLO Model Selection (.pt) & Target Override
        sec_model = tk.LabelFrame(parent, text=" 2. YOLO Model & Target Class (Case 6) ", bg=BG_SURFACE, fg=ACCENT_PURPLE, font=("Leelawadee UI", 9, "bold"), padx=8, pady=6)
        sec_model.pack(fill=tk.X, padx=pad_x, pady=4)

        self.lbl_model_name = tk.Label(sec_model, text=f"Model: {self.current_model_path.name}", font=("Leelawadee UI", 9, "bold"), fg=TEXT_MAIN, bg=BG_SURFACE, anchor="w")
        self.lbl_model_name.pack(fill=tk.X)
        self.lbl_model_classes = tk.Label(sec_model, text="Classes: [Egg, BlueBox, RedBox]", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_SURFACE, anchor="w")
        self.lbl_model_classes.pack(fill=tk.X, pady=(2, 4))

        r_m = tk.Frame(sec_model, bg=BG_SURFACE)
        r_m.pack(fill=tk.X)
        tk.Button(r_m, text="📂 Browse (.pt)", font=("Leelawadee UI", 9, "bold"), bg=BG_SURFACE2, fg=TEXT_MAIN, relief=tk.FLAT, command=self._browse_custom_model).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(r_m, text="🔄 Reset Default", font=("Leelawadee UI", 9), bg=BG_SURFACE2, fg=TEXT_MUTED, relief=tk.FLAT, command=self._reset_default_model).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Quick Class Target Selector
        r_override = tk.Frame(sec_model, bg=BG_SURFACE)
        r_override.pack(fill=tk.X, pady=(6, 2))
        tk.Label(r_override, text="Target Class:", font=("Leelawadee UI", 8, "bold"), fg=TEXT_MAIN, bg=BG_SURFACE).pack(side=tk.LEFT)
        self.cb_target_override = ttk.Combobox(
            r_override,
            textvariable=self.var_target_override,
            values=["Auto (Robot)", "Cartons (Blue + Red)", "RedBox Only", "BlueBox Only", "Egg Only"],
            state="readonly",
            width=18,
            font=("Leelawadee UI", 8)
        )
        self.cb_target_override.pack(side=tk.RIGHT, padx=2)
        self.cb_target_override.bind("<<ComboboxSelected>>", self._on_target_override_changed)

        # CARD 3: Telemetry & Safety Status (Case 1)
        sec_tele = tk.LabelFrame(parent, text=" 3. Target Telemetry & Safety (Case 1) ", bg=BG_SURFACE, fg=ACCENT_GREEN, font=("Leelawadee UI", 9, "bold"), padx=8, pady=6)
        sec_tele.pack(fill=tk.X, padx=pad_x, pady=4)

        self.lbl_det_class = tk.Label(sec_tele, text="Target: NONE", font=("Leelawadee UI", 10, "bold"), fg=TEXT_MAIN, bg=BG_SURFACE, anchor="w")
        self.lbl_det_class.pack(fill=tk.X)
        self.lbl_det_xyz = tk.Label(sec_tele, text="Robot Base XYZ: (0.00, 0.00, 0.00) mm", font=("Consolas", 9, "bold"), fg=ACCENT_GREEN, bg=BG_SURFACE, anchor="w")
        self.lbl_det_xyz.pack(fill=tk.X)

        # Safety Envelope Live Badge
        self.lbl_safety_badge = tk.Label(sec_tele, text="🛡️ SAFETY: WAITING FOR TARGET", font=("Leelawadee UI", 9, "bold"), bg=BG_SURFACE2, fg=TEXT_MUTED, pady=3)
        self.lbl_safety_badge.pack(fill=tk.X, pady=(4, 4))

        r_t = tk.Frame(sec_tele, bg=BG_SURFACE)
        r_t.pack(fill=tk.X)
        self.lbl_det_pixel = tk.Label(r_t, text="Pixel: (0, 0)", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_SURFACE)
        self.lbl_det_pixel.pack(side=tk.LEFT)
        self.lbl_det_jitter = tk.Label(r_t, text="Jitter: 0.0 mm [IDLE]", font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_SURFACE)
        self.lbl_det_jitter.pack(side=tk.RIGHT)

        self.lbl_det_cycle = tk.Label(sec_tele, text="Cycle: 0 | Time: 0.000 s", font=("Leelawadee UI", 8), fg=TEXT_MUTED, bg=BG_SURFACE, anchor="w")
        self.lbl_det_cycle.pack(fill=tk.X)

        # CARD 4: Live Tunable Settings & Safety Z Interlock
        sec_tune = tk.LabelFrame(parent, text=" 4. Live Optimization & Interlocks ", bg=BG_SURFACE, fg=ACCENT_WARN, font=("Leelawadee UI", 9, "bold"), padx=8, pady=6)
        sec_tune.pack(fill=tk.X, padx=pad_x, pady=4)

        # Confidence Slider
        f_c = tk.Frame(sec_tune, bg=BG_SURFACE)
        f_c.pack(fill=tk.X)
        tk.Label(f_c, text="Confidence:", font=("Leelawadee UI", 8), fg=TEXT_MAIN, bg=BG_SURFACE).pack(side=tk.LEFT)
        self.lbl_conf_val = tk.Label(f_c, text=f"{self.var_conf.get():.2f}", font=("Consolas", 8, "bold"), fg=ACCENT_WARN, bg=BG_SURFACE)
        self.lbl_conf_val.pack(side=tk.RIGHT)
        tk.Scale(sec_tune, from_=0.10, to=0.90, resolution=0.02, orient=tk.HORIZONTAL, variable=self.var_conf, command=lambda v: self.lbl_conf_val.config(text=f"{float(v):.2f}"), bg=BG_SURFACE, fg=TEXT_MAIN, highlightthickness=0).pack(fill=tk.X)

        # Interlock Checkboxes
        tk.Checkbutton(sec_tune, text=f"Safety Z Interlock (Safe: {ROBOT_SAFETY_CONFIG['Z_MIN_MM']} to +{ROBOT_SAFETY_CONFIG['Z_MAX_MM']}mm)", variable=self.var_safety_z_enabled, bg=BG_SURFACE, fg=TEXT_MAIN, selectcolor=BG_SURFACE2, activebackground=BG_SURFACE).pack(anchor="w")
        tk.Checkbutton(sec_tune, text="Conveyor ROI Boundary Active", variable=self.var_roi_enabled, bg=BG_SURFACE, fg=TEXT_MAIN, selectcolor=BG_SURFACE2, activebackground=BG_SURFACE).pack(anchor="w")
        tk.Checkbutton(sec_tune, text="Temporal Averaging (Jitter Guard <= 8mm)", variable=self.var_temporal_enabled, bg=BG_SURFACE, fg=TEXT_MAIN, selectcolor=BG_SURFACE2, activebackground=BG_SURFACE).pack(anchor="w")

        # CARD 5: System Logs
        sec_log = tk.LabelFrame(parent, text=" 5. Runtime Event Logs ", bg=BG_SURFACE, fg=TEXT_MUTED, font=("Leelawadee UI", 9, "bold"), padx=8, pady=6)
        sec_log.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=4)

        self.txt_log = tk.Text(sec_log, height=7, bg="#0d0e11", fg="#a1a1aa", font=("Consolas", 8), relief=tk.FLAT)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        r_act = tk.Frame(sec_log, bg=BG_SURFACE)
        r_act.pack(fill=tk.X, pady=(4, 0))
        tk.Button(r_act, text="💾 Snapshot", font=("Leelawadee UI", 8), bg=BG_SURFACE2, fg=TEXT_MAIN, command=self._save_snapshot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(r_act, text="📊 Open CSV", font=("Leelawadee UI", 8), bg=BG_SURFACE2, fg=TEXT_MAIN, command=self._open_csv_log).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(r_act, text="🧹 Clear", font=("Leelawadee UI", 8), bg=BG_SURFACE2, fg=TEXT_MUTED, command=lambda: self.txt_log.delete("1.0", tk.END)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    # -------------------------------------------------------------------------
    # CANVAS ROI DRAGGING
    # -------------------------------------------------------------------------
    def _on_canvas_press(self, event):
        self.drag_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self.drag_start:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        x1 = int(min(self.drag_start[0], event.x) / cw * 1280.0)
        y1 = int(min(self.drag_start[1], event.y) / ch * 720.0)
        x2 = int(max(self.drag_start[0], event.x) / cw * 1280.0)
        y2 = int(max(self.drag_start[1], event.y) / ch * 720.0)
        self.roi = {
            "x1": max(0, min(x1, 1270)),
            "y1": max(0, min(y1, 710)),
            "x2": max(10, min(x2, 1280)),
            "y2": max(10, min(y2, 720)),
        }

    def _on_canvas_release(self, event):
        self.drag_start = None
        self._save_config()

    # -------------------------------------------------------------------------
    # WORKER THREAD (VISION + RTDE + FAIL-CLOSED SAFETY + TCP SERVER)
    # -------------------------------------------------------------------------
    def _robot_port_open(self, timeout_s: float = 0.3) -> bool:
        """Quick TCP probe so an offline robot doesn't block the worker loop."""
        try:
            with socket.create_connection(
                (ROBOT_SAFETY_CONFIG["TARGET_ROBOT_IP"], ROBOT_SAFETY_CONFIG["RTDE_PORT"]),
                timeout=timeout_s,
            ):
                return True
        except OSError:
            return False

    def _worker_thread_loop(self):
        """Continuous background thread handling vision processing, safety checks, and UR cycle."""
        # 1. Start TCP Server
        self._init_tcp_server()

        # 2. Camera Manager initial start
        self.camera_mgr.connect()

        smoother = TemporalSmoother(buffer_size=4, max_jitter_mm=8.0)

        state = WAIT_START
        cycle_number = 0
        cycle_start_time = None
        cycle_target_cmd = None
        cycle_target_ids = []
        target_locked = False

        prev_time = time.time()
        fps_counter = 0
        fps = 0.0

        logger.info("[WORKER] Background Vision & Safety Worker thread started.")

        while self.is_running:
            now = time.time()
            fps_counter += 1
            if now - prev_time >= 1.0:
                fps = fps_counter / (now - prev_time)
                fps_counter = 0
                prev_time = now

            # -----------------------------------------------------------------
            # A. CASE 5: CAMERA AUTO-RECOVERY (PROBE EVERY 3 SECONDS)
            # -----------------------------------------------------------------
            if self.camera_reconnect_requested:
                self.camera_reconnect_requested = False
                self._manual_cam_reconnect_worker()
                self.camera_mgr.last_retry = time.time()
            elif not self.camera_mgr.is_connected:
                if (now - self.camera_mgr.last_retry) >= CAMERA_CONFIG["RETRY_INTERVAL_S"]:
                    self.camera_mgr.last_retry = now
                    success, msg = self.camera_mgr.connect()
                    if success:
                        self._log_msg(f"[CAMERA] {msg}")

            # -----------------------------------------------------------------
            # B. CASE 1: RTDE STRICT LAZY CONNECT (IP 10.42.5.151)
            # -----------------------------------------------------------------
            if (self.rtde_r is None or not (hasattr(self.rtde_r, "isConnected") and self.rtde_r.isConnected())):
                if (now - self.last_rtde_try) >= ROBOT_SAFETY_CONFIG["RTDE_RETRY_INTERVAL_S"]:
                    self.last_rtde_try = now
                    # Quick port probe first: an offline robot would otherwise block this
                    # loop ~21 s inside RTDEReceiveInterface and freeze the camera feed.
                    if RTDE_AVAILABLE and self._robot_port_open():
                        try:
                            self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_SAFETY_CONFIG["TARGET_ROBOT_IP"])
                            logger.info(f"[RTDE] Connected strictly to UR at {ROBOT_SAFETY_CONFIG['TARGET_ROBOT_IP']}")
                            self._log_msg(f"[RTDE] Connected to UR12e at {ROBOT_SAFETY_CONFIG['TARGET_ROBOT_IP']}")
                        except Exception:
                            self.rtde_r = None

            # -----------------------------------------------------------------
            # C. MANAGE UR TCP CONNECTION & MESSAGES
            # -----------------------------------------------------------------
            if self.server_active and self.tcp_server is not None:
                if self.tcp_conn is None:
                    try:
                        conn, addr = self.tcp_server.accept()
                        conn.settimeout(0.01)

                        # FUTURE EXTENSION: Case 3 TCP Keep-Alive on Windows
                        if ENABLE_TCP_KEEPALIVE and sys.platform == "win32":
                            try:
                                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                                conn.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 1000, 1000))
                            except Exception as e:
                                logger.warning(f"[TCP] Failed to set SIO_KEEPALIVE_VALS: {e}")

                        self.tcp_conn = conn
                        state = WAIT_START
                        target_locked = False
                        smoother.reset()
                        logger.info(f"[TCP] UR connected from {addr[0]}:{addr[1]}")
                        self._log_msg(f"[TCP] UR connected from {addr[0]}:{addr[1]}")
                    except (socket.timeout, BlockingIOError):
                        pass
                    except Exception as e:
                        logger.error(f"[TCP ACCEPT ERROR] {e}")

            # Receive TCP messages from UR
            if self.tcp_conn is not None:
                msgs = self._recv_tcp_messages(self.tcp_conn)
                if msgs is None:
                    # UR disconnected -> Close socket and reset state
                    try:
                        self.tcp_conn.close()
                    except Exception:
                        pass
                    self.tcp_conn = None
                    state = WAIT_START
                    target_locked = False
                    smoother.reset()
                    logger.warning("[TCP] UR disconnected -> Waiting for reconnect")
                    self._log_msg("[TCP] UR disconnected -> Resetting state")
                else:
                    for msg in msgs:
                        text = msg.upper().strip()
                        logger.info(f"[TCP RECV] '{text}'")

                        if text == "READY":
                            state = WAIT_START
                            self._log_msg("[TCP] READY received from UR")

                        elif text == "START":
                            cycle_number += 1
                            cycle_start_time = time.perf_counter()
                            cycle_target_cmd = None
                            cycle_target_ids = []
                            target_locked = False
                            state = WAIT_CLASS
                            smoother.reset()
                            self._log_msg(f"[CYCLE {cycle_number}] START received -> Awaiting class")

                        elif text in self.command_class_map:
                            cycle_target_cmd = text
                            cycle_target_ids = self.command_class_map[text]
                            target_locked = False
                            state = WAIT_TARGET
                            smoother.reset()
                            self._log_msg(f"[CLASS] Target class requested: {text} (IDs: {cycle_target_ids})")

                        elif text == "DONE":
                            duration = (time.perf_counter() - cycle_start_time) if cycle_start_time else 0.0
                            self._log_msg(f"[CYCLE {cycle_number}] Cycle DONE in {duration:.3f} s")
                            state = WAIT_START
                            target_locked = False
                            smoother.reset()

            # -----------------------------------------------------------------
            # D. ACQUIRE FRAMES (PROTECTED BY CASE 2 BYTE GUARD)
            # -----------------------------------------------------------------
            img, depth_mm = self.camera_mgr.get_frames()

            if img is not None:
                self.latest_raw_bgr = img.copy()
            if depth_mm is not None:
                self.latest_depth_raw = depth_mm.copy()

            annotated = None
            detected_target_base = None
            best_cx, best_cy = 0, 0
            best_label = "NONE"
            best_conf = 0.0
            is_stable = False
            jitter_val = 0.0
            safety_badge_stat = "WAITING"
            safety_badge_msg = "Waiting for Target"

            # Check 1-shot manual test request from GUI button
            is_manual_test = False
            if self.manual_test_requested:
                is_manual_test = True
                self.manual_test_requested = False

            if img is not None:
                annotated = img.copy()

                # Draw ROI border overlay
                if self.var_roi_enabled.get():
                    r = self.roi
                    cv.rectangle(annotated, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 255, 0), 2)
                    cv.putText(annotated, "CONVEYOR ROI", (r["x1"] + 6, r["y1"] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # -------------------------------------------------------------
                # E. YOLO INFERENCE & TARGET RESOLUTION (WAIT_TARGET or MANUAL)
                # -------------------------------------------------------------
                should_run_yolo = (
                    (state == WAIT_TARGET and cycle_target_ids and not target_locked) or
                    is_manual_test
                )

                if (should_run_yolo and
                    depth_mm is not None and
                    self.camera_matrix is not None and
                    self.T_cam2gripper is not None):

                    yolo_classes = cycle_target_ids if (state == WAIT_TARGET and cycle_target_ids) else self.command_class_map.get("CARTON", [1, 2])

                    with self.model_lock:
                        if self.model is not None:
                            results = self.model.predict(
                                source=img,
                                conf=self.var_conf.get(),
                                device="cpu",
                                verbose=False,
                                classes=yolo_classes
                            )

                    r0 = results[0] if (results and len(results) > 0) else None
                    if r0 is not None and r0.boxes is not None and len(r0.boxes) > 0:
                        candidate_boxes = []
                        for box in r0.boxes:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                            bcx = (bx1 + bx2) // 2
                            bcy = (by1 + by2) // 2
                            bconf = float(box.conf[0])
                            bcls = int(box.cls[0])

                            # Check ROI
                            if self.var_roi_enabled.get():
                                r = self.roi
                                if not (r["x1"] <= bcx <= r["x2"] and r["y1"] <= bcy <= r["y2"]):
                                    # Outside ROI
                                    cv.rectangle(annotated, (bx1, by1), (bx2, by2), (80, 80, 80), 1)
                                    continue

                            candidate_boxes.append((box, bx1, by1, bx2, by2, bcx, bcy, bconf, bcls))

                        if candidate_boxes:
                            # Strategy: Max-Y (front of conveyor) or Max-Conf
                            if self.var_strategy.get() == "MAX_Y":
                                candidate_boxes.sort(key=lambda item: item[6], reverse=True) # Sort by bcy
                            else:
                                candidate_boxes.sort(key=lambda item: item[7], reverse=True) # Sort by bconf

                            best = candidate_boxes[0]
                            best_box, x1, y1, x2, y2, best_cx, best_cy, best_conf, best_cls = best
                            best_label = self.active_classes.get(best_cls, str(best_cls))

                            cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv.circle(annotated, (best_cx, best_cy), 5, (0, 255, 255), -1)

                            # Median depth patch
                            depth_val = get_depth_median_at(depth_mm, best_cx, best_cy, radius=CAMERA_CONFIG["MEDIAN_SAMPLE_RADIUS"])

                            if depth_val is not None:
                                fx = self.camera_matrix[0, 0]
                                fy = self.camera_matrix[1, 1]
                                cx0 = self.camera_matrix[0, 2]
                                cy0 = self.camera_matrix[1, 2]

                                p_cam = np.array([
                                    (best_cx - cx0) * depth_val / fx,
                                    (best_cy - cy0) * depth_val / fy,
                                    depth_val,
                                    1.0
                                ], dtype=np.float64)

                                # Camera Frame -> Gripper Frame
                                p_grip = self.T_cam2gripper @ p_cam

                                # CASE 1: Gripper Frame -> Robot Base Frame (Fail-Closed RTDE)
                                T_grip2base = get_tcp_pose_mm(self.rtde_r)

                                if T_grip2base is not None:
                                    p_base = T_grip2base @ p_grip
                                    raw_xyz = p_base[:3]

                                    # Temporal Averaging (Module 07)
                                    if self.var_temporal_enabled.get():
                                        smoother.add(raw_xyz)
                                        smoothed_xyz, jitter_val, is_stable = smoother.get_smoothed()
                                        detected_target_base = smoothed_xyz if smoothed_xyz is not None else raw_xyz
                                    else:
                                        detected_target_base = raw_xyz
                                        is_stable = True
                                        jitter_val = 0.0

                                    # CASE 1: 3D Safety Envelope Interlock Check
                                    safety_ok = True
                                    safety_reason = "SAFE"
                                    if self.var_safety_z_enabled.get() and detected_target_base is not None:
                                        safety_ok, safety_reason = check_safety_envelope(detected_target_base)

                                    if not safety_ok:
                                        safety_badge_stat = "VIOLATION"
                                        safety_badge_msg = safety_reason
                                        logger.error(f"[SAFETY VIOLATION] Aborted target: {safety_reason}")
                                        cv.putText(annotated, f"SAFETY ABORT: {safety_reason}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                                        if is_manual_test:
                                            self._log_msg(f"[MANUAL TEST] {best_label}: Base XYZ=({detected_target_base[0]:.1f}, {detected_target_base[1]:.1f}, {detected_target_base[2]:.1f}) mm -> VIOLATION: {safety_reason}")
                                    else:
                                        safety_badge_stat = "SAFE"
                                        safety_badge_msg = "Envelope OK"
                                        if is_manual_test:
                                            self._log_msg(f"[MANUAL TEST] {best_label}: Base XYZ=({detected_target_base[0]:.1f}, {detected_target_base[1]:.1f}, {detected_target_base[2]:.1f}) mm -> SAFE")
                                        else:
                                            # CASE 1 & STANDARD SEND: Transmit Coordinates to UR
                                            if detected_target_base is not None and is_stable and not target_locked:
                                                if self.tcp_conn is not None:
                                                    try:
                                                        sent_str = self._send_target_coordinates(self.tcp_conn, detected_target_base)
                                                        target_locked = True
                                                        state = WAIT_DONE
                                                        self._log_msg(f"[TARGET SENT] {best_label}: {sent_str} mm")
                                                        self._save_cycle_csv(cycle_number, best_label, best_cx, best_cy, detected_target_base)
                                                    except Exception as e:
                                                        logger.error(f"[TCP SEND FAILED] {e}")
                                else:
                                    # RTDE not connected or pose read failed -> Fail-Closed Abort
                                    cv.putText(annotated, "RTDE NOT READY (Waiting for Pose...)", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                                    logger.warning("[RTDE FAIL-CLOSED] Pose is None, coordinate transmission blocked.")
                                    safety_badge_stat = "VIOLATION"
                                    safety_badge_msg = "RTDE Offline"
                    else:
                        if is_manual_test:
                            self._log_msg(f"[MANUAL TEST] No workpiece detected for classes: {yolo_classes}")

            # Update Telemetry Packet
            status_text = f"STATE: {state} | Filter: {cycle_target_cmd}"
            status_color = ACCENT_GREEN if state == WAIT_DONE else ACCENT_BLUE

            self.telemetry = {
                "cam_connected": self.camera_mgr.is_connected,
                "ur_connected": self.tcp_conn is not None,
                "rtde_connected": (self.rtde_r is not None and hasattr(self.rtde_r, "isConnected") and self.rtde_r.isConnected()),
                "fps": fps,
                "target_class": best_label,
                "conf": best_conf,
                "xyz": detected_target_base.tolist() if detected_target_base is not None else [0.0, 0.0, 0.0],
                "pixel": (best_cx, best_cy),
                "jitter": jitter_val,
                "is_stable": is_stable,
                "cycle": cycle_number,
                "cycle_time": (time.perf_counter() - cycle_start_time) if cycle_start_time else 0.0,
                "safety_status": safety_badge_stat,
                "safety_reason": safety_badge_msg,
                "banner_text": status_text,
                "banner_color": status_color,
            }

            # Enqueue preview frame for GUI display
            if annotated is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(annotated)

            time.sleep(0.01)

        # Worker shutdown
        self.camera_mgr.stop()
        if self.tcp_conn:
            try:
                self.tcp_conn.close()
            except Exception:
                pass
        if self.tcp_server:
            try:
                self.tcp_server.close()
            except Exception:
                pass
        logger.info("[WORKER] Worker thread exited gracefully.")

    # -------------------------------------------------------------------------
    # TCP MESSAGING & COORDINATE TRANSMISSION
    # -------------------------------------------------------------------------
    def _init_tcp_server(self):
        try:
            self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server.bind(("0.0.0.0", DEFAULT_TCP_PORT))
            self.tcp_server.listen(1)
            self.tcp_server.settimeout(0.02)
            logger.info(f"[TCP] Server listening on port {DEFAULT_TCP_PORT} (Host PC: {ROBOT_SAFETY_CONFIG['PC_IP']})")
        except Exception as e:
            logger.error(f"[TCP ERROR] Failed to bind port {DEFAULT_TCP_PORT}: {e}")

    def _recv_tcp_messages(self, conn) -> Optional[list[str]]:
        """Robust socket receiver supporting both newline and raw strings without newline via socket timeout."""
        if conn not in self._tcp_buffers:
            self._tcp_buffers[conn] = ""
        try:
            data = conn.recv(1024)
            if not data:
                rem = self._tcp_buffers.pop(conn, "").strip()
                return [rem] if rem else None

            text = data.decode("ascii", errors="replace")
            self._tcp_buffers[conn] += text

            messages = []
            buf = self._tcp_buffers[conn].replace("\r\n", "\n").replace("\r", "\n")
            self._tcp_buffers[conn] = buf

            if "\n" in self._tcp_buffers[conn]:
                lines = self._tcp_buffers[conn].split("\n")
                self._tcp_buffers[conn] = lines[-1]
                for line in lines[:-1]:
                    m = line.strip()
                    if m:
                        messages.append(m)
            return messages

        except socket.timeout:
            # If timeout occurs and residual text exists in buffer (messages without \n), treat as a message
            if conn in self._tcp_buffers and self._tcp_buffers[conn].strip():
                msg = self._tcp_buffers[conn].strip()
                self._tcp_buffers[conn] = ""
                return [msg]
            return []
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            rem = self._tcp_buffers.pop(conn, "").strip()
            return [rem] if rem else None

    def _send_target_coordinates(self, conn, target_xyz: np.ndarray) -> str:
        """
        Sends coordinates to UR.
        Single standard send per user preference.
        
        FUTURE EXTENSION HOOK (CASE 3 & 4):
        If you wish to enable the 2-Way Handshake in the future:
        1. Send: msg = f"TARGET:{x:.2f},{y:.2f},{z:.2f}"
        2. Wait for ACK response from UR: socket.recv()
        3. Verify ACK matches sent coordinates.
        4. If match, send "EXECUTE".
        """
        x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
        msg = f"{x:.2f},{y:.2f},{z:.2f}"

        # Case 1: Pre-Transmission Blackbox Log
        logger.info(f"[PRE-TRANSMISSION] Sending Target Coordinates to UR: X={x:.2f}, Y={y:.2f}, Z={z:.2f} mm")

        conn.sendall(msg.encode("ascii"))
        return msg

    def _save_cycle_csv(self, cycle: int, label: str, cx: int, cy: int, xyz: np.ndarray):
        file_exists = CSV_FILE.exists()
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = cv.FileStorage if False else None
            # Standard CSV logging
            import csv
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Cycle", "Target_Class", "Pixel_X", "Pixel_Y", "Robot_X_mm", "Robot_Y_mm", "Robot_Z_mm"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cycle,
                label,
                cx, cy,
                f"{xyz[0]:.2f}", f"{xyz[1]:.2f}", f"{xyz[2]:.2f}"
            ])

    # -------------------------------------------------------------------------
    # UI PERIODIC REFRESH (TKINTER MAIN THREAD)
    # -------------------------------------------------------------------------
    def _ui_poll(self):
        # 1. Drain log queue
        while not self.log_queue.empty():
            try:
                line = self.log_queue.get_nowait()
                self.txt_log.insert(tk.END, line + "\n")
                self.txt_log.see(tk.END)
            except queue.Empty:
                break

        # 2. Update Image Canvas
        try:
            frame = self.frame_queue.get_nowait()
            if PIL_AVAILABLE:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                cw, ch = max(10, self.canvas.winfo_width()), max(10, self.canvas.winfo_height())
                rgb_resized = cv.resize(rgb, (cw, ch), interpolation=cv.INTER_LINEAR)
                img_pil = Image.fromarray(rgb_resized)
                self.current_imgtk = ImageTk.PhotoImage(image=img_pil)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.current_imgtk)
        except queue.Empty:
            pass

        # 3. Update Status Pills and Labels
        t = self.telemetry
        self.pill_cam.set_status("ok" if t["cam_connected"] else "offline")
        self.pill_ur.set_status("ok" if t["ur_connected"] else "offline")
        self.pill_rtde.set_status("ok" if t["rtde_connected"] else "offline")
        self.pill_fps.set_status("info", f"FPS: {t['fps']:.1f}")

        self.lbl_det_class.config(text=f"Target: {t['target_class']}  (Conf: {t['conf']:.2f})")
        self.lbl_det_xyz.config(text=f"Robot Base XYZ: ({t['xyz'][0]:.2f}, {t['xyz'][1]:.2f}, {t['xyz'][2]:.2f}) mm")
        self.lbl_det_pixel.config(text=f"Pixel: ({t['pixel'][0]}, {t['pixel'][1]})")
        self.lbl_det_jitter.config(text=f"Jitter: {t['jitter']:.1f} mm [{'STABLE 🟢' if t['is_stable'] else 'WAIT 🟡'}]")
        self.lbl_det_cycle.config(text=f"Cycle: {t['cycle']} | Time: {t['cycle_time']:.3f} s")

        # Update Safety Badge Indicator
        s_stat = t.get("safety_status", "WAITING")
        s_reason = t.get("safety_reason", "")
        if s_stat == "SAFE":
            self.lbl_safety_badge.config(text=f"🛡️ SAFETY: SAFE ({s_reason})", bg="#14532d", fg="#4ade80")
        elif s_stat == "VIOLATION":
            self.lbl_safety_badge.config(text=f"⚠️ SAFETY: VIOLATION ({s_reason})", bg="#7f1d1d", fg="#f87171")
        else:
            self.lbl_safety_badge.config(text="🛡️ SAFETY: WAITING FOR TARGET", bg=BG_SURFACE2, fg=TEXT_MUTED)

        self.lbl_banner.config(text=t["banner_text"], fg=t["banner_color"])

        if self.is_running:
            self.after(33, self._ui_poll)

    def _toggle_server(self):
        self.server_active = not self.server_active
        if self.server_active:
            self.btn_server.config(text="⏹ Stop Server", bg=ACCENT_RED)
            self._log_msg("[SERVER] TCP Server started.")
        else:
            self.btn_server.config(text="▶ Start Server", bg=ACCENT_GREEN)
            self._log_msg("[SERVER] TCP Server stopped.")

    def _save_snapshot(self):
        """Comprehensive Industrial Snapshot: Saves raw RGB, depth map, and telemetry JSON."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_path = SNAPSHOT_DIR / f"snapshot_{ts}_rgb.jpg"
        depth_path = SNAPSHOT_DIR / f"snapshot_{ts}_depth.png"
        meta_path = SNAPSHOT_DIR / f"snapshot_{ts}_info.json"

        try:
            saved_count = 0
            if self.latest_raw_bgr is not None:
                cv.imwrite(str(rgb_path), self.latest_raw_bgr)
                saved_count += 1

            if self.latest_depth_raw is not None:
                depth_vis = cv.applyColorMap(cv.convertScaleAbs(self.latest_depth_raw, alpha=0.05), cv.COLORMAP_JET)
                cv.imwrite(str(depth_path), depth_vis)
                saved_count += 1

            info_data = {
                "timestamp": ts,
                "current_model": str(self.current_model_path.name),
                "target_class_override": self.var_target_override.get(),
                "telemetry": self.telemetry,
                "safety_config": ROBOT_SAFETY_CONFIG,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2, ensure_ascii=False)
            saved_count += 1

            msg = f"Saved snapshot set ({saved_count} files): {rgb_path.name}, {depth_path.name}, {meta_path.name}"
            logger.info(f"[SNAPSHOT] {msg}")
            self._log_msg(f"[SNAPSHOT] {msg}")
            messagebox.showinfo("Snapshot Saved", f"บันทึกชุดข้อมูลหลักฐานสำเร็จ (3 ไฟล์):\n- {rgb_path.name}\n- {depth_path.name}\n- {meta_path.name}")
        except Exception as e:
            logger.error(f"[SNAPSHOT ERROR] {e}")
            messagebox.showerror("Snapshot Error", str(e))

    def _open_csv_log(self):
        if not CSV_FILE.exists():
            messagebox.showinfo("CSV Log", "ยังไม่มีไฟล์ Log ข้อมูลการทำงาน")
            return
        try:
            if os.name == "nt":
                os.startfile(str(CSV_FILE))
            else:
                subprocess.Popen(["xdg-open", str(CSV_FILE)])
        except Exception as e:
            messagebox.showerror("Open CSV Error", str(e))

    def _log_msg(self, msg: str):
        self.log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _on_close(self):
        self.is_running = False
        self.destroy()


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=" * 70, flush=True)
    print("  KNOM BUABARN - COBOT VISION & PICK-AND-PLACE PRODUCTION SERVICE", flush=True)
    print("=" * 70, flush=True)
    app = KnomBuabarnApp()
    app.mainloop()


if __name__ == "__main__":
    main()
