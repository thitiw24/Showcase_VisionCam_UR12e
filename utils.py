# =============================================================================
# utils.py - Knom_Buabarn (Robot Vision & Hardware Helpers)
# =============================================================================
"""
Utility functions for Orbbec Gemini 2 camera, image conversion,
and fail-safe 3D depth extraction.
"""

from typing import Union, Optional, Any
import numpy as np
import cv2 as cv

try:
    from pyorbbecsdk import (
        VideoFrame,
        OBFormat,
        OBConvertFormat,
        FormatConvertFilter,
    )
    ORBBEC_AVAILABLE = True
except ImportError:
    VideoFrame = OBFormat = OBConvertFormat = FormatConvertFilter = None
    ORBBEC_AVAILABLE = False


def determine_convert_format(frame: Any) -> Optional[Any]:
    if not ORBBEC_AVAILABLE or frame is None:
        return None
    fmt = frame.get_format()
    if fmt == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif fmt == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif fmt == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif fmt == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif fmt == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif fmt == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    return None


def frame_to_bgr_image(frame: Any) -> Optional[np.ndarray]:
    """Convert Orbbec VideoFrame to OpenCV BGR numpy array."""
    if frame is None:
        return None
    try:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        raw_data = frame.get_data()
        if raw_data is None:
            return None

        data = np.asanyarray(raw_data)
        if data.size == 0:
            return None

        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            return cv.cvtColor(image, cv.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            return np.resize(data, (height, width, 3))
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            return cv.cvtColor(image, cv.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            return cv.imdecode(data, cv.IMREAD_COLOR)
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            return cv.cvtColor(image, cv.COLOR_YUV2BGR_UYVY)
        else:
            return None
    except Exception:
        return None


def safe_frame_to_bgr_image(frame: Any) -> Optional[np.ndarray]:
    """Protected wrapper for frame_to_bgr_image to prevent crashes."""
    try:
        return frame_to_bgr_image(frame)
    except Exception:
        return None


# =============================================================================
# CASE 2: SAFE DEPTH IMAGE EXTRACTION
# =============================================================================
def get_valid_depth_image(depth_frame: Any, expected_w: int = 1280, expected_h: int = 800) -> Optional[np.ndarray]:
    """
    Safely extract uint16 depth image with strict byte size validation.
    Prevents ValueError: buffer size must be a multiple of element size / cannot reshape.
    
    Returns:
        np.ndarray (H, W) dtype=uint16 if frame is completely valid.
        None if frame is torn, incomplete, or corrupted.
    """
    if depth_frame is None:
        return None

    try:
        w = depth_frame.get_width()
        h = depth_frame.get_height()

        # Fallback to parameters if frame width/height invalid
        if w <= 0:
            w = expected_w
        if h <= 0:
            h = expected_h

        # uint16 is 2 bytes per pixel
        expected_bytes = w * h * 2
        raw_data = depth_frame.get_data()

        if raw_data is None:
            return None

        # 1. Byte Guard: Verify exact byte length received from USB
        if len(raw_data) != expected_bytes:
            # Drop truncated/torn frame immediately without crashing
            return None

        # 2. Reshape Guard: Parse into uint16 numpy array
        depth_raw = np.frombuffer(raw_data, dtype=np.uint16).reshape((h, w))
        return depth_raw

    except (ValueError, TypeError, Exception):
        return None


def get_depth_median_at(depth_mm: np.ndarray, cx: int, cy: int, radius: int = 4) -> Optional[float]:
    """Calculate median depth (mm) within a patch around (cx, cy) to filter noise."""
    if depth_mm is None:
        return None
    h, w = depth_mm.shape
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))
