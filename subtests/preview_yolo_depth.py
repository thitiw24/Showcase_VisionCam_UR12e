"""
Live YOLO detection from an Orbbec camera with switchable display modes
and single-class filtering.

Controls (while the preview window is focused):
    'm'   -> cycle mode: 1) Detection  2) Distance  3) Object size
    'c'   -> cycle class filter: All -> class 0 -> class 1 -> ... -> All
    's'   -> save the current annotated frame
    'q'   -> quit
    ESC   -> quit
"""

import os
import cv2 as cv
import numpy as np

from ultralytics import YOLO
from utils import frame_to_bgr_image                                # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode
from datetime import datetime
from ultralytics.engine.results import Results

# ---------------- settings ----------------
MODEL_PATH = "Data/best.pt"
CALIB_PATH = "Data/calibration.npz"
SAVE_DIR = r"Results"
CONF_THRESHOLD = 0.5
DEPTH_SAMPLE_RADIUS = 3

MODE_DETECTION = 1
MODE_DISTANCE = 2
MODE_SIZE = 3
MODE_NAMES = {
    MODE_DETECTION: "Detection",
    MODE_DISTANCE: "Distance",
    MODE_SIZE: "Object Size",
}

POINT_COLOR = (0, 255, 255)


# def get_next_start_index(save_dir, prefix="det_", ext=".jpg"):
#     os.makedirs(save_dir, exist_ok=True)
#     existing = [f for f in os.listdir(save_dir)
#                 if f.startswith(prefix) and f.endswith(ext)]
#     max_idx = -1
#     for f in existing:
#         num_part = f[len(prefix):-len(ext)]
#         if num_part.isdigit():
#             max_idx = max(max_idx, int(num_part))
#     return max_idx + 1


def get_depth_at(depth_mm, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_mm.shape
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def draw_label(img, text, x, y):
    """Draw text with a filled background box for readability."""
    (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), POINT_COLOR, -1)
    cv.putText(img, text, (x + 3, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Loading calibration...")
    calib = np.load(CALIB_PATH)
    fx, fy = calib["camera_matrix"][0, 0], calib["camera_matrix"][1, 1]
    print(f"fx={fx:.1f}, fy={fy:.1f}")

    # Build an ordered list of class ids, with None meaning "show all classes"
    class_ids = sorted(model.names.keys())
    class_options = [None] + class_ids  # None = "All", then each class one by one
    class_index = 0  # start at "All"

    config = Config()
    pipeline = Pipeline()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    config.enable_stream(color_profiles.get_default_video_stream_profile())

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(depth_profiles.get_default_video_stream_profile())

    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.start(config)

    saved_count = 0
    mode = MODE_DETECTION

    win_name = "Orbbec - Live Detection"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    print("Pipeline started. 'm' = switch mode, 'c' = switch class, 's' = save, 'q'/ESC = quit.")

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

            current_class = class_options[class_index]
            current_label = "All" if current_class is None else model.names[current_class]
            class_filter = None if current_class is None else [current_class]

            results = model.predict(source=img, conf=CONF_THRESHOLD, device="cpu",
                                     verbose=False, classes=class_filter)
            r = results[0]                                                          # type: ignore[index]
            assert isinstance(r, Results)  # predict() without stream=True always returns Results objects
            assert r.boxes is not None  # detection model always populates boxes

            if mode == MODE_DETECTION:
                # Full detection view: boxes + class + confidence (built-in)
                annotated = r.plot()

            elif mode == MODE_DISTANCE:
                # No border — just center point + distance text
                annotated = img.copy()

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) #px
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    dist_mm = None
                    if 0 <= cy < depth_mm.shape[0] and 0 <= cx < depth_mm.shape[1]:
                        dist_mm = get_depth_at(depth_mm, cx, cy)
                        cv.circle(annotated, (cx, cy), 5, POINT_COLOR, -1)
                        cv.circle(annotated, (cx, cy), 7, (0, 0, 0), 1)

                    if dist_mm:
                        label = f"Distance from Camera: {dist_mm:.0f} mm"
                        label2 = f"Location: cx = {cx:.2f} px, cy = {cy:.2f} px"
                    else:
                        label = "Distance from Camera: N/A"
                        label2 = "Location: N/A"

                    draw_label(annotated, label, cx + 12, cy - 12)
                    draw_label(annotated, label2, cx + 12, cy + 12)

            else:  # MODE_SIZE
                # Border + size text in mm
                annotated = img.copy()

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    pixel_w, pixel_h = (x2 - x1), (y2 - y1)

                    cv.rectangle(annotated, (x1, y1), (x2, y2), POINT_COLOR, 2)

                    dist_mm = None
                    if 0 <= cy < depth_mm.shape[0] and 0 <= cx < depth_mm.shape[1]:
                        dist_mm = get_depth_at(depth_mm, cx, cy)

                    if dist_mm:
                        real_w_mm = (pixel_w * dist_mm) / fx
                        real_h_mm = (pixel_h * dist_mm) / fy
                        label = f"Size: {real_w_mm:.0f} x {real_h_mm:.0f} mm"
                        # label2 = f"Location: x1 = {x1:.2f}, x2 = {x2:.2f}, y1 = {y1:.2f}, y2 = {y2:.2f}"
                    else:
                        label = "Size: N/A"
                        # label2 = "Location not found"

                    draw_label(annotated, label, x1, y1 - 10)
                    # draw_label(annotated, label2, x2, y2 + 10)

            # --- status overlay (shown in all modes) ---
            cv.putText(annotated,
                       f"Mode {mode}: {MODE_NAMES[mode]}  |  Class: {current_label}  |  "
                       f"Detections: {len(r.boxes)}  |  Saved: {saved_count}",
                       (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv.putText(annotated, "M: Switch Mode | C: Switch Class | S: Save | Q/ESC: Quit",
                       (20, annotated.shape[0] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv.imshow(win_name, annotated)
            key = cv.waitKey(1) & 0xFF

            if key == ord('m'):
                mode = mode + 1 if mode < MODE_SIZE else MODE_DETECTION
                print(f"Switched to Mode {mode}: {MODE_NAMES[mode]}")

            elif key == ord('c'):
                class_index = (class_index + 1) % len(class_options)
                current = class_options[class_index]
                if current is None:
                    new_label = "All"
                else:
                    new_label = model.names[current]
                # new_label = "All" if class_options[class_index] is None else model.names[class_options[class_index]]
                print(f"Class filter switched to: {new_label}")

            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:15]  # e.g. 20260803_143205
                filename = os.path.join(SAVE_DIR, f"det_{timestamp}.jpg")
                cv.imwrite(filename, annotated)
                print(f"Saved {filename}")
                saved_count += 1

            elif key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()
        print(f"Done. {saved_count} image(s) saved to '{SAVE_DIR}/'.")


if __name__ == "__main__":
    main()