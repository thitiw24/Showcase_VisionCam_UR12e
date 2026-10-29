import os
import cv2 as cv
import numpy as np
import time
# import tkinter as tk

# from tkinter import messagebox
from utils import frame_to_bgr_image                                # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
from datetime import datetime

SAVE_DIR = r"Img"
VIDEO_DIR = r"Video"

def imwrite_unicode(path, img):
    """cv2.imwrite can fail silently on Windows paths with non-ASCII
    characters (e.g. the Thai folder name in SAVE_DIR). Writes via numpy
    instead, which handles Unicode paths correctly."""
    ext = os.path.splitext(path)[1]
    success, encoded = cv.imencode(ext, img)
    if success:
        encoded.tofile(path)
    return success

def get_next_start_index(save_dir, prefix, ext):
    existing = [f for f in os.listdir(save_dir)
                if f.startswith(prefix) and f.endswith(ext)]
    max_idx = -1
    for f in existing:
        num_part = f[len(prefix):-len(ext)]
        if num_part.isdigit():
            max_idx = max(max_idx, int(num_part))
    return max_idx + 1

def main():
    saved_count = 0 

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    config = Config()
    pipeline = Pipeline()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)
    print("Pipeline started. 's' = save frame, 'r' = start/stop recording, 'q'/ESC = quit.")

    # saved_count = get_next_start_index(SAVE_DIR, "img_", ".jpg")
    # video_count = get_next_start_index(VIDEO_DIR, "vid_", ".mp4")

    win_name = "Orbbec Gemini - Sample Capture"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    # --- video recording state ---
    recording = False
    video_writer = None
    video_path = None  
    fps_estimate = 30  # adjust to your actual stream fps if known

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            img = frame_to_bgr_image(color_frame)
            if img is None:
                continue

            display = img.copy()
            cv.putText(display, f"Saved: {saved_count}", (20, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if recording:
                cv.putText(display, "REC", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                assert video_writer is not None  # recording=True is only set after video_writer is created
                video_writer.write(img)  # write clean frame, not the overlay

            cv.putText(display, "S: Save image | R: Record | Q/ESC: Quit", (20, display.shape[0] - 20), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv.imshow(win_name, display)
            key = cv.waitKey(1) & 0xFF

            if key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:15]  # e.g. 20260803_143205
                filename = os.path.join(SAVE_DIR, f"det_{timestamp}.jpg")
                imwrite_unicode(filename, img)
                print(f"Saved {filename}")
                saved_count += 1

            elif key == ord('r'):
                if not recording:
                    h, w = img.shape[:2]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:15]
                    video_path = os.path.join(VIDEO_DIR, f"vid_{timestamp}.mp4")
                    fourcc = cv.VideoWriter_fourcc(*"mp4v")                                 # type: ignore[attr-defined]
                    video_writer = cv.VideoWriter(video_path, fourcc, fps_estimate, (w, h))
                    recording = True
                    print(f"Recording started -> {video_path}")
                else:
                    recording = False
                    assert video_writer is not None  # recording=True is only set after video_writer is created
                    video_writer.release()
                    video_writer = None
                    assert video_path is not None
                    print(f"Recording stopped. Saved {os.path.basename(video_path)}.mp4")
                    video_path = None

            elif key == ord('q') or key == 27:
                break

    finally:
        if recording and video_writer is not None:
            video_writer.release()
            print("Recording stopped on exit.")
        pipeline.stop()
        cv.destroyAllWindows()
        print(f"Done. {saved_count} images saved to '{SAVE_DIR}/'.")


if __name__ == "__main__":
    main()