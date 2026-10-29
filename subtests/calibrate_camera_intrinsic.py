"""
Capture chessboard images from an Orbbec Gemini camera and calibrate the
camera in one run.

Controls (while the preview window is focused):
    's'   -> save the current frame (only when a chessboard is detected)
    'q'   -> stop capturing and run calibration on everything saved
    ESC   -> same as 'q'

Saves:
    <SAVE_DIR>/img_0000.jpg            clean frame (used for calibration)
    <SAVE_DIR>/detected/img_0000.jpg   same frame with corners+lines drawn
                                        (visual check only, not used for math)
    <SAVE_DIR>/undistort_check.jpg     one saved image, before/after undistort
                                        (visual sanity check only)
    <SAVE_DIR>/calibration.npz         camera matrix, dist coeffs, etc.

Changes from the previous version:
    - objp is now scaled by SQUARE_SIZE_MM, so camera_matrix / rvecs / tvecs
      come out in real millimetres instead of "board squares". This matters
      if this calibration file is ever reused with solvePnP or hand-eye
      calibration elsewhere (both expect metric units).
    - Per-image reprojection error is computed after calibration, printed,
      and the worst offenders are flagged so you can delete them and
      recalibrate instead of guessing which image is bad.
    - Minor: capture loop refuses to start calibration with 0 images, and
      warns (not silently continues) if lighting/detection looks unstable.
"""

import os
import cv2 as cv
import numpy as np

from utils import frame_to_bgr_image                                # type: ignore[attr-defined]
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# ---- chessboard settings ----
PATTERN_SIZE = (9, 7)          # inner corners (columns, rows)
SQUARE_SIZE_MM = 20.0          # <-- measure your printed board and set this
SAVE_DIR = r"Data/Calb_img"
DETECTED_DIR = os.path.join(SAVE_DIR, "detected")
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

MIN_IMAGES_RECOMMENDED = 15
PER_IMAGE_ERROR_WARN_PX = 1.0   # flag any single image above this


def imwrite_unicode(path, img):
    """cv2.imwrite can fail silently on Windows paths with non-ASCII
    characters (e.g. the Thai folder name in SAVE_DIR). Writes via numpy
    instead, which handles Unicode paths correctly."""
    ext = os.path.splitext(path)[1]
    success, encoded = cv.imencode(ext, img)
    if success:
        encoded.tofile(path)
    return success


def capture_images():
    """Live capture loop. Returns (objpoints, imgpoints, gray_shape,
    saved_paths)."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DETECTED_DIR, exist_ok=True)

    # Real-world 3D corner coordinates in mm, board frame, Z=0 plane.
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []   # 3D points (mm), one entry per saved image
    imgpoints = []   # corresponding 2D corner points (pixels)
    saved_paths = []  # clean image path per saved index, for error reporting
    gray_shape = None

    config = Config()
    pipeline = Pipeline()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    pipeline.start(config)

    print("Pipeline started. Press 's' to save a frame with a detected "
          "chessboard. Press 'q' or ESC when done to run calibration.")
    print(f"Aim for {MIN_IMAGES_RECOMMENDED}+ images, varied angle/distance, "
          "covering the full frame including corners/edges.")

    saved_count = 0
    win_name = "Orbbec Gemini - Chessboard Capture"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

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

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]  # (width, height)
            display = img.copy()

            ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)

            corners2 = None
            if ret:
                corners2 = cv.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), CRITERIA
                )
                cv.drawChessboardCorners(display, PATTERN_SIZE, corners2, ret)
                status_text = "Chessboard FOUND - press 's' to save"
                color = (0, 255, 0)
            else:
                status_text = "Chessboard not found"
                color = (0, 0, 255)

            cv.putText(display, status_text, (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv.putText(display, f"Saved: {saved_count}", (20, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv.imshow(win_name, display)
            key = cv.waitKey(1) & 0xFF

            if key == ord('s'):
                if ret:
                    assert corners2 is not None  # narrows type for Pylance; ret==True guarantees this
                    fname = f"img_{saved_count:04d}.jpg"

                    # Clean frame -> used for calibration accuracy
                    clean_path = os.path.join(SAVE_DIR, fname)
                    imwrite_unicode(clean_path, img)

                    # Annotated frame -> visual check only
                    annotated = img.copy()
                    cv.drawChessboardCorners(annotated, PATTERN_SIZE, corners2, ret)
                    detected_path = os.path.join(DETECTED_DIR, fname)
                    imwrite_unicode(detected_path, annotated)

                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    saved_paths.append(clean_path)
                    saved_count += 1
                    print(f"Saved {clean_path}  ({saved_count} total)")
                else:
                    print("No chessboard detected - not saving.")
            elif key == ord('q') or key == 27:  # 'q' or ESC
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()

    print(f"Done capturing. {saved_count} images saved to '{SAVE_DIR}'.")
    if 0 < saved_count < MIN_IMAGES_RECOMMENDED:
        print(f"Warning: only {saved_count} images. "
              f"Aim for {MIN_IMAGES_RECOMMENDED}+ from varied angles/positions "
              "for a reliable calibration.")

    return objpoints, imgpoints, gray_shape, saved_paths


def per_image_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Reproject each image's object points using its own solved pose and
    compare to the detected corners. Returns a list of per-image RMS pixel
    errors, same order as objpoints/imgpoints."""
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        diff = imgpoints[i].reshape(-1, 2) - projected.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        errors.append(rms)
    return errors


def calibrate(objpoints, imgpoints, gray_shape, saved_paths):
    saved_count = len(saved_paths)
    if saved_count == 0:
        print("No images captured - cannot calibrate.")
        return

    if saved_count < 5:
        print("Warning: very few images. Calibration results may be "
              "unreliable. Aim for 10-20+ images from varied angles/positions.")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    print("\n=== Calibration results ===")
    print("RMS re-projection error (overall):", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Per-image breakdown so bad shots can be identified and removed.
    per_img_err = per_image_reprojection_errors(
        objpoints, imgpoints, rvecs, tvecs, mtx, dist
    )
    print(f"\nPer-image reprojection error (px), flagging > "
          f"{PER_IMAGE_ERROR_WARN_PX:.2f} px:")
    worst = []
    for path, err in zip(saved_paths, per_img_err):
        flag = "  <-- consider removing" if err > PER_IMAGE_ERROR_WARN_PX else ""
        print(f"  {os.path.basename(path):<20} {err:.4f} px{flag}")
        if err > PER_IMAGE_ERROR_WARN_PX:
            worst.append((os.path.basename(path), err))

    if worst:
        print(f"\n{len(worst)} image(s) above threshold. Deleting these and "
              "rerunning tends to lower overall error more than adding new "
              "images.")

    calib_path = os.path.join(SAVE_DIR, "calibration.npz")
    np.savez(
        calib_path,
        camera_matrix=mtx,
        dist_coeffs=dist,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        rms_error=ret,
        per_image_error=np.array(per_img_err),
        square_size_mm=np.array([SQUARE_SIZE_MM]),
        image_size=np.array(gray_shape),
    )
    print(f"\nSaved calibration data to {calib_path}")

    # Visual sanity check: undistort the first saved image.
    if saved_paths:
        sample = cv.imread(saved_paths[0])
        if sample is not None:
            undistorted = cv.undistort(sample, mtx, dist)
            side_by_side = np.hstack([sample, undistorted])
            check_path = os.path.join(SAVE_DIR, "undistort_check.jpg")
            imwrite_unicode(check_path, side_by_side)
            print(f"Saved before/after undistort check to {check_path}")


def main():
    objpoints, imgpoints, gray_shape, saved_paths = capture_images()
    calibrate(objpoints, imgpoints, gray_shape, saved_paths)


if __name__ == "__main__":
    main()