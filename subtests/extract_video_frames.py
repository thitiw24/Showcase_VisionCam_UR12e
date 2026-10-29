import cv2
import os

def extract_frames(video_path, output_folder, every_n_frames=1):
    """
    Extract frames from a video and save them as JPG images.

    video_path: path to your video file
    output_folder: folder where JPG frames will be saved
    every_n_frames: save every Nth frame (1 = every frame, 30 = every 30th frame, etc.)
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_count % every_n_frames == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Extracted {saved_count} frames to '{output_folder}'.")


if __name__ == "__main__":
    video_path = "Video/vid_0002.mp4"      # change to your video file
    output_folder = "Video/NoneVideo"  # change to your desired output folder
    every_n_frames = 60               # 1 = every frame, increase to skip frames

    extract_frames(video_path, output_folder, every_n_frames)