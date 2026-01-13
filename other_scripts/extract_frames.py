import cv2
import os
import argparse


def extract_frames(video_path, out_dir, img_ext="jpg"):
    """
    Extract all frames from a video and save them as images.

    Args:
        video_path (str): path to video file
        out_dir (str): directory to save extracted frames
        img_ext (str): image extension, 'jpg' or 'png'
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.{img_ext}")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print(f"[OK] {video_path}: {frame_idx} frames extracted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, 
                        default="/mnt/nas01-ak/IndividualDirs/wenxueli/eval_100/2k_shortcut_100/decoded_video/1.mp4",
                        help="Directory containing video files")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/nas01-ak/IndividualDirs/wenxueli/eval_100/4k_shortcut_100/frames/2k_motor",
                        help="Output directory for frames")
    parser.add_argument("--ext", type=str, default="png",
                        choices=["jpg", "png"],
                        help="Image format")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # for name in sorted(os.listdir(args.video_dir)):
    #     if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    #         continue

    #     video_path = os.path.join(args.video_dir, name)
    #     video_name = os.path.splitext(name)[0]
    #     out_subdir = os.path.join(args.out_dir, video_name)

    #     extract_frames(video_path, out_subdir, args.ext)
    
    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_subdir = os.path.join(args.out_dir, video_name)

    extract_frames(video_path, out_subdir, args.ext)


if __name__ == "__main__":
    main()