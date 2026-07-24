import argparse
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

def extract_frames(video_path: str, output_dir: str, target_fps: int, resize_width: int=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open video file')

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(source_fps / target_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_count = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if resize_width:
                    h, w = frame.shape[:2]
                    scale = resize_width / w
                    frame = cv2.resize(frame, (resize_width, int(h * scale)))

                out_path = output_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Extracted {saved_count} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    extract_frames(video_path=config["paths"]["raw_video"],
                   output_dir=config["paths"]["frames_dir"],
                   target_fps=config["frame_extraction"]["fps"],
                   resize_width=config["frame_extraction"].get("resize_width")
    )