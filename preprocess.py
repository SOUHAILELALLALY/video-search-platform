import os
import subprocess
import uuid
import pickle
from tqdm import tqdm
import numpy as np
import faiss
from pathlib import Path
from PIL import Image

from utils import embed_image, ensure_dir, load_image_from_path

# CONFIG
VIDEO_DIR = Path("sample_videos")
CLIP_OUTPUT_DIR = Path("clips")
FRAME_OUTPUT_DIR = Path("frames")
INDEX_PATH = Path("index.faiss")
METADATA_PATH = Path("metadata.pkl")
CLIP_LENGTH = 3  # seconds per clip
SAMPLE_RATE = 1  # fps for extracting frames if you change approach

ensure_dir(CLIP_OUTPUT_DIR)
ensure_dir(FRAME_OUTPUT_DIR)


def split_video_to_clips(video_path: Path, out_dir: Path, clip_len: int = 3):
    """
    Use ffmpeg to split video into fixed-length clips.
    Output pattern: out_dir/{video_stem}__{idx}.mp4
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = video_path.stem

    # probe duration
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    proc = subprocess.run(cmd_probe, capture_output=True, text=True)
    duration = float(proc.stdout.strip())

    clips = []
    idx = 0
    start = 0.0
    while start < duration:
        out_file = out_dir / f"{basename}__{idx:05d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(video_path), "-t", str(clip_len), "-c", "copy", str(out_file)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips.append(out_file)
        idx += 1
        start += clip_len
    return clips


def extract_middle_frame(clip_path: Path, out_frame_path: Path):
    """Extract the middle frame of a clip using ffmpeg to a JPEG."""
    # probe duration
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)
    ]
    proc = subprocess.run(cmd_probe, capture_output=True, text=True)
    try:
        duration = float(proc.stdout.strip())
    except:
        duration = 0.0
    t = duration / 2.0 if duration > 0 else 0
    out_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", str(t), "-i", str(clip_path), "-frames:v", "1", str(out_frame_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_frame_path


def build_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # use inner product on L2-normalized vectors = cosine
    # Convert to float32 and add
    index.add(embeddings)
    return index


def main():
    # Gather all videos
    videos = [p for p in VIDEO_DIR.glob("**/*") if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]]
    if not videos:
        print("No videos found in sample_videos/. Put some videos there and re-run.")
        return

    metadata = []  # list of dicts with clip path, source video, start time, duration
    embeddings = []

    for v in tqdm(videos, desc="Processing videos"):
        clips = split_video_to_clips(v, CLIP_OUTPUT_DIR, clip_len=CLIP_LENGTH)
        for clip in clips:
            # Skip empty or very small clips
            if not clip.exists() or clip.stat().st_size < 100:
                continue
            # create frame
            frame_path = FRAME_OUTPUT_DIR / (clip.stem + ".jpg")
            extract_middle_frame(clip, frame_path)
            if not frame_path.exists():
                continue
            # embed frame
            try:
                img = Image.open(frame_path).convert("RGB")
                emb = embed_image(img)
            except Exception as e:
                print("Failed to embed", clip, e)
                continue
            embeddings.append(emb)

            metadata.append({
                "clip_path": str(clip),
                "frame_path": str(frame_path)
            })

    if not embeddings:
        print("No embeddings were created; check ffmpeg and sample videos.")
        return

    embeddings_np = np.stack(embeddings).astype(np.float32)
    index = build_index(embeddings_np)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Index built! Saved to:", INDEX_PATH, METADATA_PATH)


if __name__ == "__main__":
    main()
