import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.model import HybridVideoClassifier
from configs.config import NUM_FRAMES


MODEL_PATH = BASE_DIR / "models" / "hybrid_model3.pth"

VIDEO_FILES = [
    BASE_DIR / "abuse_detection" / "test_violence.mp4",
    BASE_DIR / "abuse_detection" / "test_video.mp4",
]

FRAME_SIZE = 112


def load_model():
    model = HybridVideoClassifier(
        num_classes=1,
        freeze_backbone=True,
        fine_tune_layer4=True
    )

    try:
        weights = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        weights = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]

    model.load_state_dict(weights, strict=True)
    model.eval()

    return model


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in: {video_path}")

    # Select evenly spaced frames from the entire clip.
    indices = np.linspace(
        0, len(frames) - 1, NUM_FRAMES
    ).astype(int)

    selected = np.stack([frames[i] for i in indices])

    # Convert to [frames, channels, height, width].
    video = torch.from_numpy(selected).float() / 255.0
    video = video.permute(0, 3, 1, 2).contiguous()

    # Shape: [1, frames, channels, height, width].
    return video.unsqueeze(0), len(frames)


def main():
    print("=" * 55)
    print("SAARTHI - PRERECORDED VIDEO MODEL TEST")
    print("=" * 55)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    print("Loading model...")
    model = load_model()
    print("Model loaded successfully.")

    for video_path in VIDEO_FILES:
        print("\n" + "-" * 55)
        print(f"Video: {video_path.name}")

        if not video_path.exists():
            print("SKIPPED: Video file not found.")
            continue

        try:
            video_tensor, total_frames = read_video(video_path)

            print(f"Frames read: {total_frames}")
            print(f"Input tensor shape: {tuple(video_tensor.shape)}")

            with torch.no_grad():
                logits, uncertainty = model(video_tensor)

            raw_logit = float(logits.item())
            uncertainty_score = float(uncertainty.item())

            # Sigmoid converts the raw logit to a value from 0 to 1.
            score = 1.0 / (1.0 + np.exp(-np.clip(raw_logit, -50, 50)))

            print(f"Raw logit: {raw_logit:.4f}")
            print(f"Sigmoid score: {score:.4f}")
            print(f"Uncertainty output: {uncertainty_score:.4f}")

            print(
                "NOTE: These are experimental model outputs, "
                "not verified violence probabilities."
            )

        except Exception as error:
            print(f"VIDEO TEST FAILED: {type(error).__name__}: {error}")

    print("\nTesting finished.")


if __name__ == "__main__":
    main()