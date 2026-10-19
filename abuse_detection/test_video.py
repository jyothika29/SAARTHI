import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.model import HybridVideoClassifier
from src.video_utils import load_video_frames
from configs.config import NUM_FRAMES


MODEL_PATH = PROJECT_DIR / "models" / "hybrid_model3.pth"

VIDEO_PATHS = [
    PROJECT_DIR / "abuse_detection" / "test_video.mp4",
    PROJECT_DIR / "abuse_detection" / "test_violence.mp4",
]

DEVICE = torch.device("cpu")


def main():
    print("=" * 55)
    print("SAARTHI - AI VIDEO COMPARISON TEST")
    print("=" * 55)

    if not MODEL_PATH.is_file():
        print("ERROR: Model weights not found:", MODEL_PATH)
        return

    print("\nLoading model...")

    try:
        model = HybridVideoClassifier(
            freeze_backbone=True,
            fine_tune_layer4=True
        )

        weights = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]

        model.load_state_dict(weights, strict=True)
        model.to(DEVICE)
        model.eval()

    except Exception as error:
        print("ERROR loading model:", error)
        return

    print("Model loaded successfully.")

    for video_path in VIDEO_PATHS:
        print("\n" + "-" * 55)
        print("VIDEO:", video_path.name)

        if not video_path.is_file():
            print("ERROR: Video not found.")
            continue

        try:
            frames = load_video_frames(
                str(video_path),
                num_frames=NUM_FRAMES
            )

            frames = frames.unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits, uncertainty = model(frames)

            score = torch.sigmoid(logits.reshape(-1)[0]).item()
            uncertainty_score = uncertainty.reshape(-1)[0].item()

            print(f"Experimental model score: {score:.4f}")
            print(f"Uncertainty output:       {uncertainty_score:.4f}")

        except Exception as error:
            print("ERROR analysing video:", error)

    print("\n" + "=" * 55)
    print("Comparison complete.")
    print("Scores are experimental, not verified violence probabilities.")
    print("=" * 55)


if __name__ == "__main__":
    main()