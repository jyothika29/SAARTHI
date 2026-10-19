import csv
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALERT_FILE = PROJECT_ROOT / "alerts" / "test_alerts.csv"

# Demonstration threshold only.
# This is NOT a validated violence-detection threshold.
TEST_THRESHOLD = 0.70


def simulate_buzzer():
    """Simulate a buzzer using the computer's terminal."""
    print("\n" + "=" * 45)
    print("!!! SAARTHI SAFETY ALERT !!!")
    print("SIMULATED BUZZER: BEEP! BEEP! BEEP!")
    print("Potential risk requires human review.")
    print("=" * 45)


def log_alert(video_name, score):
    """Record a demonstration alert in a CSV file."""
    ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = ALERT_FILE.exists()

    with open(
        ALERT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Video",
                "Test Score",
                "Review Status"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            video_name,
            f"{score:.4f}",
            "Human review required"
        ])


def main():
    print("SAARTHI - Simulated Alert System")
    print("Demonstration only; scores are not verified risk probabilities.")

    videos = [
        ("test_video.mp4", 0.1077),
        ("test_violence.mp4", 0.7855)
    ]

    for video_name, score in videos:
        print(f"\nVideo: {video_name}")
        print(f"Test score: {score:.4f}")

        if score >= TEST_THRESHOLD:
            print("Configured test threshold reached.")
            simulate_buzzer()
            log_alert(video_name, score)
        else:
            print("Configured test threshold not reached.")

    print(f"\nTest complete. Alert log: {ALERT_FILE}")


if __name__ == "__main__":
    main()