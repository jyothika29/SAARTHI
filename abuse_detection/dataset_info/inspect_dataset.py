from pathlib import Path
from collections import Counter

# ==================================================
# SAARTHI - Dataset Inspection
# ==================================================

# Folder containing this Python file and the text files
BASE_DIR = Path(__file__).resolve().parent

ANNOTATION_FILE = (
    BASE_DIR / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
)

TRAIN_FILE = BASE_DIR / "Anomaly_Train.txt"


# ==================================================
# 1. Inspect Temporal Annotations
# ==================================================

def inspect_annotations(path):
    print("\n" + "=" * 50)
    print("ANNOTATION SUMMARY")
    print("=" * 50)

    if not path.exists():
        print(f"File not found: {path}")
        return

    categories = Counter()
    total_entries = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            # Expected format:
            # video_name category start_frame end_frame start2 end2
            if len(parts) < 4:
                continue

            category = parts[1]

            try:
                start_frame = int(parts[2])
                end_frame = int(parts[3])
            except ValueError:
                continue

            if start_frame < 0 or end_frame < start_frame:
                continue

            categories[category] += 1
            total_entries += 1

    print(f"Annotated entries: {total_entries}\n")

    for category, count in categories.most_common():
        print(f"{category}: {count}")


# ==================================================
# 2. Inspect Training Video List
# ==================================================

def inspect_training_list(path):
    print("\n" + "=" * 50)
    print("TRAINING LIST SUMMARY")
    print("=" * 50)

    if not path.exists():
        print(f"Training list not found: {path}")
        return

    categories = Counter()
    total_entries = 0
    invalid_lines = 0

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            video_path = line.strip()

            # Skip empty lines
            if not video_path:
                continue

            # Support both Windows and Linux path separators
            video_path = video_path.replace("\\", "/")

            # Expected format:
            # Abuse/Abuse001_x264.mp4
            parts = video_path.split("/")

            if len(parts) < 2:
                invalid_lines += 1
                continue

            category = parts[-2]
            filename = parts[-1]

            # Ignore lines that don't look like video paths
            if not filename.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                invalid_lines += 1
                continue

            categories[category] += 1
            total_entries += 1

    print(f"Total training entries: {total_entries}")
    print(f"Invalid or skipped lines: {invalid_lines}\n")

    for category, count in categories.most_common():
        print(f"{category}: {count}")

    print("\n" + "-" * 50)

    # Display categories relevant to our initial experiment
    print("SAARTHI RELEVANT CATEGORIES")

    relevant_categories = [
        "Abuse",
        "Assault",
        "Fighting",
        "Normal",
        "Normal_Videos_event",
    ]

    for category in relevant_categories:
        print(f"{category}: {categories.get(category, 0)}")


# ==================================================
# 3. Main Program
# ==================================================

if __name__ == "__main__":
    print("\nSAARTHI DATASET INSPECTION")
    print("Inspecting dataset text files...")

    inspect_annotations(ANNOTATION_FILE)
    inspect_training_list(TRAIN_FILE)

    print("\nDataset inspection completed.")