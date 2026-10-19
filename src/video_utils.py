import cv2
import numpy as np
import torch

from configs.config import (
    NUM_FRAMES,
    FRAME_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def load_video_frames(
    video_path,
    num_frames=NUM_FRAMES,
    resize=FRAME_SIZE,
):
    frames = []
    last_frame = np.zeros(
        (resize[1], resize[0], 3),
        dtype=np.uint8,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise RuntimeError(
                f"Video contains no readable frames: {video_path}"
            )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
        ).astype(int)

        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if not success or frame is None:
                frame = last_frame.copy()
            else:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_GRAY2RGB,
                    )
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported video frame format."
                    )

                frame = cv2.resize(frame, resize)
                last_frame = frame.copy()

            frame = frame.astype(np.float32) / 255.0
            frame = (
                frame - np.asarray(IMAGENET_MEAN, dtype=np.float32)
            ) / np.asarray(IMAGENET_STD, dtype=np.float32)

            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

    finally:
        cap.release()

    return torch.from_numpy(
        np.stack(frames).astype(np.float32)
    )