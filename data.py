import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class ViFam(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get all video paths and labels
        self.vid_paths = []
        self.labels = []
        self._get_label()
        self._get_video_path()

    def _get_label(self):
        for label in self.root_dir.iterdir():
            # Check if the subpath is directory
            if not label.is_dir():
                print(
                    f"Error: The datasets' structure are not right. The {label} is not directory"
                )
                continue
            else:
                self.labels.append(label.name)

        self.label2idx: Dict[str, int] = {
            label: idx for idx, label in enumerate(self.labels)
        }
        self.idx2label: Dict[int, str] = {
            idx: label for idx, label in enumerate(self.labels)
        }

    def _get_video_path(self):
        for label in self.labels:
            label_dir = self.root_dir / label
            videos: List[Path] = list(label_dir.glob("*.*"))
            self.vid_paths.extend(videos)

    def __len__(self):
        return len(self.vid_paths)


class ViFamImageCls(ViFam):
    def __init__(self, root_dir: str, transform=None):
        super().__init__(root_dir, transform)
        self.images = self._extract_frames()

    def _extract_frames(self) -> List[Tuple[np.ndarray, int]]:
        frames = []

        for video_path in self.vid_paths:
            video = cv2.VideoCapture(str(video_path))
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Convert from RGB to Gray
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append((gray_frame, self.label2idx[video_path.parent.name]))

            video.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        gray_img, label = self.frames[index]

        if self.transform:
            gray_img = self.transform(gray_img)

        return gray_img, label


class ViFamVideoCls(ViFam):
    def __init__(self, root_dir, transform=None) -> None:
        super().__init__()


def download_dataset(root_dir: str | Path) -> None:
    # Check if the path is valid
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"Error: The specified directory '{root_dir}' does not exist.")
        return
    if not root_dir.is_dir():
        print(f"Error: The specified path '{root_dir}' is not a directory.")
        return

    # Check if git and git-lfs are installed
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE)
        subprocess.run(["git", "lfs", "--version"], check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        # Install git and git lfs
        print(
            "Git or Git LFS is not installed. Please install them before proceeding it"
        )
        return

    # Attempt to clone the repository
    try:
        subprocess.run(
            f"git clone https://huggingface.co/datasets/qhuy242/vifam {root_dir}",
            shell=True,
            check=True,
        )
        print("Dataset downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to clone repository. {e}")
