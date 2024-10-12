import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

import cv2
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class ViFam(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.vid_paths = []
        self.labels = []
    
    @property
    def _get_label(self):
        for label in self.root_dir.iterdir():
            # Check if the subpath is directory
            if not label.is_dir():
                print(f'Error: The datasets\' structure are not right. The {label} is not directory')
                continue
            else:
                self.labels.append(label.name)
                
        self.label2idx: Dict = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label: Dict = {idx: label for idx, label in enumerate(self.labels)}
                
    @property
    def _get_path(self):
        for label in self.labels:
            label_dir = self.root_dir / label
            videos: List = list(label_dir.glob('**.*'))
            
            
            
            
            
class ViFamImageCls(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.vid_path = []
        self.label = []


class ViFamVideoCls(Dataset):
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
