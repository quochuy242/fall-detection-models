import os
from typing import Optional

import gdown
from torch.utils.data import DataLoader, Dataset


class ViFam(Dataset):
    def __init__(self, root_dir: str, download: bool, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.download = download

    def download_dataset(
        self,
        drive_url: str = "https://drive.google.com/drive/folders/1EHADrJ7p4Jt-rZcduCPyYtf9TI1zYlUr",
        user_agent: Optional[
            str
        ] = "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
    ) -> None:
        os.makedirs(self.root_dir, exist_ok=True)

        if self.download:
            gdown.download_folder(
                url=drive_url,
                output=self.root_dir,
                user_agent=user_agent,
            )
        return None

    def extract_dataset(self, ):
        pass
