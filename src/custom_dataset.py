import os
import pandas as pd
from sunpy.map import Map
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple

class CustomImageDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, time_offset: int, transform: Optional[object] = None) -> None:
        self.labels_df: pd.DataFrame = pd.read_csv(csv_file)
        self.img_dir: str = img_dir
        self.time_offset: int = time_offset
        self.transform: Optional[object] = transform

    def __len__(self) -> int:
        return len(self.labels_df) - self.time_offset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        aia_img_name: str = self.labels_df.iloc[idx, 3] # AIA image path
        hmi_img_name: str = self.labels_df.iloc[idx, 2]

        label_aia_img_name: str = self.labels_df.iloc[idx + self.time_offset, 3] # AIA image path
        label_hmi_img_name: str = self.labels_df.iloc[idx + self.time_offset, 2]
        
        aia_img_path: str = os.path.join(self.img_dir, aia_img_name)
        hmi_img_path: str = os.path.join(self.img_dir, hmi_img_name)
        label_aia_img_path: str = os.path.join(self.img_dir, label_aia_img_name)
        label_hmi_img_path: str = os.path.join(self.img_dir, label_hmi_img_name)

        aia_map = Map(aia_img_path)
        hmi_map = Map(hmi_img_path)
        label_aia_map = Map(label_aia_img_path)
        label_hmi_map = Map(label_hmi_img_path)

        # Reproject HMI to AIA's WCS for alignment
        hmi_map = hmi_map.reproject_to(aia_map.wcs)
        label_hmi_map = label_hmi_map.reproject_to(label_aia_map.wcs)

        # Stack as (2, H, W) numpy array: [AIA, HMI]
        image: torch.Tensor = torch.from_numpy(
            np.stack([aia_map.data, hmi_map.data], axis=0)
        ).float()
        label: torch.Tensor = torch.from_numpy(
            np.stack([label_aia_map.data, label_hmi_map.data], axis=0)
        ).float()

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
