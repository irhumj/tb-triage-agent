"""
data.py - loading chest X-rays for the TB triage project.

What this file does:
  1. Reads the frozen split CSVs (data/splits/*.csv).
  2. Opens each X-ray (8-bit TBX11K/Shenzhen and 12-bit Montgomery alike),
     rescales it to 0..1 and resizes it to IMG_SIZE x IMG_SIZE.
  3. Applies augmentation to TRAINING images only.
  4. Returns (image tensor, label) pairs that PyTorch can batch.

Usage (on Kaggle):
    from data import XrayDataset, make_datasets
    ds = make_datasets(splits_dir, roots)
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

# cv2 starts its own threads; inside DataLoader workers that can freeze silently.
cv2.setNumThreads(0)

IMG_SIZE = 224

# Default dataset locations on Kaggle. Override them when running elsewhere.
DEFAULT_ROOTS = {
    "tbx11k": "/kaggle/input/datasets/vbookshelf/tbx11k-simplified/tbx11k-simplified",
    "shenzhen": "/kaggle/input/datasets/kmader/pulmonary-chest-xray-abnormalities",
    "montgomery": "/kaggle/input/datasets/kmader/pulmonary-chest-xray-abnormalities",
}

SPLIT_NAMES = ["train", "internal_test", "val", "shenzhen", "montgomery"]


def load_xray(path, size=IMG_SIZE):
    """Read one X-ray of any bit depth and return a float32 array in 0..1."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keeps Montgomery's 12-bit values
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:  # some PNGs store grayscale as 3 identical channels
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # rescale to 0..1
    # Known limitation: non-square images (e.g. Montgomery) are slightly squashed.
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# Augmentation for training only: small, realistic changes.
# No horizontal flip - the heart is on the left, so a flipped chest is unrealistic.
train_aug = T.Compose([
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
])

# ImageNet statistics, because DenseNet121 was pre-trained on ImageNet photos.
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class XrayDataset(Dataset):
    """One split (e.g. 'train') read from its frozen CSV."""

    def __init__(self, csv_path, roots=None, train=False):
        self.df = pd.read_csv(csv_path)
        self.roots = roots or DEFAULT_ROOTS
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.roots[row["dataset"]], row["relpath"])
        x = torch.from_numpy(load_xray(path))[None]  # shape (1, H, W)
        if self.train:
            x = train_aug(x)
        x = normalize(x.repeat(3, 1, 1))  # gray -> 3 channels, like ImageNet photos
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


def make_datasets(splits_dir, roots=None):
    """Build all five datasets. Only 'train' gets augmentation."""
    return {
        name: XrayDataset(
            os.path.join(splits_dir, f"{name}.csv"),
            roots=roots,
            train=(name == "train"),
        )
        for name in SPLIT_NAMES
    }


if __name__ == "__main__":
    # Quick self-check: python data.py <splits_dir>
    import sys

    splits_dir = sys.argv[1] if len(sys.argv) > 1 else "data/splits"
    ds = make_datasets(splits_dir)
    for name, d in ds.items():
        print(f"{name:14s} {len(d)} images")
