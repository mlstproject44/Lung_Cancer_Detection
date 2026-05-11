import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict


class LUNA16PatchDataset(Dataset):
    """
    Loads pre-extracted .npz patches produced by extract_patches.py.

    Patch format (from extract_patches.py):
        scan  – float16, shape (96, 96, 96), already HU-normalised to [0, 1]
        mask  – uint8,   shape (96, 96, 96), values 0 or 255
    """

    def __init__(
        self,
        patch_dir: str,
        split_type: str = 'train',  #'train' | 'val' | 'test'
        augment: bool = True,
    ):
        self.split_dir = Path(patch_dir) / split_type
        # Only augment during training
        self.augment = augment and (split_type == 'train')

        metadata_path = self.split_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {metadata_path}.\n"
                "Run extract_patches.py first."
            )

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Build index sets used by create_patch_dataloaders for weighted sampling
        self.positive_indices = {
            idx for idx, m in enumerate(self.metadata) if m['has_nodule']  # int 1 == True
        }
        self.negative_indices = {
            idx for idx, m in enumerate(self.metadata) if not m['has_nodule']
        }

        print(
            f"  {split_type.upper():5s}: {len(self.metadata)} patches  "
            f"| {len(self.positive_indices)} positive  "
            f"| {len(self.negative_indices)} negative"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]
        data = np.load(self.split_dir / meta['filename'])

        # float16 → float32 for PyTorch; mask uint8 0/255 → float32 0/1
        scan = data['scan'].astype(np.float32)
        mask = data['mask'].astype(np.float32) / 255.0

        if self.augment:
            scan, mask = self._augment(scan, mask)

        return {
            'scan':       torch.from_numpy(scan).unsqueeze(0),   # (1, 96, 96, 96)
            'mask':       torch.from_numpy(mask).unsqueeze(0),   # (1, 96, 96, 96)
            'series_uid': meta['series_uid'],
            'has_nodule': int(meta['has_nodule']),
        }

    def _augment(self, scan: np.ndarray, mask: np.ndarray):
        """
        Augmentations for (96, 96, 96) patches:
          1. Random flips on all 3 axes
          2. Random 90° axial rotation
          3. Random crop-resize (simulates nodules near sliding-window edges)
        """
        # 1. Random flips
        for axis in range(3):
            if np.random.random() > 0.5:
                scan = np.flip(scan, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()

        # 2. Random 90° axial rotation
        k = np.random.randint(0, 4)
        if k > 0:
            scan = np.rot90(scan, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

        # 3. Random crop + resize (p=0.5)
        if np.random.random() > 0.5:
            from scipy.ndimage import zoom
            d = scan.shape[0]                                    # 96
            crop_size = np.random.randint(int(d * 0.87), d)     # 84–95
            if crop_size < d:
                z0 = np.random.randint(0, d - crop_size + 1)
                y0 = np.random.randint(0, d - crop_size + 1)
                x0 = np.random.randint(0, d - crop_size + 1)
                scan = scan[z0:z0+crop_size, y0:y0+crop_size, x0:x0+crop_size]
                mask = mask[z0:z0+crop_size, y0:y0+crop_size, x0:x0+crop_size]
                scale = d / crop_size
                scan = zoom(scan, scale, order=1)   # linear for intensity
                mask = zoom(mask, scale, order=0)   # nearest-neighbour for binary mask

        return scan, mask

def create_patch_dataloaders(
    patch_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    positive_fraction: float = 0.7,   #desired positive ratio in each training batch
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader).
    WeightedRandomSampler re-balances the training split to `positive_fraction`
    regardless of the ratio that extract_patches.py used (0.9 in our case).
    Val/test loaders are unshuffled with no augmentation.
    """
    train_ds = LUNA16PatchDataset(patch_dir, split_type='train', augment=True)
    val_ds   = LUNA16PatchDataset(patch_dir, split_type='val',   augment=False)
    test_ds  = LUNA16PatchDataset(patch_dir, split_type='test',  augment=False)

    # Weighted sampler
    sample_weights = [
        positive_fraction if idx in train_ds.positive_indices else (1.0 - positive_fraction)
        for idx in range(len(train_ds))
    ]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **_loader_kwargs)

    print(f"\n  Loaders | batch={batch_size} | workers={num_workers} | pos_fraction={positive_fraction}")
    print(f"  train={len(train_loader)} batches | val={len(val_loader)} | test={len(test_loader)}")

    return train_loader, val_loader, test_loader