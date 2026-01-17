import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict
from torch.utils.data import WeightedRandomSampler

class LUNA16PatchDataset(Dataset):
    """dataset class that loads pre-extracted .npz patches from the server with weighted sampling to balance positive/negative patches"""
    def __init__(self, patch_dir: str, split_type: str = 'train', augment: bool = True, positive_fraction: float = 0.7):
        self.split_dir = Path(patch_dir) / split_type
        self.augment = augment
        self.split_type = split_type

        #store positive_fraction for logging (actual sampling done by WeightedRandomSampler)
        self.positive_fraction = positive_fraction if split_type == 'train' else None

        metadata_path = self.split_dir / "metadata.json"  #load the metadata file created by save_all_patches function
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. Did you upload it?")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        #separate patches into positive (has nodules) and negative (background only); use metadata
        self.positive_indices = []
        self.negative_indices = []

        print(f"Scanning {len(self.metadata)} patches to separate positive/negative...")
        for idx, meta in enumerate(self.metadata):
            if meta['has_nodule']:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

        print(f"{split_type.upper()}: {len(self.positive_indices)} with nodules, {len(self.negative_indices)} without")

        if self.positive_fraction is not None:
            print(f"Training will sample {self.positive_fraction*100:.0f}% positive, {(1-self.positive_fraction)*100:.0f}% negative patches")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Weighted sampling is handled by WeightedRandomSampler"""
        meta = self.metadata[idx]
        patch_path = self.split_dir / meta['filename']

        data = np.load(patch_path)
        scan_patch = data['scan'].astype(np.float32)
        mask_patch = data['mask'].astype(np.float32) / 255.0   #convert mask from uint8 (0-255) to float32 (0-1)

        if self.augment:
            scan_patch, mask_patch = self.augment_patch(scan_patch, mask_patch)

        scan_tensor = torch.from_numpy(scan_patch).unsqueeze(0)  #add channel dimensions
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)

        return {
            'scan': scan_tensor,
            'mask': mask_tensor,
            'series_uid': meta['series_uid']
        }

    def augment_patch(self, scan_patch, mask_patch):
        """adds random crops to simulate sliding window edge cases"""
        #random flips across 3 axes
        for axis in range(3):
            if np.random.random() > 0.5:
                scan_patch = np.flip(scan_patch, axis=axis).copy()
                mask_patch = np.flip(mask_patch, axis=axis).copy()

        #random 90-degree rotations in the axial plane
        k = np.random.randint(0, 4)
        if k > 0:
            scan_patch = np.rot90(scan_patch, k=k, axes=(1, 2)).copy()
            mask_patch = np.rot90(mask_patch, k=k, axes=(1, 2)).copy()

        #random crop + pad (simulates nodules at edges during sliding window)
        if np.random.random() > 0.5:
            from scipy.ndimage import zoom

            #random crop size between 56-63 (keeps nodules but crops edges)
            crop_size = np.random.randint(56, 64)

            if crop_size < 64:
                d, h, w = scan_patch.shape
                #random crop position
                z_start = np.random.randint(0, d - crop_size + 1)
                y_start = np.random.randint(0, h - crop_size + 1)
                x_start = np.random.randint(0, w - crop_size + 1)

                #crop
                scan_crop = scan_patch[z_start:z_start+crop_size,
                                       y_start:y_start+crop_size,
                                       x_start:x_start+crop_size]
                mask_crop = mask_patch[z_start:z_start+crop_size,
                                       y_start:y_start+crop_size,
                                       x_start:x_start+crop_size]

                #resizes back to 64x64x64 (simulates different nodule positions/scales)
                scale = 64.0 / crop_size
                scan_patch = zoom(scan_crop, scale, order=1)  #linear interpolation for scan
                mask_patch = zoom(mask_crop, scale, order=0)  #nearest for mask (binary)

        return scan_patch, mask_patch

def create_patch_dataloaders(patch_dir: str, batch_size: int = 4, num_workers: int = 4, positive_fraction: float = 0.7):
    """creates the data loaders for the server"""

    train_ds = LUNA16PatchDataset(patch_dir, 'train', augment=True, positive_fraction=positive_fraction)
    val_ds = LUNA16PatchDataset(patch_dir, 'val', augment=False)
    test_ds = LUNA16PatchDataset(patch_dir, 'test', augment=False)

    #creates weighted sampler for training to achieve desired positive/negative ratio
    sample_weights = []
    for idx in range(len(train_ds)):
        is_positive = idx in train_ds.positive_indices
        #Assigns weight based on desired fraction
        weight = positive_fraction if is_positive else (1 - positive_fraction)
        sample_weights.append(weight)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader