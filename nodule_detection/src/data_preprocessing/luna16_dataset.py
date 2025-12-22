import os
import json
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random

class LUNA16Dataset(Dataset):  #inherits PyTorch Dataset class to use DataLoaders
    #PyTorch Dataset class for LUNA16 3D scans with segmentation masks
    def __init__(
            self,
            scan_dirs: List[str],
            mask_dirs: str,
            split_file: str,
            split_type: str = 'train',  #train, test or val
            patch_size: Tuple[int, int, int] = (128, 128, 128),
            num_patches_per_scan: int = 6,
            positive_ratio: float = 0.5,  #target ratio of patches containing nodules
            hu_min: int = -1000,  #min and max hu value for clipping
            hu_max: int = 400,
            augment: bool = True,
            seed: Optional[int] = None
    ):
        #validates input
        if not (0.0 <= positive_ratio <= 1.0):
            raise ValueError("Positive ratio must be between 0.0 and 1.0")
        if not all(s > 0 for s in patch_size):
            raise ValueError("Patch size dimensions must be positive")
        if num_patches_per_scan < 0:
            raise ValueError("Number of patches per scan must be positive")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.seed = seed
        self.scan_dirs = [Path(d) for d in scan_dirs]
        self.mask_dirs = Path(mask_dirs)
        self.split_type = split_type
        self.patch_size = patch_size
        self.num_patches_per_scan = num_patches_per_scan
        self.positive_ratio = positive_ratio
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.augment = augment

        with open(split_file, 'r') as f:  #loads splits from json (ensures reproducibility, shareability, no data leakage)
            splits = json.load(f)
        self.series_uids = splits[split_type]  #list of series UIDs for given split
        
        self.scan_paths = self.build_scan_index()
        self.series_uids_list = list(self.scan_paths.keys())  #keys in dict are UID
        
    def __len__(self) -> int:
        return len(self.scan_paths) * self.num_patches_per_scan
    
    def __getitem__(self, index: str) -> Dict[str, torch.Tensor]:
        #skeleton, gets single training patch - PyTorch calls this to get data during training
        scan_index = index // self.num_patches_per_scan  #this way indices 0-3 come from scan 1, 4-7 from scan 2 etc.
        series_uid = self.series_uids_list[scan_index]

        scan, mask = self.load_scans_and_masks(series_uid)
        scan = self.normalize_hu(scan)

        force_positive = random.random() < self.positive_ratio  #50% of scans will contain nodules
        scan_patch, mask_patch = self.extract_random_patch(scan, mask, force_positive=force_positive)
        if self.augment and self.split_type == 'train':
            scan_patch, mask_patch = self.augment_patch(scan_patch, mask_patch)
        
        #converts to Pytorch tensor and channels dimension, as PyTorch expects (channels, depth, height, width)
        scan_tensor = torch.from_numpy(scan_patch).unsqueeze(0)  #(1, D, H, W)
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)

        return {
            'scan': scan_tensor,
            'mask': mask_tensor,
            'series_uid': series_uid
        }

    def build_scan_index(self) -> Dict[str, Dict[str, Path]]:
        #builds mapping from series unique ID to scan file path and subset
        #it builds index to quickly look up scan file paths by their unique series ID
        scan_paths = {}
        for scan_dir in self.scan_dirs:
            # Extract subset number and normalize (e.g., 'subset0_downsampled' -> 'subset0')
            dir_name = scan_dir.name
            subset_num = ''.join(filter(str.isdigit, dir_name))
            subset_name = f'subset{subset_num}' if subset_num else dir_name

            for mhd_file in scan_dir.glob('*.mhd'):
                series_uid = mhd_file.stem  #gets the filename without extension
                if series_uid in self.series_uids:
                    scan_paths[series_uid] = {
                        'scan_path': mhd_file,
                        'subset': subset_name
                    }

        return scan_paths
    
    def load_scans_and_masks(self, series_uid: str) -> Tuple[np.ndarray, np.ndarray]:
        #loads ct scan and corresponding mask from the same subset
        scan_info = self.scan_paths[series_uid]
        scan_path = scan_info['scan_path']
        subset_name = scan_info['subset']

        scan_sitk = sitk.ReadImage(str(scan_path))
        scan_array = sitk.GetArrayFromImage(scan_sitk).astype(np.float32)

        mask_filename = f"{series_uid}_mask.npy"
        mask_path = self.mask_dirs / subset_name / mask_filename
        mask_array = np.load(mask_path).astype(np.float32)

        return scan_array, mask_array
    
    def normalize_hu(self, scan: np.ndarray) -> np.ndarray:
        #normalizes HU values to [0, 1] range
        #Hounsfield unit values measure radiodensity, how much radiation different tissues absorb
        scan = np.clip(scan, self.hu_min, self.hu_max) #clip HU
        scan = (scan - self.hu_min) / (self.hu_max - self.hu_min) #normalize
        return scan
    
    def extract_random_patch(self, scan: np.ndarray, mask: np.ndarray, force_positive: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        d, h, w = scan.shape
        patch_d, patch_h, patch_w = self.patch_size

        if force_positive and mask.sum() > 0:
            pos_voxels_coords = np.argwhere(mask > 0)  #gets coordinates of positive voxels
            anchor_index = np.random.randint(len(pos_voxels_coords))  #randomly selects voxel as anchor
            anchor_z, anchor_y, anchor_x = pos_voxels_coords[anchor_index]  #unpacks coords
            
            #centers patch around anchor (with randomness, so nodules aren't always in the center)
            center_z = anchor_z + np.random.randint(-patch_d//4, patch_d//4)
            center_y = anchor_y + np.random.randint(-patch_h//4, patch_h//4)
            center_x = anchor_x + np.random.randint(-patch_w//4, patch_w//4)

            z_start = max(0, min(d - patch_d, center_z - patch_d//2))  #calculates patch boundaries
            y_start = max(0, min(h - patch_h, center_y - patch_h//2))
            x_start = max(0, min(w - patch_w, center_x - patch_w//2))
        else:
            z_start = np.random.randint(0, max(1, d - patch_d + 1))
            y_start = np.random.randint(0, max(1, h - patch_h + 1))
            x_start = np.random.randint(0, max(1, w - patch_w + 1))

        z_end = min(d, z_start + patch_d)  #edge cases where the scan is smaller than the patch
        y_end = min(h, y_start + patch_h)
        x_end = min(w, x_start + patch_w)

        scan_patch = scan[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask[z_start:z_end, y_start:y_end, x_start:x_end]

        if scan_patch.shape != self.patch_size:
            scan_patch = self.pad_to_size(scan_patch, self.patch_size)
            mask_patch = self.pad_to_size(mask_patch, self.patch_size)

        return scan_patch, mask_patch
    
    def pad_to_size(self, array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        #if extracted patch is smaller than desired size, pad it with 0s to match expected dimensions
        pad_width = []
        for i in range(3):
            diff = target_size[i] - array.shape[i]  #calculates how much padding each dimension needs
            pad_before = diff // 2  #tells you how much 0s are needed before and after array
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))

        return np.pad(array, pad_width, mode='constant', constant_values=0)
    
    def augment_patch(self, scan_patch: np.ndarray, mask_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #randomly transforms patched during training to create variations, dataset size stays 888 and augmentation is applied on-the-fly
        #this means every time when patch is loaded it gets different augmentations - over epochs, model sees thousands of variations
        for axis in range(3):
            if random.random() > 0.5:
                scan_patch = np.flip(scan_patch, axis=axis).copy()  #axis 0(upside-down flip), axis 1(front-back), axis 2(mirror)
                mask_patch = np.flip(mask_patch, axis=axis).copy()
        
        k = random.randint(0, 3)  #random 90° rotation in z-axis
        if k > 0:
            scan_patch = np.rot90(scan_patch, k=k, axes=(1,2)).copy()
            mask_patch = np.rot90(mask_patch, k=k, axes=(1,2)).copy()
        
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)  #randomly brightens or darkens scan up to 10% (only scans)
            scan_patch = np.clip(scan_patch + shift, 0, 1)
        
        return scan_patch, mask_patch

def create_dataloaders(
        scan_dirs: List[str], mask_dirs: str, split_file: str, batch_size: int=4, num_workers: int=6, **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, ...]:
    #creates train, test and validation dataloaders
    train_dataset = LUNA16Dataset(scan_dirs=scan_dirs, mask_dirs=mask_dirs, split_file=split_file, split_type='train', augment=True, **dataset_kwargs)
    val_dataset = LUNA16Dataset(scan_dirs=scan_dirs, mask_dirs=mask_dirs, split_file=split_file, split_type='val', augment=False, **dataset_kwargs)
    test_dataset = LUNA16Dataset(scan_dirs=scan_dirs, mask_dirs=mask_dirs, split_file=split_file, split_type='test', augment=False, **dataset_kwargs)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    BASE_DIR = r"C:\Users\emirb\OneDrive\Desktop\coding\Python\DSAI\projects\data"
    SCAN_DIRS = [
        os.path.join(BASE_DIR, "luna16_downsampled", f"subset{i}_downsampled")
        for i in range(10)
    ]
    MASK_DIR = os.path.join(BASE_DIR, "luna16_masks")
    SPLIT_FILE = os.path.join(BASE_DIR, "luna16_splits.json")

    print(f"\nScan directories: {len(SCAN_DIRS)} subsets")
    print(f"Mask directory: {MASK_DIR}")
    print(f"Split file: {SPLIT_FILE}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        scan_dirs=SCAN_DIRS,
        mask_dirs=MASK_DIR,
        split_file=SPLIT_FILE,
        batch_size=4,
        num_workers=6,
        patch_size=(128, 128, 128),
        num_patches_per_scan=4,
        seed=42
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")