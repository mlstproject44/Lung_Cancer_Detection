import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from scipy import ndimage
from tqdm import tqdm

BASE_DIR = "/home/jovyan/vol.2"

def normalize_hu(scan: np.ndarray, hu_min: int = -1000, hu_max: int = 400) -> np.ndarray:
    scan = np.clip(scan, hu_min, hu_max)
    return (scan - hu_min) / (hu_max - hu_min)

def pad_to_size(array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    pad_width = []
    for i in range(3):
        diff = target_size[i] - array.shape[i]
        pad_before = diff // 2
        pad_after = diff - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def find_nodule_centers(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    labeled, n = ndimage.label(mask > 0)
    centers = []
    for i in range(1, n + 1):
        voxels = np.argwhere(labeled == i)
        center = voxels.mean(axis=0).astype(int)
        centers.append(tuple(center))
    return centers

def extract_patch(scan, mask, center, patch_size):
    d, h, w = scan.shape
    pd, ph, pw = patch_size
    cz, cy, cx = center

    z_start = max(0, min(d - pd, cz - pd // 2))
    y_start = max(0, min(h - ph, cy - ph // 2))
    x_start = max(0, min(w - pw, cx - pw // 2))

    scan_patch = scan[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]
    mask_patch = mask[z_start:z_start + pd, y_start:y_start + ph, x_start:x_start + pw]

    if scan_patch.shape != patch_size:
        scan_patch = pad_to_size(scan_patch, patch_size)
        mask_patch = pad_to_size(mask_patch, patch_size)

    return scan_patch, mask_patch

def extract_patches_from_scan(scan: np.ndarray, mask: np.ndarray, series_uid: str, num_patches: int,
    positive_ratio: float, patch_size: Tuple[int, int, int],
):
    pd, ph, pw = patch_size

    nodule_centers = find_nodule_centers(mask)
    has_nodules = len(nodule_centers) > 0
    num_positive = int(num_patches * positive_ratio) if has_nodules else 0
    num_negative = num_patches - num_positive

    patches = []

    #---- positive patches ----
    if num_positive > 0:
        per_nodule = num_positive // len(nodule_centers)
        remainder = num_positive % len(nodule_centers)

        for nod_idx, center in enumerate(nodule_centers):
            n_patches = per_nodule + (1 if nod_idx < remainder else 0)
            cz, cy, cx = center

            for _ in range(n_patches):
                for _ in range(20):
                    jz = cz + np.random.randint(-pd // 4, pd // 4 + 1)
                    jy = cy + np.random.randint(-ph // 4, ph // 4 + 1)
                    jx = cx + np.random.randint(-pw // 4, pw // 4 + 1)

                    scan_patch, mask_patch = extract_patch(scan, mask, (jz, jy, jx), patch_size)

                    if mask_patch.sum() > 0:
                        patches.append({
                            'scan': scan_patch.astype(np.float32),
                            'mask': (mask_patch * 255).astype(np.uint8),
                            'series_uid': series_uid,
                            'has_nodule': True,
                        })
                        break
                else:
                    scan_patch, mask_patch = extract_patch(scan, mask, (cz, cy, cx), patch_size)
                    patches.append({
                        'scan': scan_patch.astype(np.float32),
                        'mask': (mask_patch * 255).astype(np.uint8),
                        'series_uid': series_uid,
                        'has_nodule': True,
                    })

    #---- negative patches ----
    lung_voxels = np.argwhere(scan > 0)

    if len(lung_voxels) > 0:
        for _ in range(num_negative):
            for _ in range(20):
                idx = np.random.randint(0, len(lung_voxels))
                cz, cy, cx = lung_voxels[idx]

                scan_patch, mask_patch = extract_patch(scan, mask, (cz, cy, cx), patch_size)
                lung_ratio = (scan_patch > 0).sum() / scan_patch.size
                if lung_ratio >= 0.3 and mask_patch.sum() == 0:
                    patches.append({
                        'scan': scan_patch.astype(np.float32),
                        'mask': mask_patch.astype(np.uint8),
                        'series_uid': series_uid,
                        'has_nodule': False,
                    })
                    break

    return patches

def save_all_patches(parenchyma_dir: str, mask_dir: str, split_file: str, output_dir: str, patch_size: Tuple[int, int, int] = (64, 64, 64),
    num_patches_per_scan: int = 80, positive_ratio: float = 0.7, seed: int = 42,
):
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(split_file, 'r') as f:
        splits = json.load(f)

    print(f"Patch size: {patch_size}")
    print(f"Patches per scan: {num_patches_per_scan}")
    print(f"Positive ratio: {positive_ratio}")

    for split_type in ['train', 'val', 'test']:
        print(f"\nProcessing {split_type.upper()}")

        split_dir = output_path / split_type
        split_dir.mkdir(exist_ok=True)

        series_uids = splits[split_type]

        scan_index = {}
        parenchyma_base = Path(parenchyma_dir)
        mask_base = Path(mask_dir)

        for subset_dir in sorted(parenchyma_base.iterdir()):
            if not subset_dir.is_dir():
                continue
            for npy_file in subset_dir.glob("*_parenchyma.npy"):
                uid = npy_file.stem.replace("_parenchyma", "")
                if uid in series_uids:
                    subset_name = subset_dir.name
                    mask_path = mask_base / subset_name / f"{uid}_mask.npy"
                    if mask_path.exists():
                        scan_index[uid] = {
                            'parenchyma_path': npy_file,
                            'mask_path': mask_path,
                        }

        print(f"Found {len(scan_index)} scans for {split_type}")

        metadata = []
        patch_counter = 0

        for series_uid in tqdm(scan_index.keys(), desc=f"{split_type}"):
            info = scan_index[series_uid]

            parenchyma = np.load(info['parenchyma_path']).astype(np.float32)
            mask_array = np.load(info['mask_path']).astype(np.float32)

            scan_normalized = normalize_hu(parenchyma)

            patches = extract_patches_from_scan(scan=scan_normalized, mask=mask_array, series_uid=series_uid,
                num_patches=num_patches_per_scan, positive_ratio=positive_ratio, patch_size=patch_size,
            )

            for patch_data in patches:
                filename = f"patch_{patch_counter:06d}.npz"
                np.savez_compressed(
                    split_dir / filename,
                    scan=patch_data['scan'].astype(np.float16),
                    mask=patch_data['mask'],
                )
                metadata.append({
                    'patch_id': patch_counter,
                    'filename': filename,
                    'series_uid': patch_data['series_uid'],
                    'has_nodule': int(patch_data['has_nodule']),
                })
                patch_counter += 1

        with open(split_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        nodule_count = sum(m['has_nodule'] for m in metadata)
        total = len(metadata)
        pct = nodule_count / total * 100 if total > 0 else 0
        print(f"{split_type}: {total} patches, {nodule_count} positive ({pct:.1f}%)")

save_all_patches(
    parenchyma_dir=os.path.join(BASE_DIR, "segmented_lungs"),
    mask_dir=os.path.join(BASE_DIR, "segmented_nodules"),
    split_file=os.path.join(BASE_DIR, "luna16_splits_voxel_stratified.json"),
    output_dir=os.path.join(BASE_DIR, "unet_patches2"),
    patch_size=(96, 96, 96),
    num_patches_per_scan=80,
    positive_ratio=0.9,
    seed=42,
)