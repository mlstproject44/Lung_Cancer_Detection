import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage
from typing import Tuple


def segment_lung_mask(scan_array: np.ndarray, verbose: bool = False) -> np.ndarray:
    """segments lung parenchyma from CT scan using thresholding and morphology"""
    if verbose:
        print(f"  Input shape: {scan_array.shape}")
        print(f"  HU range: [{scan_array.min():.0f}, {scan_array.max():.0f}]")

    #threshold to get lung tissue (air-filled regions)
    binary_mask = (scan_array >= -1000) & (scan_array <= -300)  #lung tissue is typically -1000 to -300 HU

    if verbose:
        print(f"  After thresholding: {binary_mask.sum()} voxels")

    cleaned_mask = np.zeros_like(binary_mask)  #remove small objects and fill holes (process each slice)

    for i in range(binary_mask.shape[0]):  #process slice by slice
        slice_mask = binary_mask[i]

        #remove small objects
        labeled, num_labels = ndimage.label(slice_mask)
        if num_labels > 0:
            #keep only large components (likely lungs)
            label_sizes = np.bincount(labeled.ravel())
            label_sizes[0] = 0  # Ignore background

            #keep components larger than 1000 voxels (adjust if needed)
            large_labels = label_sizes > 1000
            slice_mask = large_labels[labeled]

        #fill holes in lung regions
        slice_mask = ndimage.binary_fill_holes(slice_mask)
        cleaned_mask[i] = slice_mask

    if verbose:
        print(f"  After cleaning: {cleaned_mask.sum()} voxels")

    #closing removes small holes
    struct = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_closing(cleaned_mask, structure=struct, iterations=2)

    #opening removes small protrusions
    lung_mask = ndimage.binary_opening(lung_mask, structure=struct, iterations=2)

    if verbose:
        print(f"  Final mask: {lung_mask.sum()} voxels")
        print(f"  Lung volume: {lung_mask.sum() / lung_mask.size * 100:.2f}%")

    return lung_mask.astype(np.uint8)


def process_single_scan(scan_path: Path, output_dir: Path, verbose: bool = False) -> dict:
    """process a single CT scan to create lung mask"""
    series_uid = scan_path.stem

    if verbose:
        print(f"\nProcessing: {series_uid}")
    scan_sitk = sitk.ReadImage(str(scan_path))
    scan_array = sitk.GetArrayFromImage(scan_sitk)  # (Z, Y, X)

    lung_mask = segment_lung_mask(scan_array, verbose=verbose)  #segment lung
    output_path = output_dir / f"{series_uid}_lung_mask.npy"
    np.save(output_path, lung_mask)

    if verbose:
        print(f"  Saved: {output_path.name}")

    return {
        'series_uid': series_uid,
        'lung_voxels': int(lung_mask.sum()),
        'total_voxels': int(lung_mask.size),
        'lung_percentage': float(lung_mask.sum() / lung_mask.size * 100)
    }


def create_lung_masks_for_subset(
    subset_dir: Path,
    output_dir: Path,
    subset_name: str,
    verbose: bool = True
) -> dict:
    """creates lung masks for all scans in a subset"""

    subset_output_dir = output_dir / subset_name
    os.makedirs(subset_output_dir, exist_ok=True)

    #all .mhd files
    scan_files = list(subset_dir.glob('*.mhd'))

    if len(scan_files) == 0:
        print(f"  No .mhd files found in {subset_dir}")
        return {'total_scans': 0, 'avg_lung_percentage': 0}

    print(f"Processing {subset_name}: {len(scan_files)} scans")
    stats_list = []

    for idx, scan_path in enumerate(scan_files, 1):
        if verbose:
            print(f"[{idx}/{len(scan_files)}]", end=" ")

        try:
            stats = process_single_scan(scan_path, subset_output_dir, verbose=verbose)
            stats_list.append(stats)
        except Exception as e:
            print(f"  ERROR processing {scan_path.stem}: {e}")
            continue

    if stats_list:
        avg_lung_pct = np.mean([s['lung_percentage'] for s in stats_list])
        print(f"\n{subset_name} Summary:")
        print(f"  Processed: {len(stats_list)} scans")
        print(f"  Average lung volume: {avg_lung_pct:.2f}%")

    return {
        'total_scans': len(stats_list),
        'avg_lung_percentage': avg_lung_pct if stats_list else 0
    }


def create_lung_masks_for_all_subsets(
    base_dir: str,
    output_dir: str,
    num_subsets: int = 10,
    verbose: bool = True
) -> None:
    """creates lung masks for all LUNA16 subsets"""
    base_dir_path = Path(base_dir)
    output_dir_path = Path(output_dir)

    if not base_dir_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    os.makedirs(output_dir_path, exist_ok=True)

    overall_stats = {
        'total_scans': 0,
        'total_subsets': 0
    }

    for subset_idx in range(num_subsets):
        subset_dir = base_dir_path / f"subset{subset_idx}_downsampled"

        if not subset_dir.exists():
            print(f"\nSubset directory not found: {subset_dir}")
            continue

        stats = create_lung_masks_for_subset(
            subset_dir,
            output_dir_path,
            f"subset{subset_idx}",
            verbose=verbose
        )

        overall_stats['total_scans'] += stats['total_scans']
        overall_stats['total_subsets'] += 1

    print("Summary")
    print("\n" + "-"*40)
    print(f"Total subsets processed: {overall_stats['total_subsets']}")
    print(f"Total scans processed: {overall_stats['total_scans']}")
    print("\nLung masks saved successfully!")
    print(f"Location: {output_dir_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create lung segmentation masks for LUNA16 CT scans"
    )
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to LUNA16 data (containing subset0_downsampled, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory for lung masks")
    parser.add_argument("--num_subsets", type=int, default=10,
                        help="Number of subsets to process (default: 10)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()
    
    create_lung_masks_for_all_subsets(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_subsets=args.num_subsets,
        verbose=not args.quiet
    )
    print("\nDone!")

if __name__ == "__main__":
    main()
