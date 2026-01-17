import os
import sys
import argparse
import hashlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple

def world_to_voxel(world_coords: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    #convert world coordinates (mm) to voxel coordinates (indices)
    voxel_coords = (world_coords - origin) / spacing
    return np.round(voxel_coords).astype(int)

def create_ellipsoid_mask(  #creates a binary mask with a filled ellipsoid accounting for anisotropic voxel spacing
    shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
    radius_mm: float,
    spacing: np.ndarray
) -> np.ndarray:
    #creates a 3D ellipsoid mask representing a lung nodule

    mask = np.zeros(shape, dtype=np.uint8)  #empty mask
    center_z, center_y, center_x = center  #unpack center coordinates
    spacing_x, spacing_y, spacing_z = spacing  #converts radius from mm to voxel dimensions (x,y,z -> z,y,x for array indexing)
    radius_voxels_z = radius_mm / spacing_z  #different numbers of voxels create ellipsoid in voxel space (sphere in physcial space)
    radius_voxels_y = radius_mm / spacing_y
    radius_voxels_x = radius_mm / spacing_x

    max_radius = int(np.ceil(max(radius_voxels_z, radius_voxels_y, radius_voxels_x)))  #bounding box (smallest box containing ellipsoid)
    z_min = max(0, center_z - max_radius)  #starting z-index of bounding box
    z_max = min(shape[0], center_z + max_radius + 1)  #ending z-index of bounding box
    y_min = max(0, center_y - max_radius)
    y_max = min(shape[1], center_y + max_radius + 1)
    x_min = max(0, center_x - max_radius)
    x_max = min(shape[2], center_x + max_radius + 1)

    if z_min >= z_max or y_min >= y_max or x_min >= x_max:  # FIXED: was x_min > x_max
        return mask

    #creates 3D coordinate arrays for each point in bounding box
    zz, yy, xx = np.meshgrid(
        np.arange(z_min, z_max),
        np.arange(y_min, y_max),
        np.arange(x_min, x_max),
        indexing='ij'
    )

    #calculates normalized distance from center(ellipsoid equation: (x/a)² + (y/b)² + (z/c)² ≤ 1)
    normalized_distance_sq = (
        ((zz - center_z) / radius_voxels_z)**2 +
        ((yy - center_y) / radius_voxels_y)**2 +
        ((xx - center_x) / radius_voxels_x)**2
    )
    ellipsoid_mask = normalized_distance_sq <= 1
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = ellipsoid_mask.astype(np.uint8)

    return mask

def create_masks_for_subset(  #process a single subset and create masks for all scans
        subset_dir: Path,
        annotations: pd.DataFrame,
        output_dir: Path,
        subset_name: str
) -> dict:

    # Create subset folder in output directory
    subset_output_dir = output_dir / subset_name
    os.makedirs(subset_output_dir, exist_ok=True)

    scan_files = list(subset_dir.glob('*.mhd'))  #finds all .mhd files
    if len(scan_files) == 0:
        print(f"No .mhd files found in {subset_dir}")
        return {
            'total_scans': 0,
            'scans_with_nodules': 0,
            'total_nodules_created': 0,
            'nodules_skipped': 0
        }

    total_scans = len(scan_files)
    scans_with_nodules = 0
    total_nodules_created = 0
    nodules_skipped = 0

    print(f"\nFound {total_scans} scans")

    for index, scan_path in enumerate(scan_files, 1):  #process each scan
        series_uid = scan_path.stem  #gets series UID from filename
        print(f"  [{index}/{total_scans}] Processing: {series_uid}")

        scan_sitk = sitk.ReadImage(str(scan_path))  #load CT scan
        scan_array = sitk.GetArrayFromImage(scan_sitk)

        origin = np.array(scan_sitk.GetOrigin())
        spacing = np.array(scan_sitk.GetSpacing())
        shape = scan_array.shape  #z,y,x

        mask = np.zeros(shape, dtype=np.uint8)  #empty mask
        scan_annotations = annotations[annotations['seriesuid'] == series_uid]  #filters annotations to only those belonging to current scan
        num_nodules = len(scan_annotations)

        if num_nodules == 0:
            print(f"No nodules found")
        else:
            scans_with_nodules += 1
            print(f"Found {num_nodules} nodule(s)")

            for _, row in scan_annotations.iterrows():  #process each nodule
                world_coordinates = np.array([row['coordX'], row['coordY'], row['coordZ']])
                diameter_mm = row['diameter_mm']
                radius_mm = diameter_mm / 2.0

                voxel_coords_xyz = world_to_voxel(world_coordinates, origin, spacing)  #converts world coords to voxel
                voxel_x, voxel_y, voxel_z = voxel_coords_xyz

                if (0 <= voxel_z < shape[0] and 0 <= voxel_y < shape[1] and 0 <= voxel_x < shape[2]):
                    #create ellipsoid mask for the nodule
                    center_zyx = (voxel_z, voxel_y, voxel_x)
                    try:
                        nodule_mask = create_ellipsoid_mask(
                            shape=shape,
                            center=center_zyx,
                            radius_mm=radius_mm,
                            spacing=spacing
                        )
                        mask = np.logical_or(mask, nodule_mask).astype(np.uint8)  #uses OR to handle overlapping nodules
                        total_nodules_created += 1
                        avg_radius_voxels = radius_mm / np.mean(spacing)
                        print(f"Nodule at voxel (z={voxel_z}, y={voxel_y}, x={voxel_x}), radius~{avg_radius_voxels:.2f} voxels")
                    except Exception as e:
                        print(f"Failed to create ellipsoid mask: {e}")
                        nodules_skipped += 1
                else:
                    print(f"Nodule center outside scan bounds")
                    nodules_skipped += 1

        # --- Save mask using original UID instead of hash ---
        output_path = subset_output_dir / f"{series_uid}_mask.npy"
        np.save(output_path, mask)

        num_positive_voxels = np.sum(mask)
        total_voxels = mask.size
        percentage = (num_positive_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        print(f"Mask saved: {output_path.name}")
        print(f"Positive voxels: {num_positive_voxels}/{total_voxels} ({percentage:.4f}%)")

    return {
        'total_scans': total_scans,
        'scans_with_nodules': scans_with_nodules,
        'total_nodules_created': total_nodules_created,
        'nodules_skipped': nodules_skipped
    }

def create_masks_for_all_subsets(
        base_dir: str,
        annotations_file: str,
        output_dir: str,
        num_subsets: int = 10
) -> None:

    base_dir_path = Path(base_dir)
    annotations_file_path = Path(annotations_file)
    output_dir_path = Path(output_dir)

    if not base_dir_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not annotations_file_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    os.makedirs(output_dir_path, exist_ok=True)  #output directory

    try:
        annotations = pd.read_csv(annotations_file)
    except Exception as e:
        raise ValueError(f"Failed to load annotations.csv: {e}")

    overall_stats = {
        'total_scans': 0,
        'scans_with_nodules': 0,
        'total_nodules_created': 0,
        'nodules_skipped': 0
    }

    subset_stats = []

    for subset_idx in range(num_subsets):
        subset_dir = base_dir_path / f"subset{subset_idx}"

        if not subset_dir.exists():
            print(f"Subset directory not found: {subset_dir}")
            continue

        stats = create_masks_for_subset(subset_dir, annotations, output_dir_path, f"subset{subset_idx}")

        overall_stats['total_scans'] += stats['total_scans']
        overall_stats['scans_with_nodules'] += stats['scans_with_nodules']
        overall_stats['total_nodules_created'] += stats['total_nodules_created']
        overall_stats['nodules_skipped'] += stats['nodules_skipped']

        subset_stats.append({
            'subset': subset_idx,
            **stats
        })

    print(f"\nTotal scans processed: {overall_stats['total_scans']}")
    print(f"Scans with nodules: {overall_stats['scans_with_nodules']}")
    print(f"Scans without nodules: {overall_stats['total_scans'] - overall_stats['scans_with_nodules']}")
    print(f"Total nodules created: {overall_stats['total_nodules_created']}")
    print(f"Nodules skipped (out of bounds): {overall_stats['nodules_skipped']}")

def main():
    parser = argparse.ArgumentParser(description="Create segmentation masks for LUNA16 CT scans")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to LUNA16 data")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to annotations.csv file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory for masks")
    parser.add_argument("--num_subsets", type=int, default=10,
                        help="Number of subsets to process (default: 10)")

    args = parser.parse_args()

    print("Creating segmentation masks for all LUNA16 subsets...")
    create_masks_for_all_subsets(args.base_dir, args.annotations, args.output_dir, num_subsets=args.num_subsets)
    print("\nDone!")

if __name__ == "__main__":
    main()