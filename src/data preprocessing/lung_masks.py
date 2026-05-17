import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import SimpleITK as sitk

MIN_LUNG_AREA = 1500
MIN_SPLIT_FRACTION = 0.10
AIR_THRESHOLD = -400
BASE_DIR = Path(".../luna16_OG")
OUTPUT_DIR = Path(".../segmented_lungs")


def binarize(scan_array: np.ndarray, threshold: int = AIR_THRESHOLD) -> np.ndarray:
    return (scan_array < threshold).astype(np.uint8)

def _split_merged_lungs(region_mask: np.ndarray) -> np.ndarray:
    total = np.sum(region_mask)
    min_piece = total * MIN_SPLIT_FRACTION

    for iters in range(3, 30):
        eroded = ndimage.binary_erosion(region_mask, iterations=iters)  #erosion removes pixels from edges
        labels_e, n_e = ndimage.label(eroded)
        if n_e < 2:
            continue

        sizes_e = np.array(ndimage.sum(eroded, labels_e, range(1, n_e+1)))
        top_2 = np.argsort(sizes_e)[-2:] + 1

        seeds = np.zeros_like(labels_e)
        seeds[labels_e == top_2[0]] = 1  #point A
        seeds[labels_e == top_2[1]] = 2  #point B

        #distance_transform_edt calculates distance transform of the input
        distance_1 = distance_transform_edt(seeds != 1)
        distance_2 = distance_transform_edt(seeds != 2)
        assigned = np.where(distance_1 < distance_2, 1, 2)  #Voronoi partition
        result = np.where(region_mask, assigned, 0)

        s1 = np.sum(result == 1)
        s2 = np.sum(result == 2)
        if s1 < min_piece or s2 < min_piece:
            continue
        return result

    return region_mask.astype(np.int32)  #no valid split

def extract_lung_mask(binary_mask: np.ndarray) -> np.ndarray:

    #---- flood-fill to detect interior air ----
    interior_mask = np.zeros_like(binary_mask)
    for j in range(binary_mask.shape[0]):
        air_slice = binary_mask[j].astype(bool)
        padded = np.pad(air_slice, 1, mode="constant", constant_values=True)  #border of True vals
        labeled_pad, _ = ndimage.label(padded)
        exterior_label = labeled_pad[0,0]  # always true
        exterior = labeled_pad[1:-1, 1:-1] == exterior_label
        interior_mask[j] = (air_slice & ~exterior).astype(np.uint8)  # interior air and not exterior are in final mask
    
    print(f"Interior mask voxel count: {interior_mask.sum()}")

    #---- first pass, size filtering ----
    per_slice_labels = {}
    two_lung_sizes = []

    for k in range(binary_mask.shape[0]):
        slice = interior_mask[k]
        label, n = ndimage.label(slice)
        if n == 0:
            continue
        sizes = np.array(ndimage.sum(slice, label, range(1, n+1)))
        keep = sizes > MIN_LUNG_AREA
        kept_labels = np.where(keep)[0]+1
        if keep.sum() == 0:
            continue

        filtered = np.zeros_like(label)
        kept_sizes = []
        for new_label, old_label in enumerate(kept_labels, start=1):
            filtered[label == old_label] = new_label
            kept_sizes.append(sizes[old_label - 1])
        
        per_slice_labels[k] = (filtered, len(kept_labels), np.array(kept_sizes))

        if len(kept_sizes) >= 2:
            sorted_s = sorted(kept_sizes, reverse=True)
            two_lung_sizes.append(sorted_s[0] + sorted_s[1])
    
    #---- merge threshold ----
    if two_lung_sizes:
        median = np.median(two_lung_sizes)
        threshold = median * 0.8
    else:
        threshold = 40000
    
    #---- second pass, split merges ----
    result = np.zeros_like(binary_mask)
    for j in range(binary_mask.shape[0]):
        if j not in per_slice_labels:
            continue
        filtered, n_regions, kept_sizes = per_slice_labels[j]

        if n_regions == 1 and kept_sizes[0] > threshold:
            region_mask = (filtered == 1).astype(np.uint8)
            split = _split_merged_lungs(region_mask)
            if split.max() >= 2:
                s1, s2 = np.sum(split==1), np.sum(split==2)
                filtered = split
                n_regions = 2
                kept_sizes = np.array([s1, s2])
        
        if n_regions >= 2:
            top2_labels = np.argsort(kept_sizes)[-2:] + 1
            result[j] = np.isin(filtered, top2_labels).astype(np.uint8)
        
        if n_regions == 1:
            result[j] = (filtered > 0).astype(np.uint8)
    
    #---- remove small 3d fragments (noise) ----
    structure_3d = ndimage.generate_binary_structure(3, 1)
    labeled_3d, n_3d = ndimage.label(result, structure_3d)
    if n_3d > 2:
        sizes_3d = np.array(ndimage.sum(result, labeled_3d, range(1, n_3d + 1)))
        top2 = np.argsort(sizes_3d)[-2:] + 1
        result = np.isin(labeled_3d, top2).astype(np.uint8)
    else:
        pass

    return result

def complete_mask(lung_mask: np.ndarray) -> np.ndarray:
    filled_mask = np.zeros_like(lung_mask)
    for j in range(lung_mask.shape[0]):
        filled_mask[j] = ndimage.binary_fill_holes(lung_mask[j]).astype(np.uint8)

    structure = ndimage.generate_binary_structure(3, 1)
    filled_mask = ndimage.binary_closing(filled_mask, structure, iterations=3).astype(np.uint8)
    filled_mask = ndimage.binary_dilation(filled_mask, structure, iterations=1).astype(np.uint8)

    return filled_mask

def apply_mask(lung_mask: np.ndarray, scan: np.ndarray, fill_value: int = -1000) -> np.ndarray:
    mask = np.full_like(scan, fill_value)
    mask[lung_mask == 1] = scan[lung_mask == 1]
    return mask

def process_scan(scan_path: Path, output_dir: Path) -> None:
    scan_uid = scan_path.stem
    scan = sitk.ReadImage(str(scan_path))
    scan_array = sitk.GetArrayFromImage(scan)

    binary = binarize(scan_array)
    lung_mask = extract_lung_mask(binary)
    cmplt_mask = complete_mask(lung_mask)
    parenchyma = apply_mask(cmplt_mask, scan_array)
    lung_pct = cmplt_mask.sum() / cmplt_mask.size

    print(f"Scan ID: {scan_uid}, lung voxels: {int(cmplt_mask.sum())}")
    print(f"Lung percentage: {lung_pct:.4f}")

    np.save(output_dir / f"{scan_uid}_parenchyma.npy", parenchyma)

def process_all_subsets(base_dir: Path, output_dir: Path, num_subsets: int = 10) -> None:
    output_dir.mkdir(exist_ok=True)

    for j in range(num_subsets):
        subset_dir = base_dir / f"subset{j}"
        if not subset_dir.exists():
            continue

        scans = sorted(subset_dir.glob("*.mhd"))
        subset_output = output_dir / f"subset{j}"
        subset_output.mkdir(exist_ok=True)

        for scan in scans:
            process_scan(scan, subset_output)

process_all_subsets(BASE_DIR, OUTPUT_DIR)