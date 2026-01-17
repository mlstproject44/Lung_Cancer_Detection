import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

def build_subset_mapping(scan_dirs: List[Tuple[int, str]]) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """builds bidirectional mapping between subsets and UIDs"""
    subset_to_uids = defaultdict(list)
    uid_to_subset = {}
    for subset_number, scan_dir in scan_dirs:  #iterates through subsets
        scan_path = Path(scan_dir)
        for mhd_file in scan_path.glob('*.mhd'):
            uid = mhd_file.stem  #filename without extension
            subset_to_uids[subset_number].append(uid)  #adds UID to the list for given subset
            uid_to_subset[uid] = subset_number  #maps UID to its subset number

    return dict(subset_to_uids), uid_to_subset

def load_voxel_characteristics(patch_dir: str, all_uids: List[str]) -> Tuple[Dict[str, Dict], Dict]:
    """loads actual voxel counts from extracted patches for stratification"""

    print("Analyzing patch voxel distribution for stratification...")
    patch_dir_path = Path(patch_dir)

    scan_stats = {}

    #check all possible split directories for patches
    for split_name in ['train', 'val', 'test', '']:  # '' for flat structure
        if split_name:
            split_dir = patch_dir_path / split_name
        else:
            split_dir = patch_dir_path

        metadata_file = split_dir / "metadata.json"

        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"  Found {len(metadata)} patches in {split_dir.name or 'root'}/")

        #group patches by series_uid
        uid_patches = defaultdict(list)
        for record in metadata:
            uid_patches[record['series_uid']].append(record['filename'])

        #analyze each scan's patches
        for series_uid, patch_filenames in tqdm(uid_patches.items(), desc=f"  Analyzing {split_name or 'patches'}"):
            if series_uid not in scan_stats:
                scan_stats[series_uid] = {
                    'series_uid': series_uid,
                    'total_nodule_voxels': 0,
                    'num_patches': 0,
                    'num_positive_patches': 0
                }

            for filename in patch_filenames:
                patch_path = split_dir / filename

                if not patch_path.exists():
                    continue

                try:
                    data = np.load(patch_path)
                    mask = data['mask']

                    #count positive voxels
                    if mask.dtype == np.uint8:
                        positive_voxels = int((mask > 0).sum())
                    else:
                        positive_voxels = int((mask > 0.5).sum())

                    scan_stats[series_uid]['num_patches'] += 1

                    if positive_voxels > 0:
                        scan_stats[series_uid]['total_nodule_voxels'] += positive_voxels
                        scan_stats[series_uid]['num_positive_patches'] += 1

                    data.close()
                except Exception as e:
                    continue

    #add scans without patches (no nodules)
    for uid in all_uids:
        if uid not in scan_stats:
            scan_stats[uid] = {
                'series_uid': uid,
                'total_nodule_voxels': 0,
                'num_patches': 0,
                'num_positive_patches': 0
            }

    #compute percentiles for stratification
    all_voxel_counts = [s['total_nodule_voxels'] for s in scan_stats.values() if s['total_nodule_voxels'] > 0]

    if len(all_voxel_counts) > 0:
        percentiles = {
            'p25': float(np.percentile(all_voxel_counts, 25)),
            'p50': float(np.percentile(all_voxel_counts, 50)),
            'p75': float(np.percentile(all_voxel_counts, 75)),
            'p90': float(np.percentile(all_voxel_counts, 90))
        }
    else:
        percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0}

    print(f"\nVoxel count percentiles:")
    print(f"  25th: {percentiles['p25']:,.0f} voxels")
    print(f"  50th: {percentiles['p50']:,.0f} voxels")
    print(f"  75th: {percentiles['p75']:,.0f} voxels")

    #assign voxel-based strata
    strata_counts = defaultdict(int)
    for series_uid, stats in scan_stats.items():
        stratum = assign_voxel_stratum(stats['total_nodule_voxels'], percentiles)
        stats['stratum'] = stratum
        strata_counts[stratum] += 1

    print(f"\nLoaded stats for {len(scan_stats)} scans")
    print("\nStratum distribution:")
    for stratum in ['no_nodules', 'tiny', 'small', 'medium', 'large']:
        count = strata_counts.get(stratum, 0)
        print(f"  {stratum:<15} {count:>4} scans ({count/len(scan_stats)*100:.1f}%)")

    return scan_stats, percentiles

def assign_voxel_stratum(total_voxels: int, percentiles: Dict) -> str:
    """assigns stratum based on total nodule voxels using percentiles."""
    if total_voxels == 0:
        return 'no_nodules'
    elif total_voxels < percentiles['p25']:
        return 'tiny'       # < 25th percentile
    elif total_voxels < percentiles['p50']:
        return 'small'      # 25th-50th percentile
    elif total_voxels < percentiles['p75']:
        return 'medium'     # 50th-75th percentile
    else:
        return 'large'      # > 75th percentile

def split_uids_stratified(
        uids: List[str],
        scan_stats: Dict[str, Dict],
        train_ratio: float=0.7,
        test_ratio: float=0.2,
        val_ratio: float=0.1,
        random_seed: int=42
) -> Tuple[List[str], List[str], List[str], Dict]:
    """split UIDs with stratification by nodule size"""

    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(random_seed)

    #group UIDs by stratum
    strata_uids = defaultdict(list)
    for uid in uids:
        stratum = scan_stats[uid]['stratum']
        strata_uids[stratum].append(uid)

    #split each stratum separately
    train_all, test_all, val_all = [], [], []
    stratum_info = {}

    for stratum, stratum_uids_list in strata_uids.items():
        shuffled = rng.permutation(stratum_uids_list).tolist()
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)

        train_uids = shuffled[:n_train]
        test_uids = shuffled[n_train:n_train + n_test]
        val_uids = shuffled[n_train + n_test:]

        train_all.extend(train_uids)
        test_all.extend(test_uids)
        val_all.extend(val_uids)

        #calculate voxel statistics for this stratum
        voxel_counts = [scan_stats[uid].get('total_nodule_voxels', 0) for uid in stratum_uids_list]

        stratum_info[stratum] = {
            'total': len(stratum_uids_list),
            'train': len(train_uids),
            'test': len(test_uids),
            'val': len(val_uids),
            'mean_voxels': float(np.mean(voxel_counts)),
            'median_voxels': float(np.median(voxel_counts))
        }

    return train_all, test_all, val_all, stratum_info

def luna16_splits(
        scan_dirs: List[Tuple[int, str]],
        output_path: str,
        patch_dir: str,
        train_ratio: float=0.7,
        test_ratio: float=0.2,
        val_ratio: float=0.1,
        random_seed: int=42
) -> None:
    """creates train/val/test splits with voxel-based stratification"""

    subset_to_uids, uid_to_subset = build_subset_mapping(scan_dirs)

    all_uids = []
    for uids in subset_to_uids.values():
        all_uids.extend(uids)

    print("Creating voxel stratified splits")

    scan_stats, percentiles = load_voxel_characteristics(patch_dir, all_uids)
    train_uids, test_uids, val_uids, stratum_info = split_uids_stratified(
        all_uids, scan_stats, train_ratio, test_ratio, val_ratio, random_seed
    )

    splits = {
        'train': train_uids,
        'test': test_uids,
        'val': val_uids,
        'stratum_info': stratum_info
    }

    splits['metadata'] = {
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'val_ratio': val_ratio,
        'random_seed': random_seed,
        'total_scans': len(all_uids),
        'stratification': 'voxel_count',
        'strata': ['no_nodules', 'tiny', 'small', 'medium', 'large'],
        'percentiles': percentiles
    }

    #overall voxel statistics
    train_voxels = [scan_stats[uid]['total_nodule_voxels'] for uid in train_uids]
    val_voxels = [scan_stats[uid]['total_nodule_voxels'] for uid in val_uids]
    test_voxels = [scan_stats[uid]['total_nodule_voxels'] for uid in test_uids]

    splits['overall_voxel_stats'] = {
        'train_median': float(np.median(train_voxels)),
        'val_median': float(np.median(val_voxels)),
        'test_median': float(np.median(test_voxels))
    }

    print("Stratification results")
    for stratum, info in sorted(stratum_info.items()):
        print(f"\n{stratum.upper()}:")
        print(f"  Total:  {info['total']:>3} ({info['total']/len(all_uids)*100:.1f}%)")
        print(f"  Train:  {info['train']:>3} ({info['train']/max(1, info['total'])*100:.0f}%)")
        print(f"  Val:    {info['val']:>3} ({info['val']/max(1, info['total'])*100:.0f}%)")
        print(f"  Test:   {info['test']:>3} ({info['test']/max(1, info['total'])*100:.0f}%)")
        if info['median_voxels'] > 0:
            print(f"Median voxels: {info['median_voxels']:,.0f}")

    print(f"Train: {len(train_uids)} scans ({len(train_uids)/len(all_uids)*100:.1f}%)")
    print(f"Val:   {len(val_uids)} scans ({len(val_uids)/len(all_uids)*100:.1f}%)")
    print(f"Test:  {len(test_uids)} scans ({len(test_uids)/len(all_uids)*100:.1f}%)")
    print("="*80)

    splits['uid_to_subset'] = uid_to_subset
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n✓ Splits saved to: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for LUNA16 dataset with voxel-based stratification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to LUNA16 data")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for splits JSON file")
    parser.add_argument("--patch_dir", type=str, required=True,
                        help="Path to extracted patches for voxel-based stratification")
    parser.add_argument("--num_subsets", type=int, default=10,
                        help="Number of subsets to process (default: 10)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    scan_dirs = [
        (i, os.path.join(args.base_dir, f"subset{i}"))
        for i in range(args.num_subsets)
    ]

    luna16_splits(
        scan_dirs=scan_dirs,
        output_path=args.output,
        patch_dir=args.patch_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()