import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def build_subset_mapping(scan_dirs: List[Tuple[int, str]]) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    subset_to_uids = defaultdict(list)
    uid_to_subset = {}
    for subset_number, scan_dir in scan_dirs:
        scan_path = Path(scan_dir)
        for mhd_file in scan_path.glob('*.mhd'):
            uid = mhd_file.stem
            subset_to_uids[subset_number].append(uid)
            uid_to_subset[uid] = subset_number

    return dict(subset_to_uids), uid_to_subset

def load_nodule_volumes(annotations_csv: str, all_uids: List[str]) -> Tuple[Dict[str, Dict], Dict]:
    df = pd.read_csv(annotations_csv)

    #normalize column names
    df.columns = df.columns.str.strip()
    uid_col = next((c for c in df.columns if 'uid' in c.lower()), None)
    diam_col = next((c for c in df.columns if 'diameter' in c.lower()), None)

    df = df.dropna(subset=[uid_col, diam_col])

    scan_stats = {}

    for series_uid, group in df.groupby(uid_col):
        diameters = group[diam_col].values
        if len(diameters) == 0:
            continue
        volumes_mm3 = (np.pi / 6.0) * (diameters ** 3)  #nodule volume as sphere: V = (π/6) * d³
        total_volume = float(np.sum(volumes_mm3))

        scan_stats[series_uid] = {
            'series_uid': series_uid,
            'total_nodule_voxels': total_volume,  #mm³ volume as voxel proxy
            'num_nodules': len(diameters),
            'mean_diameter': float(np.mean(diameters)),
            'max_diameter': float(np.max(diameters)),
        }

    #scans not in annotations have no annotated nodules
    for uid in all_uids:
        if uid not in scan_stats:
            scan_stats[uid] = {
                'series_uid': uid,
                'total_nodule_voxels': 0.0,
                'num_nodules': 0,
                'mean_diameter': 0.0,
                'max_diameter': 0.0,
            }

    all_volumes = [s['total_nodule_voxels'] for s in scan_stats.values() if s['total_nodule_voxels'] > 0]

    if all_volumes:
        percentiles = {
            'p25': float(np.percentile(all_volumes, 25)),
            'p50': float(np.percentile(all_volumes, 50)),
            'p75': float(np.percentile(all_volumes, 75)),
            'p90': float(np.percentile(all_volumes, 90)),
        }
    else:
        percentiles = {'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0}

    #assign strata
    strata_counts = defaultdict(int)
    for stats in scan_stats.values():
        stratum = assign_voxel_stratum(stats['total_nodule_voxels'], percentiles)
        stats['stratum'] = stratum
        strata_counts[stratum] += 1

    print(f"\nLoaded stats for {len(scan_stats)} scans")
    print("\nStratum distribution:")
    for stratum in ['no_nodules', 'tiny', 'small', 'medium', 'large']:
        count = strata_counts.get(stratum, 0)
        print(f"  {stratum:<15} {count:>4} scans ({count / len(scan_stats) * 100:.1f}%)")

    return scan_stats, percentiles

def assign_voxel_stratum(total_voxels: float, percentiles: Dict) -> str:
    if total_voxels == 0:
        return 'no_nodules'
    elif total_voxels < percentiles['p25']:
        return 'tiny'
    elif total_voxels < percentiles['p50']:
        return 'small'
    elif total_voxels < percentiles['p75']:
        return 'medium'
    else:
        return 'large'

def split_uids_stratified(
        uids: List[str],
        scan_stats: Dict[str, Dict],
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str], Dict]:

    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(random_seed)

    strata_uids = defaultdict(list)
    for uid in uids:
        stratum = scan_stats[uid]['stratum']
        strata_uids[stratum].append(uid)

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

        volumes = [scan_stats[uid]['total_nodule_voxels'] for uid in stratum_uids_list]
        stratum_info[stratum] = {
            'total': n,
            'train': len(train_uids),
            'test': len(test_uids),
            'val': len(val_uids),
            'mean_volume_mm3': float(np.mean(volumes)),
            'median_volume_mm3': float(np.median(volumes)),
        }

    return train_all, test_all, val_all, stratum_info

def luna16_splits(
        scan_dirs: List[Tuple[int, str]],
        output_path: str,
        annotations_csv: str,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
) -> None:

    subset_to_uids, uid_to_subset = build_subset_mapping(scan_dirs)

    all_uids = []
    for uids in subset_to_uids.values():
        all_uids.extend(uids)

    scan_stats, percentiles = load_nodule_volumes(annotations_csv, all_uids)
    train_uids, test_uids, val_uids, stratum_info = split_uids_stratified(
        all_uids, scan_stats, train_ratio, test_ratio, val_ratio, random_seed
    )

    #overlap check
    assert not (set(train_uids) & set(val_uids)), "Train/val overlap detected"
    assert not (set(val_uids) & set(test_uids)), "Val/test overlap detected"
    assert not (set(train_uids) & set(test_uids)), "Train/test overlap detected"

    train_vols = [scan_stats[uid]['total_nodule_voxels'] for uid in train_uids]
    val_vols = [scan_stats[uid]['total_nodule_voxels'] for uid in val_uids]
    test_vols = [scan_stats[uid]['total_nodule_voxels'] for uid in test_uids]

    splits = {
        'train': train_uids,
        'test': test_uids,
        'val': val_uids,
        'metadata': {
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'random_seed': random_seed,
            'total_scans': len(all_uids),
            'stratification': 'voxel_volume',
            'strata': ['no_nodules', 'tiny', 'small', 'medium', 'large'],
            'percentiles_mm3': percentiles,
        },
        'stratum_info': stratum_info,
        'overall_volume_stats': {
            'train_median_mm3': float(np.median(train_vols)),
            'val_median_mm3': float(np.median(val_vols)),
            'test_median_mm3': float(np.median(test_vols)),
        },
        'uid_to_subset': uid_to_subset,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

def main():
    BASE_DIR    = ".../luna16_OG"
    ANNOTATIONS = ".../annotations.csv"
    OUTPUT      = ".../luna16_splits.json"
    NUM_SUBSETS = 10

    scan_dirs = [(i, os.path.join(BASE_DIR, f"subset{i}")) for i in range(NUM_SUBSETS)]

    luna16_splits(
        scan_dirs=scan_dirs,
        output_path=OUTPUT,
        annotations_csv=ANNOTATIONS,
    )


if __name__ == "__main__":
    main()
