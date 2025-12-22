import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def build_subset_mapping(scan_dirs: List[Tuple[int, str]]) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    #builds bidirectional mapping between subsets and UIDs
    subset_to_uids = defaultdict(list)
    uid_to_subset = {}
    for subset_number, scan_dir in scan_dirs:  #iterates through subsets
        scan_path = Path(scan_dir)
        for mhd_file in scan_path.glob('*.mhd'):
            uid = mhd_file.stem  #filename without extension
            subset_to_uids[subset_number].append(uid)  #adds UID to the list for given subset
            uid_to_subset[uid] = subset_number  #maps UID to its subset number

    return dict(subset_to_uids), uid_to_subset

def split_uids(
        uids: List[str], train_ratio: float=0.7, test_ratio: float=0.2, val_ratio: float=0.1, random_seed: int=42
) -> Tuple[List[str], List[str], List[str]]:
    #randomly splits unique IDs to train/test/val sets
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(random_seed)  #random number generator
    shuffled = rng.permutation(uids).tolist()  #random permutation of UIDs

    n = len(shuffled)
    n_train = int(n * train_ratio)  #calculates split points
    n_test = int(n * test_ratio)

    return (
        shuffled[:n_train],  #70%
        shuffled[n_train:n_train + n_test],  #20%
        shuffled[n_train + n_test:]  #10%
    )

def luna16_splits(
        scan_dirs: List[Tuple[int, str]],
        output_path: str,
        train_ratio: float=0.7,
        test_ratio: float=0.2,
        val_ratio: float=0.1,
        random_seed: int=42
) -> None:
    
    subset_to_uids, uid_to_subset = build_subset_mapping(scan_dirs)  #build mappings (eliminates redundant scanning later)
    splits = {'train': [], 'test': [], 'val': [], 'subset_info': {}}

    for subset_num, uids in sorted(subset_to_uids.items()):  #split each subset
        train_uids, test_uids, val_uids = split_uids(
            uids, train_ratio, test_ratio, val_ratio, random_seed
        )
        for split_name, uid_list in [('train', train_uids), ('test', test_uids), ('val', val_uids)]:
            splits[split_name].extend(uid_list)
        splits['subset_info'][f'subset{subset_num}'] = {
            'total': len(uids),
            'train': len(train_uids),
            'test': len(test_uids),
            'val': len(val_uids)
        }
    #adds metadata to document and preserve params and config used to create splits
    splits['metadata'] = {
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'val_ratio': val_ratio,
        'random_seed': random_seed,
        'total_scans': sum(info['total'] for info in splits['subset_info'].values()),
        'num_subsets': len(subset_to_uids)
    }
    splits['uid_to_subset'] = uid_to_subset  #saves uid to subset mapping
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)  #saves splits to json file
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\emirb\OneDrive\Desktop\coding\Python\DSAI\projects\data\data_preprocessing"
    SCAN_DIRS = [
        (i, os.path.join(BASE_DIR, "luna16_downsampled", f"subset{i}_downsampled"))
        for i in range(10)
    ]
    MASK_DIR = os.path.join(BASE_DIR, "luna16_masks")
    OUTPUT_PATH = os.path.join(BASE_DIR, "luna16_splits.json")

    luna16_splits(scan_dirs=SCAN_DIRS, output_path=OUTPUT_PATH, train_ratio=0.7, test_ratio=0.2,
        val_ratio=0.1, random_seed=42
    )