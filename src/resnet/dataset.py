import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class TrainDataset(Dataset):
    """Training dataset - loads directly from patch directory."""

    def __init__(self, train_dir, neg_pos_ratio=20, transform=None):
        self.train_dir = Path(train_dir)
        self.transform = transform

        print("Scanning training directory...")
        all_files = list(self.train_dir.glob("*.npz"))

        self.positive_files = []
        self.negative_files = []

        for f in tqdm(all_files, desc="Categorizing patches"):
            try:
                data = np.load(f)
                if int(data['label']) == 1:
                    self.positive_files.append(f)
                else:
                    self.negative_files.append(f)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        num_positives = len(self.positive_files)
        num_negatives_to_use = num_positives * neg_pos_ratio

        np.random.seed(42)
        if len(self.negative_files) > num_negatives_to_use:
            self.negative_files = list(np.random.choice(
                self.negative_files, num_negatives_to_use, replace=False
            ))

        self.files = self.positive_files + self.negative_files

        print(f"\nTraining dataset:")
        print(f"  Positives: {len(self.positive_files)}")
        print(f"  Negatives: {len(self.negative_files)}")
        print(f"  Total: {len(self.files)}")
        print(f"  Ratio: 1:{len(self.negative_files)/max(len(self.positive_files),1):.1f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        patch = torch.from_numpy(data['scan']).unsqueeze(0).float()
        label = torch.tensor(int(data['label']), dtype=torch.float32)

        if self.transform:
            patch = self.transform(patch)

        return patch, label


class ValidationDataset(Dataset):
    """Validation dataset - loads all patches from directory."""

    def __init__(self, val_dir):
        self.val_dir = Path(val_dir)
        self.files = list(self.val_dir.glob("*.npz"))

        self.num_pos = 0
        self.num_neg = 0
        for f in self.files:
            try:
                data = np.load(f)
                if int(data['label']) == 1:
                    self.num_pos += 1
                else:
                    self.num_neg += 1
            except:
                pass

        print(f"Validation dataset:")
        print(f"  Positives: {self.num_pos}")
        print(f"  Negatives: {self.num_neg}")
        print(f"  Total: {len(self.files)}")
        print(f"  Ratio: 1:{self.num_neg/max(self.num_pos,1):.1f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        patch = torch.from_numpy(data['scan']).unsqueeze(0).float()
        label = torch.tensor(int(data['label']), dtype=torch.float32)
        return patch, label
