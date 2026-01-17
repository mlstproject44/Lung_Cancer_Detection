import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

from model import ResNet3D_18, ResNet3D_34
from dataset import TrainDataset, ValidationDataset


class RandomFlip3D:
    """Randomly flip 3D volume along each axis."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, volume):
        if np.random.random() < self.prob:
            volume = torch.flip(volume, [1])
        if np.random.random() < self.prob:
            volume = torch.flip(volume, [2])
        if np.random.random() < self.prob:
            volume = torch.flip(volume, [3])
        return volume


class RandomRotate90_3D:
    """Randomly rotate 3D volume by 90 degrees in axial plane."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, volume):
        if np.random.random() < self.prob:
            k = np.random.randint(0, 4)
            if k > 0:
                volume = torch.rot90(volume, k, [2, 3])
        return volume


class RandomIntensityShift:
    """Slight intensity shift for robustness."""

    def __init__(self, shift_range=0.05, prob=0.3):
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, volume):
        if np.random.random() < self.prob:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            volume = volume + shift
            volume = torch.clamp(volume, 0, 1)
        return volume


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volume):
        for transform in self.transforms:
            volume = transform(volume)
        return volume


def get_train_transforms(flip_prob=0.5, rotate_prob=0.5, intensity_shift=0.05):
    """Create standard training augmentation pipeline."""
    return Compose([
        RandomFlip3D(prob=flip_prob),
        RandomRotate90_3D(prob=rotate_prob),
        RandomIntensityShift(shift_range=intensity_shift, prob=0.3)
    ])


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE
        return F_loss.mean()


def find_threshold_for_sensitivity(labels, probs, target_sens=0.90):
    """Find the threshold that achieves target sensitivity."""
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]
    sorted_probs = probs[sorted_indices]

    total_positives = sorted_labels.sum()
    if total_positives == 0:
        return 0.5

    target_tp = int(np.ceil(target_sens * total_positives))

    tp_count = 0
    for i, (prob, label) in enumerate(zip(sorted_probs, sorted_labels)):
        if label == 1:
            tp_count += 1
        if tp_count >= target_tp:
            return prob - 1e-6

    return 0.0


def validate_model(model, val_loader, criterion, device, target_sensitivity=0.90):
    """Validate with threshold calibration for target sensitivity."""
    model.eval()

    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for patches, labels in val_loader:
            patches = patches.to(device)
            labels_dev = labels.to(device)

            outputs = model(patches)
            loss = criterion(outputs, labels_dev)

            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()

            total_loss += loss.item()
            all_labels.extend(labels_np)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(val_loader)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    preds_05 = (all_probs > 0.5).astype(float)
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(all_labels, preds_05, labels=[0, 1]).ravel()
    sens_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0
    spec_05 = tn_05 / (tn_05 + fp_05) if (tn_05 + fp_05) > 0 else 0

    optimal_thresh = find_threshold_for_sensitivity(all_labels, all_probs, target_sensitivity)
    preds_opt = (all_probs > optimal_thresh).astype(float)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(all_labels, preds_opt, labels=[0, 1]).ravel()
    sens_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    spec_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0

    total_neg = tn_opt + fp_opt
    fp_reduction = tn_opt / total_neg if total_neg > 0 else 0

    return {
        'loss': avg_loss,
        'auc': auc,
        'sens_05': sens_05 * 100,
        'spec_05': spec_05 * 100,
        'optimal_threshold': optimal_thresh,
        'sens_opt': sens_opt * 100,
        'spec_opt': spec_opt * 100,
        'fp_reduction': fp_reduction * 100,
        'tp_opt': int(tp_opt),
        'fp_opt': int(fp_opt),
        'fn_opt': int(fn_opt),
        'tn_opt': int(tn_opt),
    }


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Augmentation
    train_transform = None
    if args.augmentation:
        train_transform = get_train_transforms(
            flip_prob=args.flip_prob,
            rotate_prob=args.rotate_prob
        )
        print("Data augmentation enabled")

    # Datasets
    train_dataset = TrainDataset(
        args.train_dir,
        neg_pos_ratio=args.neg_pos_ratio,
        transform=train_transform
    )
    val_dataset = ValidationDataset(args.val_dir)

    # Weighted sampling
    if args.weighted_sampling:
        num_pos = len(train_dataset.positive_files)
        num_neg = len(train_dataset.negative_files)

        pos_weight_sample = 1.0 / num_pos
        neg_weight_sample = 1.0 / num_neg

        sample_weights = []
        for f in train_dataset.files:
            if f in train_dataset.positive_files:
                sample_weights.append(pos_weight_sample)
            else:
                sample_weights.append(neg_weight_sample)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        print("Using WeightedRandomSampler")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model
    if args.model == 'resnet18':
        model = ResNet3D_18(dropout=args.dropout).to(device)
    else:
        model = ResNet3D_34(dropout=args.dropout).to(device)

    print(f"Model: {args.model} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Loss
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        print("Loss: FocalLoss")
    else:
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss (pos_weight={args.pos_weight})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr
        )

    # Training loop
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0
        all_train_probs = []
        all_train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for patches, labels_batch in pbar:
            patches = patches.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_train_probs.extend(probs)
            all_train_labels.extend(labels_batch.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = total_loss / len(train_loader)
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_probs)
        except:
            train_auc = 0.5

        val_metrics = validate_model(model, val_loader, criterion, device, args.target_sensitivity)

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  TRAIN - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"  VAL   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  VAL @{val_metrics['optimal_threshold']:.3f} - Sens: {val_metrics['sens_opt']:.1f}%, Spec: {val_metrics['spec_opt']:.1f}%")
        print(f"  FP Reduction: {val_metrics['fp_reduction']:.1f}%")

        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['auc'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  LR reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            epochs_no_improve = 0

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimal_threshold': val_metrics['optimal_threshold'],
                'auc': val_metrics['auc'],
                'sensitivity': val_metrics['sens_opt'],
                'specificity': val_metrics['spec_opt'],
                'epoch': epoch + 1
            }
            torch.save(checkpoint, Path(args.checkpoint_dir) / 'best_model.pth')
            print(f"  *** New best AUC: {val_metrics['auc']:.4f} ***")
        else:
            epochs_no_improve += 1

        if args.early_stopping and epochs_no_improve >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete! Best validation AUC: {best_val_auc:.4f}")

    best_ckpt = torch.load(Path(args.checkpoint_dir) / 'best_model.pth')
    print(f"\nBest model (epoch {best_ckpt['epoch']}):")
    print(f"  AUC: {best_ckpt['auc']:.4f}")
    print(f"  Optimal threshold: {best_ckpt['optimal_threshold']:.4f}")
    print(f"  Sensitivity: {best_ckpt['sensitivity']:.1f}%")
    print(f"  Specificity: {best_ckpt['specificity']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet3D for nodule classification")

    # Data
    parser.add_argument("--train_dir", type=str, required=True, help="Training patches directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation patches directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Output checkpoint directory")

    # Model
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--dropout", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Loss
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--pos_weight", type=float, default=1.0)

    # Class balancing
    parser.add_argument("--neg_pos_ratio", type=int, default=20)
    parser.add_argument("--weighted_sampling", action="store_true", default=True)
    parser.add_argument("--no_weighted_sampling", action="store_false", dest="weighted_sampling")

    # Augmentation
    parser.add_argument("--augmentation", action="store_true", default=True)
    parser.add_argument("--no_augmentation", action="store_false", dest="augmentation")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_prob", type=float, default=0.5)

    # Scheduler
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    parser.add_argument("--no_scheduler", action="store_false", dest="use_scheduler")
    parser.add_argument("--lr_patience", type=int, default=7)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-7)

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=15)

    # Threshold calibration
    parser.add_argument("--target_sensitivity", type=float, default=0.90)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
