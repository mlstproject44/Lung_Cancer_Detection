import argparse
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from unet3d import UNet3D
from luna16_dataset_and_dataloader_v4 import create_patch_dataloaders


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        predict = predict.view(-1)
        target = target.view(-1)
        intersection = (predict * target).sum()
        dice = (2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
        return 1 - dice


def combined_loss(pred, target, pos_weight=125.0):
    """Combined BCE + Dice loss optimized for high recall."""
    pos_weight_tensor = torch.tensor([pos_weight]).to(pred.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(pred, target)
    dice = DiceLoss()(pred, target)
    return 0.7 * bce + 0.3 * dice


def calculate_batch_metrics(predictions, targets, threshold=0.5):
    """Calculate recall, precision, dice, F1 for a batch."""
    preds_prob = torch.sigmoid(predictions)
    preds_binary = (preds_prob > threshold).float()

    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    TP = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
    FP = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (targets_flat == 0)).sum().float()

    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'TP': TP.item(), 'FP': FP.item(), 'FN': FN.item(), 'TN': TN.item(),
        'recall': recall.item(), 'precision': precision.item(),
        'dice': dice.item(), 'f1': f1.item()
    }


def train(args):
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir, 'all_epochs').mkdir(parents=True, exist_ok=True)

    print("TRAINING CONFIGURATION")
    print(f"Device:         {device}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"pos_weight:     {args.pos_weight}")
    print(f"Patch Dir:      {args.patch_dir}")

    # Data loaders
    train_loader, val_loader, test_loader = create_patch_dataloaders(
        patch_dir=args.patch_dir,
        batch_size=args.batch_size,
        num_workers=0,
        positive_fraction=args.positive_fraction,
    )

    # Model
    model = UNet3D(
        input_channels=1,
        output_channels=1,
        init_features=args.init_features,
        dropout=args.dropout,
        checkpointing=False
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.lr_patience,
        factor=args.lr_factor
    )

    best_val_recall = 0.0
    best_balanced_dice = 0.0
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_recalls, val_recalls = [], []
    train_precisions, val_precisions = [], []
    train_dices, val_dices = [], []
    learning_rates = []

    print("\nSTARTING TRAINING")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_TP, train_FP, train_FN, train_TN = 0, 0, 0, 0

        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train] LR={current_lr:.4f}")

        for batch in pbar:
            scans = batch['scan'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(scans)
            loss = combined_loss(outputs, masks, args.pos_weight)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                metrics = calculate_batch_metrics(outputs, masks)
                train_TP += metrics['TP']
                train_FP += metrics['FP']
                train_FN += metrics['FN']
                train_TN += metrics['TN']

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_recall = train_TP / (train_TP + train_FN + 1e-8)
        train_precision = train_TP / (train_TP + train_FP + 1e-8)
        train_dice = (2 * train_TP) / (2 * train_TP + train_FP + train_FN + 1e-8)

        # Validation
        model.eval()
        val_loss = 0
        val_TP, val_FP, val_FN, val_TN = 0, 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                scans = batch['scan'].to(device)
                masks = batch['mask'].to(device)
                outputs = model(scans)

                val_loss += combined_loss(outputs, masks, args.pos_weight).item()

                metrics = calculate_batch_metrics(outputs, masks)
                val_TP += metrics['TP']
                val_FP += metrics['FP']
                val_FN += metrics['FN']
                val_TN += metrics['TN']

        avg_val_loss = val_loss / len(val_loader)
        val_recall = val_TP / (val_TP + val_FN + 1e-8)
        val_precision = val_TP / (val_TP + val_FP + 1e-8)
        val_dice = (2 * val_TP) / (2 * val_TP + val_FP + val_FN + 1e-8)

        # Record history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        learning_rates.append(current_lr)

        old_lr = current_lr
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != old_lr:
            print(f"\nLR reduced: {old_lr:.6f} -> {current_lr:.6f}\n")

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  TRAIN - Loss: {avg_train_loss:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}, Dice: {train_dice:.4f}")
        print(f"  VAL   - Loss: {avg_val_loss:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, Dice: {val_dice:.4f}")

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU Memory: {allocated_gb:.2f}GB allocated")
            torch.cuda.reset_peak_memory_stats()

        # Checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'val_losses': val_losses,
            'train_recalls': train_recalls, 'val_recalls': val_recalls,
            'train_precisions': train_precisions, 'val_precisions': val_precisions,
            'train_dices': train_dices, 'val_dices': val_dices,
            'learning_rates': learning_rates,
            'best_val_recall': best_val_recall,
            'best_balanced_dice': best_balanced_dice,
        }

        # Save every epoch
        torch.save(checkpoint, Path(args.checkpoint_dir) / 'all_epochs' / f'epoch_{epoch+1:02d}.pth')

        # Save best recall
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(checkpoint, Path(args.checkpoint_dir) / 'best_recall.pth')
            print(f"  *** New best recall: {val_recall:.4f} ***")

        # Save balanced model (85-90% recall + best Dice)
        if 0.85 <= val_recall <= 0.90 and val_dice > best_balanced_dice:
            best_balanced_dice = val_dice
            torch.save(checkpoint, Path(args.checkpoint_dir) / 'best_balanced.pth')
            print(f"  *** Target hit! Recall={val_recall:.4f}, Dice={val_dice:.4f} ***")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save last checkpoint
        torch.save(checkpoint, Path(args.checkpoint_dir) / 'last_checkpoint.pth')

        # Early stopping
        if args.early_stopping and epochs_no_improve >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\nTRAINING COMPLETE")
    print(f"Best Recall: {best_val_recall:.4f}")
    print(f"Best Balanced Dice: {best_balanced_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for lung nodule segmentation")

    # Data
    parser.add_argument("--patch_dir", type=str, required=True, help="Directory with training patches")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Output checkpoint directory")

    # Model
    parser.add_argument("--init_features", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--positive_fraction", type=float, default=0.7)

    # Loss
    parser.add_argument("--pos_weight", type=float, default=125.0)

    # Scheduler
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=15)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
