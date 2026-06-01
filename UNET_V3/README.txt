UNet3D — best checkpoint
========================
Epoch         : 14 / 44 total trained
Early stop    : patience 30, triggered at epoch 44

── val metrics at epoch 14 (threshold 0.50) ──────────────────
Recall        : 0.9204
Precision     : 0.7415
Dice coeff    : 0.8213
F2 (β=2)      : 0.8780
MCC           : 0.8259
Specificity   : 0.9996
NPV           : 0.9999
Soft recall   : 0.5430

── val losses at epoch 14 ────────────────────────────────────
Combined      : 0.3467
BCE (×0.20)   : 0.0075
Focal (×0.30) : 0.0080
Dice (×0.50)  : 0.5484
SoftRec (×0.15): 0.4570

── train metrics at epoch 14 ─────────────────────────────────
Recall        : 0.8816
Precision     : 0.7874
Dice coeff    : 0.8318
F2 (β=2)      : 0.8610
MCC           : 0.8329
Specificity   : 0.9997
NPV           : 0.9998
Soft recall   : 0.8674

── train losses at epoch 14 ──────────────────────────────────
Combined      : 0.1291
BCE (×0.20)   : 0.0112
Focal (×0.30) : 0.0127
Dice (×0.50)  : 0.2063
SoftRec (×0.15): 0.1326

── test set results (threshold 0.20) ─────────────────────────
Recall        : 0.8952
Precision     : 0.6740
Dice coeff    : 0.7690
F2 (β=2)      : 0.8401
MCC           : 0.7766
Specificity   : 0.9996
NPV           : 0.9999
TP            : 9406077
FP            : 4549166
FN            : 1100945
TN            : 12512805572

── model config ──────────────────────────────────────────────
  version: v11
  batch_size: 4
  accum_steps: 4
  positive_fraction: 0.75
  optimizer: AdamW
  initial_lr: 0.0001
  importance_lr: 0.01
  weight_decay: 1e-05
  betas: (0.9, 0.999)
  scheduler: ReduceLROnPlateau
  pos_weight: 10.0
  focal_alpha: 0.85
  loss_weights: 0.2*BCE + 0.3*Focal + 0.5*Dice + 0.15*SoftRec
  dropout: 0.2
  covariance_dropout: 0.2
  patch_size: (96, 96, 96)
  amp: True
  device: cuda

── model ─────────────────────────────────────────────────────
Architecture  : unet3d_3.py
Params        : 51.78M
Threshold     : 0.20 (chosen via val sweep)
Checkpoint    : epoch_14_recall_0.9204_threshold_0.20.pth
