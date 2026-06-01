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

── full val recall history ───────────────────────────────────
  Epoch   1: 0.6387
  Epoch   2: 0.7893
  Epoch   3: 0.8277
  Epoch   4: 0.8218
  Epoch   5: 0.8285
  Epoch   6: 0.7930
  Epoch   7: 0.8584
  Epoch   8: 0.8789
  Epoch   9: 0.9031
  Epoch  10: 0.8651
  Epoch  11: 0.8503
  Epoch  12: 0.8776
  Epoch  13: 0.9001
  Epoch  14: 0.9204
  Epoch  15: 0.8800
  Epoch  16: 0.9059
  Epoch  17: 0.8788
  Epoch  18: 0.8967
  Epoch  19: 0.8541
  Epoch  20: 0.8848
  Epoch  21: 0.8894
  Epoch  22: 0.9005
  Epoch  23: 0.8957
  Epoch  24: 0.8834
  Epoch  25: 0.8892
  Epoch  26: 0.8953
  Epoch  27: 0.8996
  Epoch  28: 0.8798
  Epoch  29: 0.8845
  Epoch  30: 0.8710
  Epoch  31: 0.8784
  Epoch  32: 0.8742
  Epoch  33: 0.8552
  Epoch  34: 0.8650
  Epoch  35: 0.8701
  Epoch  36: 0.8583
  Epoch  37: 0.8776
  Epoch  38: 0.8790
  Epoch  39: 0.8809
  Epoch  40: 0.8853
  Epoch  41: 0.8666
  Epoch  42: 0.9021
  Epoch  43: 0.8743
  Epoch  44: 0.8520

── full val loss history ─────────────────────────────────────
  Epoch   1: 0.4127
  Epoch   2: 0.3739
  Epoch   3: 0.3660
  Epoch   4: 0.3662
  Epoch   5: 0.3674
  Epoch   6: 0.3639
  Epoch   7: 0.3611
  Epoch   8: 0.3565
  Epoch   9: 0.3535
  Epoch  10: 0.3476
  Epoch  11: 0.3487
  Epoch  12: 0.3491
  Epoch  13: 0.3507
  Epoch  14: 0.3467
  Epoch  15: 0.3473
  Epoch  16: 0.3478
  Epoch  17: 0.3467
  Epoch  18: 0.3415
  Epoch  19: 0.3454
  Epoch  20: 0.3452
  Epoch  21: 0.3467
  Epoch  22: 0.3496
  Epoch  23: 0.3417
  Epoch  24: 0.3432
  Epoch  25: 0.3443
  Epoch  26: 0.3442
  Epoch  27: 0.3450
  Epoch  28: 0.3438
  Epoch  29: 0.3449
  Epoch  30: 0.3471
  Epoch  31: 0.3441
  Epoch  32: 0.3467
  Epoch  33: 0.3434
  Epoch  34: 0.3487
  Epoch  35: 0.3461
  Epoch  36: 0.3472
  Epoch  37: 0.3419
  Epoch  38: 0.3482
  Epoch  39: 0.3454
  Epoch  40: 0.3419
  Epoch  41: 0.3437
  Epoch  42: 0.3403
  Epoch  43: 0.3423
  Epoch  44: 0.3444

── learning rate history ─────────────────────────────────────
  Epoch   1: 1.00e-04
  Epoch   2: 1.00e-04
  Epoch   3: 1.00e-04
  Epoch   4: 1.00e-04
  Epoch   5: 1.00e-04
  Epoch   6: 1.00e-04
  Epoch   7: 1.00e-04
  Epoch   8: 1.00e-04
  Epoch   9: 1.00e-04
  Epoch  10: 1.00e-04
  Epoch  11: 1.00e-04
  Epoch  12: 1.00e-04
  Epoch  13: 1.00e-04
  Epoch  14: 1.00e-04
  Epoch  15: 1.00e-04
  Epoch  16: 1.00e-04
  Epoch  17: 1.00e-04
  Epoch  18: 1.00e-04
  Epoch  19: 1.00e-04
  Epoch  20: 1.00e-04
  Epoch  21: 1.00e-04
  Epoch  22: 1.00e-04
  Epoch  23: 5.00e-05
  Epoch  24: 5.00e-05
  Epoch  25: 5.00e-05
  Epoch  26: 1.00e-04
  Epoch  27: 1.00e-04
  Epoch  28: 1.00e-04
  Epoch  29: 1.00e-04
  Epoch  30: 1.00e-04
  Epoch  31: 1.00e-04
  Epoch  32: 1.00e-04
  Epoch  33: 1.00e-04
  Epoch  34: 1.00e-04
  Epoch  35: 1.00e-04
  Epoch  36: 1.00e-04
  Epoch  37: 1.00e-04
  Epoch  38: 1.00e-04
  Epoch  39: 1.00e-04
  Epoch  40: 1.00e-04
  Epoch  41: 1.00e-04
  Epoch  42: 5.00e-05
  Epoch  43: 5.00e-05
  Epoch  44: 5.00e-05

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
