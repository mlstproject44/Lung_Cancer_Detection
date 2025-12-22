# Lung_Cancer_Detection
# Lung Nodule Detection, Localization, and Classification Using LUNA16

This project focuses on detecting, localizing, and classifying lung nodules using the **LUNA16** dataset.  
All development was done in **Python**, using medical-imaging libraries such as SimpleITK and NumPy.

---

## 1. Dataset: LUNA16

The LUNA16 dataset is a curated subset of the LIDC-IDRI CT scan collection, designed specifically for lung nodule analysis.

🔗 **Dataset link:**  
[https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/Download/)

LUNA16 consists of **10 subsets** of CT scans.  
Due to storage limitations, our team downloaded **3 subsets each (so 9 in total or 90%) locally** for initial processing.

---

## 2. Data Collection

1. Explored available datasets and selected **LUNA16** for lung nodule analysis.  
2. Downloaded **nine out of ten** subsets (each team member downloaded three subsets each locally).  
3. The full dataset size was approximately **66 GB**, which exceeded our available device storage, so we performed data reduction to approximately 6.6G

---

## 3. Data Reduction (66 GB → 6.6 GB)##
This script downscales `.mhd` CT scans using **SimpleITK** to reduce dataset size while keeping the scans usable for machine learning and visualization.

- Loads a CT scan  
- Changes its voxel spacing (resolution)  
- Resamples the image to the new spacing  
- Saves the downsampled scan  
- Reports how much space was saved  
- Can process an entire folder of scans  
Key functions:

### `downsample_scan()`
- Reads the scan  
- Computes the new image size based on the target spacing  
- Resamples the scan using linear interpolation  
- Saves the output and prints the size reduction  

### `batch_downsample()`
- Finds all `.mhd` files in a directory  
- Applies `downsample_scan()` to each one  
- Prints a summary of total space saved  

### `compare_quality()`
- Shows original vs. downsampled slices side-by-side  
- Saves a comparison image  


---

## 4. Data Preprocessing

Data preprocessing pipeline has been implemented with the following components:

### 4.1 Segmentation Mask Creation (`segmentation_masks.py`)
- Converts nodule annotations (world coordinates + diameter) to binary 3D masks
- Creates ellipsoid masks accounting for anisotropic voxel spacing
- Handles overlapping nodules using logical OR operation
- Processes all 10 LUNA16 subsets

### 4.2 PyTorch Dataset (`luna16_dataset.py`)
- Custom PyTorch Dataset class for loading CT scans and masks
- Extracts 3D patches (default: 128×128×128) for training
- Implements data augmentation: random flips, rotations, intensity shifts
- Handles class imbalance with configurable positive/negative patch ratio
- HU value normalization (clipping to [-1000, 400] range)

### 4.3 Train/Test/Val Splits (`train_test_val.py`)
- Stratified splitting across all subsets (70% train, 20% test, 10% val)
- Reproducible splits with random seed
- JSON-based split storage for consistency
- Metadata tracking for experiment reproducibility

---

## 5. Two-Stage Detection Pipeline

### Stage 1: U-Net Segmentation

**3D U-Net** for volumetric segmentation (22.6M parameters)
- Input: CT scan patches (128×128×128)
- Output: Binary segmentation mask
- Loss: Combined BCE + Dice
- Metrics: Dice coefficient, Sensitivity, Specificity

**Train U-Net:**
```bash
cd nodule_detection/src/models
python train_unet.py --epochs 50 --batch_size 2
```

### Stage 2: ResNet False Positive Reduction

**3D ResNet** for candidate classification (3.6M parameters)
- Input: Candidate patches from U-Net (32×32×32)
- Output: True nodule vs False positive
- Extracts connected components from U-Net masks
- Classifies each candidate using ResNet

**Extract candidates and train ResNet:**
```bash
# Extract candidates from trained U-Net
python train_resnet.py --mode extract --unet_checkpoint checkpoints/unet_best.pth

# Train ResNet on candidates
python train_resnet.py --mode train --epochs 30
```

### Pipeline Overview
```
CT Scan → U-Net → Segmentation Mask → Extract Candidates → ResNet → Final Detections
```

For detailed usage, see [`nodule_detection/src/models/README.md`](nodule_detection/src/models/README.md)

---

## 6. Project Structure

```
Lung_Cancer_Detection/
├── README.md
└── nodule_detection/
    ├── data/
    │   └── annotations.csv
    ├── notebooks/
    │   └── LUNA16_EDA.ipynb
    └── src/
        ├── data_preprocessing/
        │   ├── segmentation_masks.py    # Create binary masks
        │   ├── luna16_dataset.py         # PyTorch Dataset
        │   └── train_test_val.py         # Data splitting
        └── models/
            ├── unet3d.py                 # U-Net segmentation
            ├── resnet3d.py               # ResNet classification
            ├── losses.py                 # Loss functions
            ├── metrics.py                # Metrics
            ├── extract_candidates.py     # Extract candidates from U-Net
            ├── train_unet.py             # Train U-Net
            ├── train_resnet.py           # Extract & train ResNet
            └── README.md                 # Usage guide
```

---

## 7. References

### Papers
1. Setio et al. (2017). "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge"
2. Çiçek et al. (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
3. Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
4. Milletari et al. (2016). "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"

### Datasets
- LUNA16: https://luna16.grand-challenge.org/

### Implementation References
- [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
- [ellisdg/3DUnetCNN](https://github.com/ellisdg/3DUnetCNN)
- [fepegar/unet](https://github.com/fepegar/unet)

---

