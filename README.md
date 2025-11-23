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

> _This section will be completed later._  
> (Normalization, segmentation, labeling, train/validation split, etc.)

---

