# Lung Nodule Detection on Luna16 Dataset

This repository contains implementation of lung nodule deteion model trained on LUNA16 dataset. Pipeline consists of 2 stages: candidates detection using U-Net and false positives reduion with ResNet.

---

## Data Collection

The LUNA16 dataset is a curated subset of the LIDC-IDRI CT scan collection, designed specifically for lung nodule analysis. (**LU**ng **N**odule **A**nalysis). In total, 888 CT scans are included, which are separated into 10 distinct subsets. You can find out more about the dataset here:

🔗 **Dataset link:**  
[https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/Download/)

---

## Data Preprocessing

Thorough data preprocessing is necessary for reliable and accurate model. It consists of several steps:


### Segmentation Masks for Nodules

Uses annotations.csv to locate nodules on a CT scan, and creates a binary mask with a filled ellipsoid accounting for anisotropic voxel spacing. Segmented masks are used as target value for U-Net model. Radius is converted from mm to voxel dimension (x,y,z -> z,y,x for array indexing). Different number of voxels create ellipsoid in voxel space (sphere in physical).

### Lung Masks

We've segmented lung parenchyma from CT scans to filter out bone, heart or muscle tissue. This reduces detection area and false positives, as model can mistake other tissue as nodule. Everything outside lungs is converted to black color, that is HU value of air (~ -1000). Lung masks are used during patch extraction, to focus it on lung tissue.

### Patch Extraction

Script extracts 64x64x64 3D patches for training segmentation model. Prior to that, Hounsfield Units (HU - the standard unit for CT scan pixel intensity values) are normalized to [0,1] range for neural network imput. On each scan 80 patches was extracted (71,040 patches in total).

For each scan, it extracts:
 - **Positive patches**: Centered around nodule locations (from the mask) with random jitter
 - **Negative patches**: Random locations within the lung tissue (using pre-computed lung voxel coordinates)

Ratio of positive to negative patches is 7:3. This, however, doesn't introduce class imbalance because nodules are small, and model needs more exposure to nodule samples because even on positive scans they occupy around 1% of an image at most. Nodules in patches should be spread across entire patch, and not located in the middle always. If all patches had centered nodules, U-Net would produce suboptimal results during inference on full scans.

### Train-Test-Val Split

We've tried many splits, namely random splitting patients inside each LUNA16 subfolder (0.7 for train, 0.1 for val and 0.2 for test) and stratified data splitting, categorizing data based on nodule diametar. Both methods produced poor models, because random split created class imbalance among nodules by separating most large (>15mm) in test, medium (10-15mm) in train and small (<10mm) nodules in validation set. Nodule diametar also plays minor role, as nodules with different number of voxels can have same diametar. That is why stratified voxel split produced the best and most generalizable results. It ensures that each set is balanced based on count of voxels in nodules.

---

## References

### Papers
 - Zhang, H., Peng, Y. & Guo, Y. [Pulmonary nodules detection based on multi-scale attention networks](https://doi.org/10.1038/s41598-022-05372-y). *Sci Rep* 12, 1466 (2022).
 - Hu, Q. et al. [Effective lung nodule detection using deep CNN with dual attention mechanisms](https://www.nature.com/articles/s41598-024-51833-x). *Sci Rep* (2024).
 - Wang, Y. et al. [An attention-based deep learning network for lung nodule malignancy discrimination](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1106937/full). *Front Neurosci* (2022)
 - Hendrix, W., Hendrix, N., Scholten, E.T. et al. [Deep learning for the detection of benign and malignant pulmonary nodules in non-screening chest CT scans](https://doi.org/10.1038/s43856-023-00388-5) *Commun Med* 3, 156 (2023).

### Datasets
- LUNA16: https://luna16.grand-challenge.org/

### Implementation References

---

