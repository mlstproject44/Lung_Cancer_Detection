# Lung Nodule Detection on Luna16 Dataset

This repository contains implementation of lung nodule deteion model trained on LUNA16 dataset. Pipeline consists of 2 stages: **candidates detection** using U-Net and **false positives reduction** with ResNet.

---

## Data Collection

The LUNA16 dataset is a curated subset of the **LIDC-IDRI** CT scan collection, designed specifically for lung nodule analysis. (**LU**ng **N**odule **A**nalysis). In total, **888** CT scans are included, which are separated into 10 distinct subsets. You can find out more about the dataset here:

🔗 **Dataset link:**  
[https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/Download/)

---

## Data Preprocessing

Thorough data preprocessing is necessary for reliable and accurate model. It consists of several steps:


### Segmentation Masks for Nodules

Uses **annotations.csv** to locate nodules on a CT scan, and creates a binary mask with a filled ellipsoid accounting for anisotropic voxel spacing. Segmented masks are used as target value for U-Net model. Radius is converted from mm to voxel dimension (x,y,z -> z,y,x for array indexing). Different number of voxels create **ellipsoid** in voxel space (sphere in physical).

### Lung Masks

We've segmented **lung parenchyma** from CT scans to filter out bone, heart or muscle tissue. This reduces detection area and false positives, as model can mistake other tissue as nodule. Everything outside lungs is converted to black color, that is **HU value of air** (~ -1000). Lung masks are used during patch extraction, to focus it on lung tissue.

### Patch Extraction

Script extracts **64x64x64** 3D patches for training segmentation model. Prior to that, **Hounsfield Units** (HU - the standard unit for CT scan pixel intensity values) are normalized to [0,1] range for neural network imput. On each scan 80 patches was extracted (71,040 patches in total).

For each scan, it extracts:
 - **Positive patches**: Centered around nodule locations (from the mask) with random jitter
 - **Negative patches**: Random locations within the lung tissue (using pre-computed lung voxel coordinates)

Ratio of positive to negative patches is 7:3. This, however, doesn't introduce class imbalance because nodules are small, and model **needs more exposure** to nodule samples because even on positive scans they occupy around **1%** of an image at most. Nodules in patches should be spread across entire patch, and not located in the middle always. If all patches had centered nodules, U-Net would produce suboptimal results during inference on full scans.

### Train-Test-Val Split

We've tried many splits, namely random splitting patients inside each LUNA16 subfolder (0.7 for train, 0.1 for val and 0.2 for test) and stratified data splitting, categorizing data based on nodule diametar. Both methods produced poor models, because random split created **class imbalance** among nodules by separating most large (>15mm) in test, medium (10-15mm) in train and small (<10mm) nodules in validation set. Nodule diametar also plays minor role, as nodules with different number of voxels can have same diametar. That is why **stratified voxel split** produced the best and most generalizable results. It ensures that each set is balanced based on count of voxels in nodules.

---

## Stage 1: Candidates Generation

First part of the pipeline is to go through CT scans and generate candidates, that is potential lung nodules. Training goals are to achieve high recall (**>90%**) to capture all positives, and maximize precision as much as possible.

### U-Net Architecture

**U-Net** is powerful Convolutional Neural Network architecture, named for its U-shaped design. It is primarily used for image segmentation and we decided to use it because it precisely outlines objects of interest (nodules), even with limited data. We build  **5-level U-Net** (4 encoder/decoder levels + 1 bottleneck level) to successfully capture essential nodule features. Attention gates are added at the end of every skip connection to filter irrelevant skip connection features. Dropout is added for regularization and gradient checkpoint as optional memory optimization.

### Training the model

To achieve target recall without overfitting, we tried many combinations of **focal loss**, **dice loss** and **Binary Cross Entropy with logits loss**. Our final model was trained using **0.7 * bce + 0.3 * dice** (which doesn't imply that it is the best combination of weights, only that it worked). Among other parameters, we used batch size of **16**, learning rate of **0.001** and **50** epochs. Interesting parameter is positive fraction of **0.7**, which ensures that 70% of patches model is exposed to have nodules. Since nodules are small, even positive patches are mostly background so this ensures enough exposure. 

Epoch we decided to use as final model had recall of **87%** and **19%** precision. However, because of strong generalizability, model had **92%** recall and **25%** precision on test set.

To generate candidates, we ran trained U-Net through original CT scans from LUNA16 dataset using sliding windows. To label found candidates, we used annotations.csv file from LUNA16 website, which contains locations of true positives (all others are false positives). U-Net generated **100,374** total candidates, among which 995 were positives. At this point, we moved to stage 2.

---

## Stage 2: False Positives Reduction

### Data processing

To prepare data for training **ResNet** we had to repeat patch extraction process, this time around candidates. Extraction process was pretty much the same as for U-Net. Voxel stratified split from stage 1 was used too.

### Hard-negatives or Random Sampling?

Training with every extracted patch wouldn't be possible because of **extreme class imbalance**. To fix imbalance, we could either randomly sample negatives or train warmup model to find out 'hard' nodules (hard nodules would be hard-to-detect nodules) and reduce negatives to desired ratio. We prepared both ways, but current model was trained on random negatives, with **680** positive patches and **13,600** patches (1:20 ratio).

### ResNet Architecture and Training

To reduce false positives, we decided to use **ResNet-18** - another powerful Convolutional Neural Network with 18 layers. It is known for efficient use of residual blocks to combat vanishing gradients, which make it great complement to stage 1 U-Net. We trained ResNet, implementing padding for patches taken from edges and augmentation techniques, namely random rotation, random flip and random itensity shift to simulate different scan angles and noise levels. Using **Adam** as optimizer, **0.0001** learning rate,**BCE with logits loss** as loss function (because of class imbalance) we got astonishing results on test set - only **20 false negatives** and more than 20,000 false positives rejected. Around 204 false positives were classified as true nodules, however, certified supervisor would be able to spot the missclassification. Final metrics are: Test AUC: **0.941**, Sensitivity: **90.0%**, Specificity: **98.9%**.

---

## Demo APP

You can find demo app on [this link](https://drive.google.com/drive/folders/1on5i8kU2F7qcIUXiWxasoV9Yc8U3-Q_F?usp=drive_link) and try it yourself.

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

