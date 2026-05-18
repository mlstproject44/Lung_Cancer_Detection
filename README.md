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

Segmented binary masks are used as target value for U-Net model. They are created as **anisotropic ellipsoids** rather than spheres to correctly represent each nodule's extent in voxel space given the non-uniform voxel spacing of CT scanners. Nodule centers from the LUNA16 annotations are provided in world coordinates (mm) and are converted to voxel indices using each scan's origin and spacing metadata.

### Lung Masks

**Lung parenchyma**, is segmented using HU thresholding ([-1000, -400]). Algorithm utilizes flood-fill technique to first remove external air from CT slice, followed by labelling connected components inside interior mask. If any component is greater than 0.8 times median of largest two components, then two lungs touch visually in that slice and they appear as one blob. For the purpose of accurate diagnosis, it is necessary to split them into two regions. That is done by eroding that region until two separate seeds are left, followed by reconstruction of lungs through Voronoi partition where every voxel in original mask gets assigned to seed closest to it. At the end, noise was removed and lung masks were mapped to original CT scans. Only lung voxels keep their value.

### Patch Extraction

Training on entire CT scans would be computationally heavy. Instead, script extracts 3D patches for training segmentation model. Prior to that, **Hounsfield Units** (HU - the standard unit for CT scan pixel intensity values) are normalized to [0,1] range for neural network imput. On each scan 80 patches was extracted (71,040 patches in total).

For each scan, it extracts:
 - **Positive patches**: Centered around nodule locations (from the mask) with random jitter
 - **Negative patches**: Random locations within the lung tissue (using pre-computed lung voxel coordinates)

Around 60% of patches contain nodules. This, however, doesn't introduce class imbalance because nodules are small, and model **needs more exposure** to nodule samples because even on positive scans they occupy around **1%** of an image at most. Nodules in patches should be spread across entire patch, and not located in the middle always. If all patches had centered nodules, U-Net would produce suboptimal results during inference on full scans.

### Train-Test-Val Split

In search for efficient sampling strategy, it would be wrong to opt for random sampling. It sounds as reasonable option, but nodule count across scans varies enormously. Some scans have no nodules at all, some have a single sub-millimeter nodule, and some have multiple large masses with tens of thousands of positive voxels each.

To solve problem of class imbalance and propose efficient alternative to 10-fold cross validation, we introduced **stratified voxel split**. It is a dataset partitioning strategy that ensures train, validation, and test sets have statistically similar distributions of total nodule voxel counts per scan. Each scan is binned in into one of five strata based on data-driven percentile thresholds computed from the full dataset. stratification uses (π/6) * d³ sphere volume, which is a diameter-cubed proxy. This is better than raw diameter (closer to volume), but it is not actual voxel counts. The thresholds are in mm³, not voxels. Scans without no nodules at all are placed in no-nodules strata. This is important because no-nodule scans are necessary to teach the model to suppress false positives. If we exclude them from splitting logic, it would silently corrupt all three sets.

**Table I: Stratum Distribution by Split**

| Stratum    | Train | Val | Test |
|------------|-------|-----|------|
| no_nodules | 200   | 30  | 57   |
| tiny       | 105   | 15  | 30   |
| small      | 105   | 15  | 30   |
| medium     | 105   | 15  | 30   |
| large      | 105   | 16  | 30   |
| **Total**  | **620**| **91** | **177** |

---

## Stage 1: Candidates Generation

First part of the pipeline is to go through CT scans and generate candidates, that is potential lung nodules. Training goals are to achieve high recall (**>90%**) to capture all true positives, and maximize precision as much as possible.

### U-Net Architecture

U-Net serves as the candidate generation stage of the two-stage nodule detection pipeline. Given a preprocessed 96x96x96 voxel patch extracted from a CT scan, the model produces a binary segmentation mask that indicates the probability of each voxel belonging to a nodule. The architecture has been extended from the basic U-Net, to operate on three-dimensional data and augmented with custom covariance attention gates and residual connections, which basically gives us a 3D Attention U-Net architecture tailored for LUNA16 dataset and challenge.

Within the two-stage detection framework, the 3D Attention U-Net serves exclusively as a candidate generation module. Its task is to produce a segmentation map over the input patch, identifying all voxels that may belong to a nodule. The objective at this stage is to maximise recall-ensuring that true nodules are not missed-even at the cost of generating false positive candidates

### Training the model

The model is trained using a weighted combination of three loss terms: 0.4 * BCE + 0.35 * Focal + 0.25 * Dice. Binary Cross-Entropy (BCE) handles per-voxel classification with a positive class weight of 10 to account for the severe class imbalance between nodule and background voxels. A Channel-Aware Focal Loss is also included, which dynamically increases the positive weight based on the mean channel importance across the attention gates. Dice loss directly optimizes the overlap between the predicted and ground-truth nodule masks, which is especially important under class imbalance since it is naturally insensitive to the large number of true negatives.

The model is trained with AdamW using two separate parameter groups: the main network parameters use a learning rate of 10^−4 with weight decay 10^−5, while the learnable channel importance scalars use a higher learning rate of 10−2 with no weight decay. A ReduceLROnPlateau scheduler reduces the learning rate by a factor of 0.5 after 3 epochs without improvement in validation loss, with a minimum learning rate of 10^−6.

---

## Stage 2: False Positives Reduction

### Data processing

To prepare data for training **ResNet** we had to repeat patch extraction process, this time around candidates. Extraction process was pretty much the same as for U-Net. Voxel stratified split from stage 1 was used.

### Hard-negatives or Random Sampling?

Training with every extracted patch wouldn't be possible because of **extreme class imbalance**. To fix imbalance, we could either randomly sample negatives or train warmup model to find out 'hard' nodules (hard nodules would be hard-to-detect nodules) and reduce negatives to desired ratio. We prepared both ways, but current model was trained on random negatives, with **680** positive patches and **13,600** patches (1:20 ratio).

### ResNet Architecture and Training

ResNet serves as the false positive reduction stage of the two-stage nodule detection pipeline. After generating the candidates for true nodules which we get using 3D Attention U-Net, this stage receives a set of candidate locations and classifies each as either a true nodule or a false positive. The U-Net stage prioritizes high recall, capturing as many true nodules as possible even at the cost of generating numerous false positives. The ResNet stage is designed to maximize precision by detecting whether the candidates are genuine nodules or anatomically similar structures such as blood vessels, airway walls, and scar tissue.

The biggest problem in the false positive reduction stage is handling the imbalance of the classes between the positive and negative candidates (true positives and false positives). To address this imbalance, the present implementation employs weighted random sampling during training. This sampling method makes sure that each mini-batch consists of a balanced representation of positive and negative class. Weighted sampling prevents the classifier from learning only the majority class and forces it to learn discriminative features for true nodules. This approach is complemented by the use of Binary Cross-Entropy with Logits Loss, which, when combined with balanced sampling, has been shown to effectively handle imbalanced medical imaging classification tasks.

The network is trained using Binary Cross-Entropy with Logits Loss, which combines a sigmoid activation with binary cross-entropy loss. This loss function is appropriate for binary classification tasks and, when combined with weighted sampling to address class imbalance, has been shown to achieve strong performance for false positive reduction. Optimization is performed using the Adam optimizer with a learning rate of 0.001. Adam was selected for its adaptive learning rate properties, which often lead to faster convergence compared to stochastic gradient descent, particularly in the early stages of training. A batch size of 32 is used, which balances computational efficiency with gradient stability given the memory constraints of 3D convolutions.

---

## References

### Papers
 - Zhang, H., Peng, Y. & Guo, Y. [Pulmonary nodules detection based on multi-scale attention networks](https://doi.org/10.1038/s41598-022-05372-y). *Sci Rep* 12, 1466 (2022).
 - Hu, Q. et al. [Effective lung nodule detection using deep CNN with dual attention mechanisms](https://www.nature.com/articles/s41598-024-51833-x). *Sci Rep* (2024).
 - Wang, Y. et al. [An attention-based deep learning network for lung nodule malignancy discrimination](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1106937/full). *Front Neurosci* (2022)
 - Hendrix, W., Hendrix, N., Scholten, E.T. et al. [Deep learning for the detection of benign and malignant pulmonary nodules in non-screening chest CT scans](https://doi.org/10.1038/s43856-023-00388-5) *Commun Med* 3, 156 (2023).

---

