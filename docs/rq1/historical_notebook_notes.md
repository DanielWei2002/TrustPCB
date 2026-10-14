# Historical notebook explanations

Archived prose, not new execution or independently verified conclusions. See [conversion](notebook_extraction.md).

## 02_similarity_aware_split — cell 0

# TrustPCB: Similarity-Aware Dataset Split

## Purpose

This notebook constructs and validates a similarity-aware train-validation split for the DsPCBSD+ dataset used in the TrustPCB research project.

The original DsPCBSD+ dataset contains:

- 10,259 images
- 8,208 training images
- 2,051 validation images
- 20,276 bounding-box annotations
- 9 defect classes

A previous dataset audit found:

- no unreadable images,
- no missing image-label pairs,
- no structurally invalid YOLO annotations,
- no exact pixel-identical duplicates, and
- a small number of highly similar image pairs occurring across the supplied training and validation split.


## 02_similarity_aware_split — cell 1

## 1.0: Environment and Dataset Path Verification

Before constructing a new split, the project environment and dataset paths are verified.

This step ensures that:

- the notebook is using the TrustPCB Python environment,
- the original DsPCBSD+ YOLO dataset is accessible,
- the supplied training and validation image folders are available,
- the corresponding label folders are available,
- previously generated audit outputs can be accessed, and
- the directory for new split manifests is identified.

The original files under `data/raw/` will remain unchanged throughout the similarity-aware split construction.

## 02_similarity_aware_split — cell 3

## 1.1 Supplied Train-Validation Split Inventory

Before constructing a similarity-aware split, the original DsPCBSD+ train-validation split is reconstructed directly from the official dataset folders.

This establishes the baseline split that will later be compared against the similarity-aware split.

The following checks are performed:

- count training and validation images,
- count corresponding label files,
- verify that image filenames are unique within each split,
- check whether any filename appears in both training and validation, and
- confirm the overall train-validation proportion.

No files are moved or modified during this step.

## 02_similarity_aware_split — cell 5

## 1.2 Load Confirmed Cross-Split Similarity Pairs

The previous dataset audit identified a small set of high-confidence near-duplicate image pairs that occur across the supplied training and validation split.

This section loads the manually reviewed near-duplicate results so they can be used to guide the construction of a similarity-aware split.

Only the confirmed high-confidence pairs are used at this stage.

## 02_similarity_aware_split — cell 7

## 1.3 Similarity Group Construction

The confirmed near-duplicate results are currently stored as individual train-validation image pairs.

Before constructing a similarity-aware split, these pairwise relationships are converted into similarity groups.

This is necessary because an image may potentially be related to more than one other image. For example, if image A is similar to image B and image B is similar to image C, all three images should be treated as one connected group rather than as two independent pairs.

The following operations are performed:

- combine the confirmed train-validation pair relationships,
- identify connected groups of related images,
- assign a unique group ID to each similarity group,
- count the number of images in each group, and
- verify how many supplied training and validation images belong to each group.

All images within the same similarity group will later be kept on the same side of the similarity-aware split.

No dataset files are moved or modified during this step.

## 02_similarity_aware_split — cell 9

## 1.4 Similarity Group Class Composition

Before assigning the similarity groups to the new train-validation split, the annotation composition of each group is examined.

This is important because moving a similarity group from one side of the split to the other can change the distribution of defect classes. The new split should remove confirmed cross-split similarity while preserving the original class distribution as closely as practical.

The following information is collected for each similarity group:

- the supplied training image,
- the supplied validation image,
- the number of bounding boxes in each image,
- the defect class IDs present in each image, and
- whether both images contain the same set of defect classes.

This information will later be used when deciding how the similarity groups should be assigned in the similarity-aware split.

No train-validation assignments are changed during this step.

## 02_similarity_aware_split — cell 11

## 1.5 Supplied Split Class Distribution

The class distribution of the supplied train-validation split is calculated before creating the similarity-aware split.

This gives us a reference to compare against later.

The following are checked:

- annotation count for each class,
- class percentage in training and validation, and
- difference between the two splits.

The similarity-aware split should keep these distributions as close as possible to the supplied split.

## 02_similarity_aware_split — cell 13

## 1.6 Similarity-Aware Split Strategy

The similarity-aware split will be created by modifying the supplied split as little as possible.

The 19 confirmed similarity groups each contain:

- 1 training image,
- 1 validation image, and
- 2 images in total.

Both images from each similarity group must be assigned to the same split.

Instead of creating a completely new random split, the supplied split is kept unchanged for all unaffected images.

For the 19 similarity groups:

- some groups will be assigned entirely to training,
- the remaining groups will be assigned entirely to validation,
- only one image from each group needs to change split, and
- the final train-validation ratio should remain as close as possible to the original 80/20 ratio.

This minimal-change approach allows the supplied and similarity-aware splits to remain directly comparable.

## 02_similarity_aware_split — cell 15

## 1.7 Select Similarity Group Allocation

The 19 similarity groups must be assigned entirely to either training or validation.

Two group-count allocations can keep the dataset close to the original 80/20 split:

- 9 groups to training and 10 groups to validation, or
- 10 groups to training and 9 groups to validation.

The specific groups should not be selected randomly.

Each possible allocation is compared based on:

- training and validation class distribution,
- difference from the supplied split class percentages, and
- final train-validation image count.

The allocation with the smallest class-distribution change will be selected.

## 02_similarity_aware_split — cell 17

## 1.8 Verify Selected Split Distribution

The selected group allocation is checked before creating the final similarity-aware split.

The following are compared:

- supplied training class distribution,
- similarity-aware training class distribution,
- supplied validation class distribution,
- similarity-aware validation class distribution, and
- percentage-point change for each class.

The purpose is to confirm that removing cross-split similarity does not significantly change the class composition of the dataset.

## 02_similarity_aware_split — cell 19

## 1.9 Create Similarity-Aware Split Manifests

The selected group allocation is now used to create the similarity-aware train-validation split.

The original dataset files are not moved or modified.

Instead, split manifests are created to record:

- which images belong to training,
- which images belong to validation,
- the original supplied split,
- the new similarity-aware split, and
- which images changed split.

All images within the same confirmed similarity group are assigned to the same split.

## 02_similarity_aware_split — cell 21

## 1.10 Validate Similarity-Aware Split

The new similarity-aware split is checked before it is used for training.

The following are verified:

- no image appears in both training and validation,
- all 10,259 images are included,
- all confirmed similarity groups stay within one split,
- final train-validation size remains close to 80/20, and
- the number of moved images matches the expected 19 images.

This confirms that the new split removes the confirmed cross-split similarity without changing the dataset unnecessarily.

## 02_similarity_aware_split — cell 23

## 1.11 Summary

A similarity-aware train-validation split was constructed from the supplied DsPCBSD+ split.

The main results are:

- original split: 8,208 training and 2,051 validation images,
- new split: 8,207 training and 2,052 validation images,
- final proportion: 80.00% training and 20.00% validation,
- 19 confirmed cross-split near-duplicate groups were considered,
- 19 images changed split assignment,
- no image appears in both splits,
- no images are missing or added,
- no confirmed similarity group remains across training and validation, and
- maximum class-distribution change is only 0.0177 percentage points.

The similarity-aware split therefore removes the confirmed high-confidence cross-split similarities while keeping the dataset size and class distribution almost unchanged.

The split manifests are saved under `data/splits/` for use in later experiments.

## 03_yolo_baseline — cell 0

# TrustPCB: Lightweight YOLO Baseline

## 1.0 Purpose

This notebook establishes the baseline PCB defect detector used in the TrustPCB research project.

The baseline provides a reference detector before introducing the reliability framework.

The main tasks are:

- verify the YOLO training environment,
- prepare the supplied and similarity-aware dataset configurations,
- train one lightweight YOLO detector,
- evaluate standard object-detection performance,
- record model size and inference efficiency, and
- prepare predictions for later reliability analysis.

The first baseline will use the supplied DsPCBSD+ split.

The similarity-aware split will later be used as an evaluation safeguard to check whether the identified cross-split similarities affect detector performance.

## 03_yolo_baseline — cell 1

## 1.1 Environment Verification

The TrustPCB Python environment is checked before preparing the baseline experiment.

The following are verified:

- Python environment,
- PyTorch version,
- Ultralytics version,
- CUDA availability,
- GPU model, and
- project and dataset paths.

The baseline training will only begin after GPU access is confirmed.

## 03_yolo_baseline — cell 3

## 1.2 Dataset and Split Verification

The DsPCBSD+ dataset and the supplied and similarity-aware split definitions are verified before baseline training.

This step confirms:

- the expected image and label directories,
- matching image-label counts,
- the expected training and validation split sizes,
- no overlap between training and validation assignments,
- use of the same complete image set, and
- the number of images reassigned by the similarity-aware split.

This ensures that the baseline experiments use a complete and consistent dataset before the YOLO configurations are prepared.

## 03_yolo_baseline — cell 5

## 1.3 Class Mapping Verification

The numerical YOLO class IDs are verified against the published DsPCBSD+ class statistics before baseline training.

The dataset paper defines nine PCB defect categories and reports the number of annotations for each category in the supplied training and validation sets.

The observed annotation counts from the local YOLO labels are compared with these published values to recover and verify the mapping between numerical class IDs and defect names.

## 03_yolo_baseline — cell 7

## 1.4 YOLO Dataset Configuration

YOLO dataset configurations are prepared for the supplied and similarity-aware split schemes.

The original DsPCBSD+ image and label directories remain unchanged. Each experimental split is represented using image-path lists so that alternative train-validation assignments can be evaluated without moving or duplicating the raw dataset.

This step:

- generates image-path lists for both split schemes,
- creates the corresponding YOLO dataset configuration files, and
- validates the image paths, annotations, split sizes, and class definitions.

Both configurations use the same 10,259 images and the same nine verified defect classes. Only the training and validation assignments differ.

## 03_yolo_baseline — cell 10

# 2.0 Preliminary Baseline Experiment

The preliminary baseline experiment establishes a lightweight reference detector for TrustPCB.

A pretrained YOLOv8n detector is used as the fixed baseline model. The purpose is not to modify the detector architecture, but to establish standard object-detection performance before the reliability framework is introduced.

The experiment first performs a short smoke test to verify the complete training pipeline. Two matched baseline runs are then conducted using:

- the supplied DsPCBSD+ split, and
- the similarity-aware split.

Both runs use the same model, training settings, random seed, and software environment so that the dataset split is the primary experimental difference.

## 03_yolo_baseline — cell 11

## 2.1 Baseline Configuration

The preliminary baseline training configuration is fixed before any model training is performed.

The experiment uses:

- YOLOv8n with pretrained weights,
- an input image size of 640 × 640,
- 100 training epochs,
- a fixed random seed of 24209199,
- deterministic training where supported,
- a fixed batch size,
- standard Ultralytics training augmentation, and
- the same configuration for both dataset splits.

Ultralytics training outputs and run arguments are retained so that the experiment can be reproduced and compared consistently.

## 03_yolo_baseline — cell 13

## 2.2 Smoke Test

A short training run is performed on the supplied DsPCBSD+ split before the full baseline experiment.

The smoke test uses the same model and core training configuration as the preliminary baseline, but training is limited to two epochs.

The purpose is to verify that:

- the pretrained YOLOv8n model loads correctly,
- the dataset and annotations are accepted by Ultralytics,
- GPU training operates without errors,
- training and validation complete successfully, and
- experiment outputs are saved to the expected project directory.

The smoke-test metrics are used only for pipeline verification and are not treated as baseline results.

## 03_yolo_baseline — cell 15

## 2.3 Supplied-Split Baseline

The first full preliminary baseline is trained using the original supplied DsPCBSD+ train-validation split.

YOLOv8n is trained for 100 epochs using the frozen baseline configuration established in Section 2.1.

This run provides:

- standard object-detection performance on the supplied split,
- per-class detection behaviour,
- training and validation learning curves,
- model weights and experiment metadata, and
- the reference result for the later comparison with the similarity-aware split.

The resulting metrics are treated as preliminary baseline results for TrustPCB.

## 03_yolo_baseline — cell 17

## 2.4 Similarity-Aware Baseline

The second full preliminary baseline is trained using the similarity-aware DsPCBSD+ train-validation split established during the dataset audit.

The detector architecture, training configuration, random seed, and software environment are kept identical to the supplied-split baseline. The dataset split is therefore the primary experimental difference between the two runs.

This experiment evaluates whether the limited cross-split near-duplicate groups identified in the supplied DsPCBSD+ split materially affect baseline detection performance.

The results are compared with the supplied-split baseline in Section 2.5.

## 03_yolo_baseline — cell 19

## 2.5 Baseline Comparison

The supplied-split and similarity-aware YOLOv8n baselines are compared using their best validation checkpoints, selected according to mAP@0.5:0.95.

Both experiments use the same detector architecture, training configuration, random seed, and software environment. The dataset split is the primary experimental difference.

The comparison focuses on precision, recall, mAP@0.5, and mAP@0.5:0.95 to assess whether the limited cross-split image similarity identified during the dataset audit materially affects aggregate baseline performance.

This comparison is treated as preliminary single-seed evidence for the RQ1 evaluation safeguard rather than as a final repeated-run statistical conclusion.

## 03_yolo_baseline — cell 21

### Preliminary Comparison Result

The similarity-aware split produced performance comparable to the supplied DsPCBSD+ split. The similarity-aware baseline achieved slightly higher recall, mAP@0.5, and mAP@0.5:0.95, although the absolute differences were small.

The similarity-aware run achieved an mAP@0.5 of 0.8486 and an mAP@0.5:0.95 of 0.5084, compared with 0.8452 and 0.5053, respectively, for the supplied split.

These preliminary single-seed results suggest that the limited confirmed cross-split near-duplicate groups identified during the dataset audit do not materially affect aggregate YOLOv8n baseline performance under the current experimental setting. The result supports the use of similarity-aware splitting as an evaluation safeguard while avoiding a claim that the supplied split is free from all forms of data leakage.
