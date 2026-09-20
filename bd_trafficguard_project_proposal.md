# BD-TrafficGuard: Robust Bangladeshi Traffic Sign Detection and Recognition

## 1. Project Overview

**Proposed title:** BD-TrafficGuard: Robust Bangladeshi Traffic Sign Detection and Recognition Under Real-World Road Conditions

This project aims to build a deep learning based computer vision system that can detect and recognize Bangladeshi traffic signs from real road images. In simple terms, the model should look at a road image, find traffic signs, draw bounding boxes around them, and classify what type of sign they are.

The important part is robustness. In Bangladesh, traffic signs may appear far away, faded, tilted, partially blocked by trees or vehicles, blurred by motion, affected by low light, or captured from unusual angles. A model that works only on clean images is not enough. The project therefore focuses on evaluating and improving performance under real-world image degradations such as blur, rain/fog simulation, low brightness, glare, occlusion, compression, and small object size.

This makes the project suitable for a Pattern Recognition Lab because it includes data preparation, baseline comparison, motivated training choices, proper evaluation metrics, ablation studies, and qualitative failure analysis.

## 2. Beginner-Friendly Explanation

Traffic sign recognition has two related tasks:

1. **Detection:** Find where the sign is in the image.
2. **Recognition/classification:** Identify which traffic sign it is.

For example, if an image contains a "Sharp left turn" sign, the system should locate the sign and label it correctly. If the sign is small or blurry, the system should still try to recognize it reliably.

The basic pipeline can be:

1. Collect/download Bangladeshi traffic sign image datasets.
2. Prepare labels and train/validation/test splits.
3. Train a baseline object detector such as YOLO.
4. Evaluate the model on clean test images.
5. Create degraded test sets with blur, low light, fog, rain, and occlusion.
6. Train a more robust model using degradation-aware augmentation.
7. Compare baseline vs robust model using mAP, precision, recall, F1-score, and inference speed.

## 3. Motivation

Traffic sign detection and recognition is important for:

- driver assistance systems,
- autonomous or semi-autonomous vehicles,
- road safety monitoring,
- traffic rule awareness,
- dashcam-based warning systems,
- smart city road infrastructure mapping.

Many popular traffic sign datasets are from countries such as Germany or China. However, Bangladeshi roads have different sign designs, environmental conditions, road layouts, and visual noise. Therefore, a Bangladesh-specific study can be more locally relevant and potentially publishable.

## 4. Literature Review and Recent Works

### 4.1 Global traffic sign recognition benchmarks

The German Traffic Sign Recognition Benchmark, or GTSRB, is one of the classic traffic sign classification datasets. The official benchmark page describes it as a single-image, multi-class classification problem with more than 40 classes and more than 50,000 images. It also highlights real-world challenges such as illumination change, occlusion, rotation, and weather conditions.

Link: [GTSRB official benchmark page](https://benchmark.ini.rub.de/gtsrb_news.html)

The Mapillary Traffic Sign Dataset is another large-scale dataset containing more than 300 verified traffic sign classes and more than 320,000 labeled traffic signs. It is useful as a reference for large-scale traffic sign recognition, although it may require login or access control depending on the current Mapillary page state.

Link: [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)

### 4.2 YOLO-based traffic sign detection

Recent traffic sign detection research commonly uses YOLO-family models because they are fast and suitable for real-time systems. A 2024 systematic review titled "Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm" summarizes YOLO-based approaches for traffic sign detection and recognition.

Link: [YOLO traffic sign detection systematic review](https://www.mdpi.com/2227-7390/12/2/297)

Ultralytics YOLO models are practical for this project because they provide easy training, validation, prediction, and export tools. YOLO11 and YOLOv8/YOLO11-style detectors can be used as strong baselines.

Links:

- [Ultralytics object detection task documentation](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)

### 4.3 Bangladeshi traffic sign research

A recent Bangladesh-specific dataset paper presents a Bangladesh road traffic sign benchmark dataset with 10,259 real-world traffic sign images and 10,259 annotated images across 31 traffic sign classes. The dataset includes traffic signs captured from various locations in Bangladesh and is designed for training and testing deep learning models.

Links:

- [ScienceDirect dataset article](https://www.sciencedirect.com/science/article/pii/S2352340925002550)
- [Zenodo dataset record](https://zenodo.org/records/11511846)

Another Bangladeshi dataset, BDRoadSigns, is an image-based dataset for supervised classification models. It contains 18 classes of traffic signs captured from several regions of Bangladesh, including Dhaka and Rajshahi.

Link: [BDRoadSigns on Mendeley Data](https://data.mendeley.com/datasets/rnkd74xdtr/1)

FUSED-Net is a recent Bangladeshi traffic sign detection paper focused on limited-data learning. It is built on Faster R-CNN and uses ideas such as pseudo-support sets, embedding normalization, and domain adaptation. The paper evaluates on a Bangladeshi Traffic Sign Detection Dataset and is useful as a strong related work reference.

Link: [FUSED-Net: Detecting Traffic Signs with Limited Data](https://arxiv.org/abs/2409.14852)

### 4.4 Robust traffic sign detection under adverse conditions

A 2025 Scientific Reports paper, "DSF-YOLO for robust multiscale traffic sign detection under adverse weather conditions," addresses traffic sign detection under complex weather. It uses YOLOv8 as a baseline and applies adverse-condition augmentation such as fog, snow, rain, shadow, brightness contrast, and sun flare. This is highly relevant to the proposed robustness direction.

Link: [DSF-YOLO paper in Scientific Reports](https://www.nature.com/articles/s41598-025-02877-0)

## 5. Research Gap

Existing traffic sign recognition work has several limitations:

- Many benchmark datasets are not Bangladesh-specific.
- Bangladesh-specific datasets are still relatively new.
- Many studies report only clean-image performance.
- Real-world degradations such as blur, low light, occlusion, rain, fog, and compression are often not evaluated systematically.
- Lightweight deployment for dashcams or mobile devices is not always considered.

This project can address the gap by creating a Bangladesh-specific robustness evaluation protocol and comparing clean training against degradation-aware training.

## 6. Proposed Novelty

The novelty should not be simply "we used YOLO." A stronger novelty statement is:

**This project proposes a Bangladesh-specific robustness benchmark and baseline system for traffic sign detection and recognition under common real-world image degradations.**

Possible novel contributions:

- A structured corruption benchmark for Bangladeshi traffic signs, called BD-TrafficSign-C.
- Evaluation under blur, low light, glare, fog/rain simulation, occlusion, JPEG compression, and small-object scaling.
- Comparison between standard training and degradation-aware training.
- Class-wise robustness analysis showing which Bangladeshi traffic signs are most vulnerable.
- Lightweight real-time baseline suitable for dashcam or smartphone use.
- Qualitative failure analysis for small, faded, occluded, and confusing traffic signs.

## 7. Datasets

### Primary dataset

**Bangladesh Road Traffic Sign Dataset in Real-World Images**

- 10,259 raw traffic sign images.
- 10,259 annotated images.
- 31 traffic sign classes.
- Captured from different locations in Bangladesh.
- Suitable for detection and recognition tasks.

Links:

- [Zenodo dataset record](https://zenodo.org/records/11511846)
- [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S2352340925002550)

### Optional supporting dataset

**BDRoadSigns**

- 18 Bangladeshi traffic sign classes.
- Image classification dataset.
- Useful for cropped sign classification or transfer learning.

Link: [BDRoadSigns on Mendeley Data](https://data.mendeley.com/datasets/rnkd74xdtr/1)

### Related dataset/paper

**Bangladeshi Traffic Sign Detection Dataset used in FUSED-Net**

- Used for few-shot Bangladeshi traffic sign detection experiments.
- Useful as related work and possible comparison if accessible.

Link: [FUSED-Net paper](https://arxiv.org/abs/2409.14852)

## 8. Candidate Models

### Baseline models

1. **YOLOv8n / YOLOv8s or YOLO11n / YOLO11s**
   - Good first baseline.
   - Fast and practical.
   - Supports real-time detection.

2. **Faster R-CNN**
   - Strong two-stage detector.
   - Often accurate but slower than YOLO.
   - Useful as a baseline comparison.

3. **EfficientNet-B0 or ResNet-50 classifier**
   - Can classify cropped traffic sign images.
   - Useful for a two-stage pipeline: detection first, classification second.

Useful documentation:

- [Ultralytics object detection documentation](https://docs.ultralytics.com/tasks/detect/)
- [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)

## 9. Proposed Methodology

### Step 1: Dataset preparation

- Download the Bangladesh Road Traffic Sign Dataset.
- Inspect label format and class distribution.
- Remove corrupted or unusable images if needed.
- Convert labels to YOLO format if necessary.
- Create train/validation/test split, for example 70/15/15 or 80/10/10.
- Ensure that test images are not augmented versions of training images.

### Step 2: Baseline training

- Train YOLO on the clean dataset.
- Use the validation set for hyperparameter tuning.
- Test only once on the held-out test set.
- Record clean performance.

### Step 3: Robustness test set creation

Create corrupted versions of the test set:

- Gaussian blur.
- Motion blur.
- Low brightness.
- High brightness/glare.
- Rain or fog simulation.
- Partial occlusion.
- JPEG compression.
- Small-object scaling.

This creates a controlled robustness benchmark.

### Step 4: Robust training

Train another model using targeted augmentations:

- blur augmentation,
- brightness/contrast changes,
- random rain/fog/shadow,
- random crop and scale,
- mosaic/mixup,
- class-balanced sampling,
- optional focal loss if class imbalance is severe.

### Step 5: Evaluation

Compare the clean baseline and robust model on:

- clean test set,
- corrupted test sets,
- per-class performance,
- inference speed,
- model size.

### Step 6: Ablation study

Possible ablations:

- no augmentation vs standard augmentation vs degradation-aware augmentation,
- YOLO-nano vs YOLO-small,
- different input sizes, such as 416, 640, and 960,
- with/without class-balanced sampling,
- one-stage YOLO vs two-stage detector + classifier.

### Step 7: Qualitative analysis

Show examples of:

- correct detections in difficult conditions,
- missed small signs,
- wrong classification between visually similar signs,
- false positives from sign-like objects,
- failure under extreme blur or occlusion.

## 10. Evaluation Metrics

Recommended metrics:

- **mAP@0.5:** common object detection metric.
- **mAP@0.5:0.95:** stricter COCO-style metric.
- **Precision:** how many predicted signs are actually correct.
- **Recall:** how many actual signs the model finds.
- **F1-score:** balance between precision and recall.
- **Class-wise recall:** important for rare or safety-critical signs.
- **Robustness drop:** performance decrease from clean test set to corrupted test set.
- **FPS/inference time:** important for real-time use.
- **Model size:** important for mobile or edge deployment.

Example robustness drop:

`Robustness Drop = Clean mAP - Corrupted mAP`

Lower robustness drop means the model is more stable under real-world degradation.

## 11. Expected Results

Expected findings:

- YOLO should perform well on clean images.
- Performance will likely drop under blur, small-object conditions, low light, and occlusion.
- Degradation-aware augmentation should improve corrupted-set performance.
- Some classes will be more fragile than others, especially visually similar signs or signs with small internal symbols.
- Lightweight models may be fast enough for dashcam/mobile use but may struggle with small signs.

## 12. Publication Potential

This project has publication potential if it goes beyond a simple course implementation. A good conference/workshop paper could be built around:

- a Bangladesh-specific traffic sign robustness benchmark,
- systematic evaluation of common visual degradations,
- strong reproducible baselines,
- ablation studies,
- qualitative failure analysis,
- lightweight deployment experiments.

Possible paper title:

**Robust Detection and Recognition of Bangladeshi Traffic Signs Under Visual Degradation: A Benchmark and Lightweight Deep Learning Baseline**

Potential venues:

- local or regional AI/computer vision conferences,
- IEEE/ACM student research tracks,
- intelligent transportation workshops,
- computer vision applications workshops,
- journal extension after adding larger experiments or deployment.

The publication chance becomes stronger if the team releases:

- code,
- train/validation/test split protocol,
- corruption benchmark scripts,
- trained baseline weights,
- result tables and failure cases.

## 13. Future Scope

Possible future extensions:

- Android app for real-time traffic sign alerts.
- Dashcam-based driver assistance system.
- Raspberry Pi / Jetson Nano deployment.
- Video-based tracking across frames.
- Night-time sign detection.
- Bengali text-aware road sign recognition.
- Integration with road inventory mapping.
- Semi-supervised learning from unlabeled Bangladeshi dashcam footage.
- Expanding the dataset with new cities, rural roads, and weather conditions.

## 14. Four-Minute Proposal Presentation Outline

### Slide 1: Problem and Motivation

- Bangladeshi traffic signs are important for road safety.
- Real-world signs can be blurry, small, occluded, faded, or affected by weather.
- Goal: robust detection and recognition from real road images.

### Slide 2: Dataset and Related Work

- Bangladesh Road Traffic Sign Dataset: 10,259 images, 31 classes.
- Related works: GTSRB, YOLO traffic sign detection, FUSED-Net, DSF-YOLO.
- Gap: limited Bangladesh-specific robustness evaluation.

### Slide 3: Methodology

- Train baseline YOLO detector.
- Create corrupted test sets.
- Train robust model with degradation-aware augmentation.
- Compare baseline vs robust model.

### Slide 4: Evaluation and Expected Contribution

- Metrics: mAP, precision, recall, F1, robustness drop, FPS.
- Ablations: augmentation, model size, input resolution.
- Contribution: BD-specific robustness benchmark and lightweight baseline.

## 15. Form-Ready Project Description

We propose to build BD-TrafficGuard, a robust computer vision system for detecting and recognizing Bangladeshi traffic signs in real-world road images. The project will use publicly available Bangladeshi traffic sign datasets and train deep learning models such as YOLO, Faster R-CNN, and EfficientNet/ResNet-based classifiers. Beyond normal clean-image accuracy, we will evaluate robustness under blur, low light, glare, rain/fog simulation, occlusion, compression, and small-object conditions. We will compare baseline models with degradation-aware training, report mAP, precision, recall, F1-score, robustness drop, and inference speed, and perform ablation and qualitative failure analysis. The goal is to create a practical and publishable Bangladesh-specific traffic sign recognition benchmark and baseline system.

## 16. Reference Links

1. [Bangladesh Road Traffic Sign Dataset article, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925002550)
2. [Bangladesh Road Traffic Sign Dataset, Zenodo](https://zenodo.org/records/11511846)
3. [BDRoadSigns, Mendeley Data](https://data.mendeley.com/datasets/rnkd74xdtr/1)
4. [FUSED-Net: Detecting Traffic Signs with Limited Data, arXiv](https://arxiv.org/abs/2409.14852)
5. [DSF-YOLO for robust multiscale traffic sign detection, Scientific Reports](https://www.nature.com/articles/s41598-025-02877-0)
6. [Traffic Sign Detection and Recognition Using YOLO: Systematic Review, MDPI](https://www.mdpi.com/2227-7390/12/2/297)
7. [GTSRB official benchmark page](https://benchmark.ini.rub.de/gtsrb_news.html)
8. [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)
9. [Ultralytics object detection documentation](https://docs.ultralytics.com/tasks/detect/)
10. [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
11. [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)

