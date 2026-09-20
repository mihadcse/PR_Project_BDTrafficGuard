# BD-TrafficGuard — Results Analysis
### Robust Bangladeshi Traffic Sign Detection & Recognition (YOLO11n)

---

## 1. Overview

This notebook trains and stress-tests a YOLO11n detector on a 31-class Bangladeshi road-sign dataset, then benchmarks it against a purpose-built corruption suite (**BD-TrafficSign-C**) covering 8 real-world degradation types × 3 severities. Three models are compared:

| Model | Training regime |
|---|---|
| `YOLO11n_baseline` | Standard Ultralytics defaults, no extra augmentation |
| `YOLO11n_robust_builtin_aug` | Stronger built-in Ultralytics augmentation (rotation, shear, perspective, mixup, wider HSV-V, no horizontal flip) |
| `YOLO11n_robust_explicit_corruptions` | Trained on a dataset explicitly augmented with the same corruption families used in the test benchmark (noise/blur/light/weather) |

All training ran on a single Kaggle Tesla T4 GPU, 50 epochs each, `imgsz=640`, `batch=16`, `seed=42`.

---

## 2. Dataset

**Source:** Roboflow-hosted "BD Road Traffic Sign" dataset (31 classes), pre-split into train/valid/test.

| Split | Images | Annotated objects |
|---|---|---|
| Train | 7,165 | 7,207 |
| Valid | 2,046 | 2,064 |
| Test  | 1,024 | 1,025 |
| **Total** | **10,235** | **10,296** |

**Image geometry:** every image is a fixed 640×640, so there is no native resolution variability to exploit — this matters for the resolution ablation in §7.

**Object scale:** mostly medium/large signs — Small (<1% of image area) 9.4%, Medium (1–5%) 42.3%, Large (>5%) 48.3%. Average ~1.0 object per image (max 3), so this is closer to a single/few-object detection task than a dense-scene one.

**Class balance — the dataset's biggest weakness:**
- Most frequent: *Side Road On Right* (1,109 objects, 10.8%), *Height Limit 5.7m* (950, 9.2%), *Side Road On Left* (853, 8.3%).
- Long tail: *Hospital Ahead* (14), *Traffic Merges From Right* (22), *Junction Ahead* (48), *Tolls Ahead* (56).
- **`Weight Limit 10T` and `Weight Limit 27T` have zero instances in train, valid, and test.** Two of the 31 declared classes are never observed anywhere in the data. The model cannot learn them, and any per-class metric for these classes is undefined (`NaN`) — this shows up later in the class-wise AP tables. This is a labeling/export issue in the source dataset, not a modeling issue, but it means the "31-class" framing is really a 29-class problem in practice.

---

## 3. Clean Test-Set Results (official test split, 1,024 images / 1,025 instances)

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Baseline | 0.9866 | 0.9781 | 0.9823 | 0.9895 | **0.9697** |
| Robust (built-in aug) | 0.9876 | 0.9804 | 0.9840 | 0.9914 | **0.9139** |
| Robust (explicit corruptions) | 0.9859 | 0.9838 | 0.9849 | 0.9899 | **0.9668** |

All three models are near-saturated on precision/recall/mAP@0.5 on clean data — the task is close to solved at the coarse (IoU 0.5) level. The interesting signal is in **mAP@0.5:0.95**, which is far more sensitive to bounding-box localization quality:

- The **built-in-augmentation** model loses over 5.5 points of clean mAP@0.5:0.95 (0.9697 → 0.9139) relative to the baseline. Heavier geometric augmentation (rotation, shear, perspective, mixup) improved classification/coarse localization slightly but visibly hurt tight box regression on clean images — a real trade-off, not noise.
- The **explicit-corruption** model keeps clean performance essentially intact (0.9668 vs. 0.9697 baseline), while also (see §5) becoming dramatically more robust — this is the standout result of the whole study.

---

## 4. Corruption Benchmark — BD-TrafficSign-C

A corrupted evaluation set was generated from the 1,024 test images: **8 corruption types × 3 severities = 24 conditions × 1,024 images = 24,576 corrupted images** (generation took 8.62 min). Corruption types: Gaussian noise, Gaussian blur, motion blur, JPEG compression, low light, low contrast, fog, rain.

### 4.1 Baseline model robustness, by corruption type (mean over 3 severities, mAP@0.5:0.95)

| Corruption | Mean mAP@0.5:0.95 | Worst-severity mAP@0.5:0.95 | Mean drop % | Worst drop % |
|---|---|---|---|---|
| Gaussian noise | 0.7540 | **0.4324** | **22.25%** | **55.41%** |
| Motion blur | 0.9029 | 0.7951 | 6.89% | 18.01% |
| Low light | 0.9521 | 0.9219 | 1.81% | 4.92% |
| JPEG compression | 0.9659 | 0.9562 | 0.40% | 1.39% |
| Low contrast | 0.9662 | 0.9574 | 0.36% | 1.27% |
| Gaussian blur | 0.9682 | 0.9616 | 0.15% | 0.83% |
| Rain | 0.9699 | 0.9695 | −0.02% | 0.02% |
| Fog | 0.9715 | 0.9711 | −0.19% | −0.15% |

**Gaussian noise is by far the dominant failure mode** — it drops mAP@0.5:0.95 by more than half at severe strength, an order of magnitude worse than any other corruption. Motion blur is a distant second concern. Fog, rain, gaussian blur, JPEG compression and low contrast are essentially non-issues for the baseline model even at severe strength (all under 1.3% drop) — the sign shapes/colors are large and high-contrast enough to survive these degradations, but not sensor noise.

### 4.2 Baseline model robustness, by severity

| Severity | Mean mAP@0.5:0.95 | Mean drop % | Worst-case mAP@0.5:0.95 |
|---|---|---|---|
| Mild | 0.9703 | −0.06% (no real drop) | 0.9685 |
| Moderate | 0.9529 | 1.73% | 0.8610 |
| Severe | 0.8708 | 10.20% | **0.4324** |

### 4.3 Worst individual conditions (baseline)

1. Gaussian noise / severe — mAP@0.5:0.95 = 0.4324 (−55.4%)
2. Motion blur / severe — 0.7951 (−18.0%)
3. Gaussian noise / moderate — 0.8610 (−11.2%)
4. Low light / severe — 0.9219 (−4.9%)
5. Motion blur / moderate — 0.9438 (−2.7%)

---

## 5. Model Comparison — Does the Robust Training Actually Help?

The notebook's own evaluation criterion is the right one to apply here: *"A robust model is better only if its mean corrupted mAP@0.5:0.95 is higher than the clean-trained baseline and/or its robustness drop is lower."*

| Model | Clean mAP50:95 | Mean Corrupted mAP50:95 | Worst Corrupted mAP50:95 | Mean Drop | Mean Drop % | Gain vs. Baseline |
|---|---|---|---|---|---|---|
| Baseline | 0.9697 | 0.9313 | 0.4324 | 0.0383 | 3.95% | 0.0000 |
| Robust — built-in aug | 0.9139 | 0.8699 | 0.4843 | 0.0439 | 4.81% | **−0.0614** |
| Robust — explicit corruptions | 0.9668 | **0.9646** | **0.9481** | **0.0022** | **0.23%** | **+0.0332** |

**Findings:**

- **Built-in Ultralytics augmentation alone does not buy robustness here — it slightly hurts.** It has both a lower clean score *and* a lower mean corrupted mAP@0.5:0.95 than the plain baseline (0.8699 vs 0.9313). Its worst-case score (0.4843) barely improves on the baseline's worst case (0.4324), so it does not fix the Gaussian-noise failure mode either. This is a useful negative result: generic geometric/color augmentation is not a substitute for degradation-specific augmentation when the target failure mode (sensor noise) isn't well represented in that augmentation policy.
- **Explicit-corruption training is a clear, decisive win.** It nearly eliminates the robustness gap: mean corrupted mAP@0.5:0.95 rises from 0.9313 → 0.9646, the mean drop collapses from 3.95% → 0.23%, and — most importantly — the **worst-case score jumps from 0.4324 to 0.9481**, i.e. the catastrophic Gaussian-noise failure mode is essentially fixed. It does all this while keeping clean-data performance within noise of the baseline (0.9668 vs 0.9697).
- **Takeaway for the project's core hypothesis:** degradation-aware training only works when the augmentation actually resembles the corruption distribution you'll be evaluated/deployed on. Training with the explicit corruption pipeline (matched to the BD-TrafficSign-C benchmark) is the method that should be recommended/deployed; built-in augmentation should not be presented as a robustness solution on its own.

---

## 6. Class-wise Analysis

Per-class AP@0.5 / AP@0.5:0.95 was computed for all three models on the clean test set. Two structural caveats limit this analysis:

- **`Weight Limit 10T`** and **`Weight Limit 27T`** (class IDs 29, 30) have **zero instances anywhere in the dataset** — AP is undefined (`NaN`) for every model. These two classes should be dropped from the class list or the dataset re-collected/re-annotated before any per-class claim is made about "31-class" coverage.
- A small number of other rare classes (`Underpass Ahead` showed `NaN` for the explicit-robust model's classwise table despite having 180 total instances/24 in test) suggest the classwise-AP computation is sensitive to how detections are binned per model/run — worth double-checking the classwise export code before quoting these numbers in a paper.

For classes with enough support, per-class AP@0.5 is ≥0.94 across the board on clean data, and even the weakest classes (`Hospital Ahead`: 3 test instances, AP50=0.913; `Mosque Ahead`: R=0.876, AP50-95=0.889) are driven by **support size**, not corruption sensitivity — i.e., the main risk to class-wise reliability is data scarcity, not the robustness intervention.

---

## 7. Deployment Metrics (speed / size)

| Model | Size (MB) | Avg inference (ms/img) | FPS (Tesla T4) |
|---|---|---|---|
| Baseline | 5.23 | 16.56 | 60.4 |
| Robust — built-in aug | 5.23 | 16.15 | 61.9 |
| Robust — explicit corruptions | 5.23 | 16.35 | 61.1 |

All three checkpoints are the identical YOLO11n architecture (2.59M params, 6.4 GFLOPs), so model size and inference speed are unaffected by the training regime — **the robustness gain from explicit-corruption training is essentially free**: same footprint, same latency, dramatically better worst-case accuracy. This is a strong practical argument for deployment.

---

## 8. Resolution Ablation

Evaluated the (baseline) model at three inference resolutions:

| imgsz | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| 416 | 0.9892 | 0.9838 | 0.9865 | 0.9899 | 0.9642 |
| 640 | 0.9866 | 0.9781 | 0.9823 | 0.9895 | 0.9697 |
| 960 | 0.9789 | 0.9673 | 0.9730 | 0.9743 | **0.8992** |

Since every source image is natively 640×640, upscaling to 960 adds no real information — it only interpolates — and performance degrades noticeably at that resolution (mAP@0.5:0.95 falls ~7 points). Downscaling to 416 actually holds up very well (best precision/recall/F1 of the three) while presumably running faster. **Recommendation: 640 (or even 416 for a speed/accuracy trade-off) is the right inference resolution for this dataset; there's no benefit to running at 960.**

---

## 9. Components Not Executed in This Run

The notebook has flags for several optional experiments that were **skipped** in this pass (their corresponding cells printed "Skipped..." with no output):

- Crop-dataset construction for a two-stage pipeline
- EfficientNet-B0 second-stage classifier training
- Faster R-CNN baseline comparison

These would need `RUN_CROP_DATASET_BUILD=True` (and equivalents) set before re-running to get comparative numbers against YOLO11n.

---

## 10. Summary & Recommendations

1. **Best model:** `YOLO11n_robust_explicit_corruptions`. It matches the baseline on clean data (mAP@0.5:0.95 0.9668 vs 0.9697) while cutting the mean robustness drop from 3.95% to 0.23% and raising the worst-case score from 0.43 to 0.95 — at no cost in model size or inference speed. This is the checkpoint to ship.
2. **Built-in augmentation is not sufficient** as a robustness strategy for this problem; it costs clean accuracy without buying corrupted-condition robustness. Don't market/report it as a robust model.
3. **Gaussian sensor noise is the single biggest real-world risk** for the baseline model (up to −55% mAP@0.5:0.95), far ahead of blur, weather, or compression artifacts. Any deployment plan (e.g., dashcam capture in low-light or older sensors) should prioritize denoising or noise-augmented training over other degradation defenses.
4. **Fix the dataset before publishing class-level claims:** two classes have zero examples and cannot be evaluated; several other classes have single-digit test support, so class-wise AP for those should be reported with explicit sample-size caveats, not as flat percentages.
5. **Inference resolution:** stay at 640 (native) or drop to 416 for speed; 960 actively hurts accuracy on this dataset.
6. To complete the study as originally scoped, re-run with the two-stage detector+classifier and Faster R-CNN flags enabled to get the comparative baselines referenced in the notebook's markdown plan but not yet executed.

---

*Source: analysis of `bd-trafficguard-complete-implementation.ipynb` (41 cells, Kaggle/Tesla T4 run, Ultralytics 8.4.147, YOLO11n). All figures above are taken directly from the notebook's own printed outputs and exported summary tables (`model_comparison_summary.csv`, `baseline_robustness_df`, `classwise` tables, `deployment_df`, `resolution_ablation_df`).*
