# Edge-Optimized YOLO11n Ceylon Tea Leaf Disease Detection

This repository hosts the research and implementation framework for an edge-optimized YOLO11n model designed to detect tea leaf diseases and pests on constraint-heavy hardware. 

Developed targeting an **NVIDIA RTX 3050 6GB VRAM** laptop GPU under strict memory constraints.

---

## 💻 Environment & Constraints
*   **Operating System:** Windows 11
*   **Environment Manager:** `uv` (Python 3.11.13)
*   **Frameworks:** PyTorch 2.5.1+cu121, Ultralytics 8.4.115, OpenVINO
*   **Hardware Specs:** 6GB VRAM Laptop GPU, 31.6 GB System RAM, 20-thread CPU

---

## 📂 Project Directory Structure

```text
icac_tea_disease_research/
├── configs/
│   └── config.yaml             # Single source of truth for paths, hyperparameters, thresholds
├── datasets/                   # Tea Leaf Disease and Pest Dataset (8 classes)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml               # Relative dataset path config (Ultralytics spec)
├── exports/                    # Target directory for OpenVINO engine exports
├── logs/                       # Log files and evaluation metrics
├── notebooks/                  # Interactive notebooks
├── scripts/
│   ├── verify_gpu.py           # VRAM check script
│   ├── run_sanity.py           # Phase 0 validation (smoke test)
│   └── train_baseline.py        # Phase 1 training execution
├── src/
│   ├── data/                   # Data processing and subset creation utilities
│   ├── models/                 # Model instantiation and custom modules
│   ├── optimization/           # Pruning & quantization tools
│   ├── evaluation/             # Metrics & Edge Feasibility Score (EFS) calculation
│   └── utils/                  # Memory checkers, caching, and logging setup
└── weights/                    # PyTorch check-point (.pt) weights
```

---

## 📈 Phased Progress Status

### 🟢 PHASE 0: Environment & Pipeline Smoke Test (PASSED)
*   **Goal:** Verify data flow, model execution, logging, and memory release on a 50 train / 20 val subset for 2 epochs.
*   **Results:** Verified successful config parsing, VRAM guard execution, and clean memory cache clearing. 
*   **Artifacts:** Checked in temporary weights at `runs/detect/logs/phase0_sanity/weights/best.pt`.

### 🟢 PHASE 1: Baseline YOLO11n Fine-Tuning (COMPLETED)
*   **Goal:** Fine-tune COCO pre-trained `yolo11n.pt` for 150 epochs on the complete dataset (7,260 train images).
*   **Execution Time:** 3.81 hours (150 epochs completed on GPU).
*   **Safety Constraints Met:** Training locked stable at **2.98 GB VRAM** (caching set to `"disk"` and `workers=4` to avoid `MemoryError` issues on system RAM).
*   **Official Weights:** [baseline_yolo11n_fp32.pt](weights/baseline_yolo11n_fp32.pt)
*   **Baseline Metrics Summary:**

| Metric | Value | Context |
|---|---|---|
| **Parameters** | 2.584 M | YOLO11n compact size |
| **Model Size** | 5.23 MB | Under 6 MB footprint |
| **Complexity** | 6.4 GFLOPs | Highly optimized operations |
| **mAP@0.5** | **0.9826** | Excellent class matching accuracy |
| **mAP@0.5:0.95** | **0.7306** | Highly precise bounding-box overlaps |
| **Precision** | 0.9557 | Extremely low false positive rates |
| **Recall** | 0.9588 | Very low false negatives (disease miss rate) |

### 🟡 PHASE 2: Structured Channel Pruning (PENDING)
*   **Goal:** Physically reduce network convolutional filters in C3k2 modules by ~30% (~1.8M params remaining) and run 20-epoch recovery fine-tuning runs.

### 🟡 PHASE 3: OpenVINO Post-Training INT8 Quantization (PENDING)
*   **Goal:** Convert pruned model to OpenVINO INT8 using representative dataset calibration (500 images) for CPU deployment.

### 🟡 PHASE 4: Benchmarking & Edge Feasibility Score (EFS) (PENDING)
*   **Goal:** Generate comparative CPU vs GPU benchmarking metrics and calculate the Edge Feasibility Score:
$$EFS = \frac{mAP_{0.5} \times FPS_{CPU}}{\log_{10}(Model\_Size_{MB} \times GFLOPs)}$$

---

## 🛠 Usage Instructions

### 1. Verification & Sanity Test
```bash
.venv\Scripts\python.exe scripts\run_sanity.py
```
*Loads config, runs VRAM checks, and executes a 2-epoch subset smoke test.*

### 2. Fine-Tuning Baseline
```bash
.venv\Scripts\python.exe scripts\train_baseline.py
```
*Trains the model for 150 epochs using configs in [config.yaml](configs/config.yaml).*
