# Edge-Optimized YOLO11n Ceylon Tea Leaf Disease Detection

This repository hosts the research and implementation framework for an edge-optimized YOLO11n model designed to detect tea leaf diseases and pests on constraint-heavy edge hardware.

Developed targeting an **NVIDIA RTX 3050 6GB VRAM** laptop GPU and **13th Gen Intel Core i7 CPU** under strict memory and hardware constraints.

---

## 💻 Environment & Hardware Constraints
* **Operating System:** Windows 11
* **Environment Manager:** `uv` (Python 3.11.13)
* **Frameworks:** PyTorch 2.5.1+cu121, Ultralytics 8.4.115, OpenVINO 2026.3.0, NNCF 3.3.0
* **Hardware Specs:** 6GB VRAM Laptop GPU (RTX 3050), 31.6 GB System RAM, Intel Core i7 CPU (20 threads)

---

## 📂 Project Directory Structure

```text
icac_tea_disease_research/
├── configs/
│   └── config.yaml             # Single source of truth for paths, hyperparams, and device settings
├── datasets/                   # Tea Leaf Disease and Pest Dataset (8 target classes)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml               # Dataset relative path configuration
├── exports/                    # OpenVINO XML/BIN engine exported files
│   ├── pruned_yolo11n_int8.xml
│   └── pruned_yolo11n_int8.bin
├── logs/                       # Comprehensive evaluation logs and metrics
│   ├── benchmark_results.json  # Raw benchmark evaluation metrics
│   ├── benchmark_table.tex     # Manuscript-ready IEEE LaTeX comparative table
│   └── phase3_quantization_metrics.json
├── scripts/
│   ├── verify_gpu.py           # Pre-execution VRAM check script
│   ├── run_sanity.py           # Phase 0: Pipeline smoke test
│   ├── train_baseline.py       # Phase 1: FP32 baseline fine-tuning
│   ├── prune_model.py          # Phase 2: Iterative L1 structured channel pruning
│   ├── quantize_openvino.py    # Phase 3: OpenVINO INT8 calibration & export
│   └── benchmark_efs.py        # Phase 4: CPU/GPU latency & Edge Feasibility Score (EFS)
├── src/
│   ├── data/                   # Dataset loader and subset helpers
│   ├── models/                 # Model builders and wrappers
│   ├── optimization/           # Channel pruning logic and mask applications
│   ├── evaluation/             # Metrics calculation (mAP, Latency, EFS)
│   └── utils/                  # Memory checkers, cache flushes, and loggers
└── weights/                    # PyTorch model checkpoints (.pt)
    ├── baseline_yolo11n_fp32.pt
    └── pruned_yolo11n_fp32.pt
```

---

## 📈 Research Phases & Results Summary

### 🟢 PHASE 0: Environment & Pipeline Smoke Test (COMPLETED)
* **Goal:** Verify end-to-end data loading, forward pass, loss calculation, and VRAM memory release on a 50 train / 20 val image subset.
* **Results:** Clean execution with active VRAM guards and automated memory cleanup.

### 🟢 PHASE 1: Baseline YOLO11n Fine-Tuning (COMPLETED)
* **Goal:** Fine-tune COCO pre-trained `yolo11n.pt` for 150 epochs on the complete Ceylon Tea Disease dataset.
* **Execution:** 150 epochs completed on GPU (**mAP50 = 0.9822**, **mAP50-95 = 0.7302**, 2.591M params).
* **Official Weights:** [baseline_yolo11n_fp32.pt](weights/baseline_yolo11n_fp32.pt)

### 🟢 PHASE 2: Structured Channel Pruning & Fine-Tuning (COMPLETED)
* **Goal:** Surgically remove redundant convolutional filters in C3k2 modules across 8 pruning iterations up to ~30% parameter reduction, followed by 20-epoch recovery fine-tuning phases.
* **Results:** Successfully zeroed out **29.45% of total parameters** (retaining **1.829M parameters** from 2.591M baseline) while maintaining accuracy (**mAP50 = 0.9462**).
* **Official Weights:** [pruned_yolo11n_fp32.pt](weights/pruned_yolo11n_fp32.pt)

### 🟢 PHASE 3: OpenVINO Post-Training INT8 Quantization (COMPLETED)
* **Goal:** Export the pruned PyTorch model into an OpenVINO engine and apply asymmetric INT8 PTQ calibration (784 validation images) using Neural Network Compression Framework (NNCF).
* **Results:** Storage footprint reduced from **5.23 MB** down to **3.25 MB** (**37.71% physical storage reduction**).
* **Exported Engine:** [pruned_yolo11n_int8.xml](exports/pruned_yolo11n_int8.xml) & [bin](exports/pruned_yolo11n_int8.bin)

### 🟢 PHASE 4: Benchmarking & Edge Feasibility Score (COMPLETED)
* **Goal:** Benchmark baseline FP32, pruned FP32, and pruned INT8 variants on CPU (Intel i7) and GPU (RTX 3050), and compute the Edge Feasibility Score (EFS):

$$EFS = \frac{mAP_{0.5} \times FPS_{CPU}}{\log_{10}(Model\_Size_{MB} \times GFLOPs)}$$

---

## 📊 Final Comparative Performance Matrix

| Model Architecture / Variant | Format | Params (M) | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | CPU Lat (ms) | CPU FPS | GPU FPS | EFS Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline FP32** | PyTorch | 2.591 M | 5.23 MB | 0.9822 | 0.7302 | 66.67 ms | 15.00 FPS | 99.04 FPS | **9.66** |
| **Pruned FP32** | PyTorch | 1.829 M | 5.22 MB | 0.9462 | 0.6566 | 74.26 ms | 13.47 FPS | 104.04 FPS | **8.36** |
| **Pruned INT8** | OpenVINO | 1.829 M | **3.25 MB** | **0.9341** | 0.6299 | **13.09 ms** | **76.38 FPS** | N/A | **54.13** |

*Key Findings:* OpenVINO INT8 post-training quantization achieved a **5.09x CPU frame-rate speedup** (from 15.0 FPS to 76.4 FPS) and a **5.60x increase in Edge Feasibility Score (EFS)** compared to the baseline model while maintaining **93.41% mAP50**.

---

## 🛠 Usage Instructions

### 1. Verify VRAM & Sanity Test
```bash
.venv\Scripts\python.exe scripts/run_sanity.py
```

### 2. Fine-Tune Baseline YOLO11n
```bash
.venv\Scripts\python.exe scripts/train_baseline.py
```

### 3. Run Structured Pruning Loop
```bash
.venv\Scripts\python.exe scripts/prune_model.py
```

### 4. Quantize to OpenVINO INT8
```bash
.venv\Scripts\python.exe scripts/quantize_openvino.py
```

### 5. Benchmark & Compute EFS
```bash
.venv\Scripts\python.exe scripts/benchmark_efs.py
```
