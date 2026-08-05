# GEMINI.md: Antigravity CLI Execution Directives
**Project:** Edge-Optimized YOLO11n Framework for Ceylon Tea Leaf Disease Detection
**Environment:** Windows 11, `uv`, Python 3.11.13, PyTorch 2.5.1+cu121, Ultralytics 8.4.x, OpenVINO
**Hardware Constraints:** NVIDIA RTX 3050 6GB VRAM, 31.6 GB System RAM, 20-thread Intel CPU

---

## 1. CORE ARCHITECTURAL & CODING PRINCIPLES

### A. SOLID Principles Compliance
1. **Single Responsibility Principle (SRP):** Each module must do exactly one job. Separate data parsing (`src/data/`), model management (`src/models/`), pruning (`src/optimization/prune.py`), quantization (`src/optimization/quantize.py`), and evaluation (`src/evaluation/`).
2. **Open/Closed Principle (OCP):** Model pipelines must be extendable via configuration (`configs/config.yaml`) without rewriting core execution scripts.
3. **Liskov Substitution Principle (LSP):** Base evaluator interfaces must allow seamless switching between PyTorch `.pt` models and OpenVINO `.xml/.bin` engines.
4. **Interface Segregation Principle (ISP):** Keep metric loggers concise; do not force inference benchmarks to depend on training configuration objects.
5. **Dependency Inversion Principle (DIP):** Pass parameters via YAML configuration objects or dataclasses rather than hardcoding paths/hyperparameters inside functions.

### B. Defensive Memory Management (OOM Safeguards for 6GB VRAM)
To guarantee zero Out-Of-Memory (OOM) failures on the 6GB VRAM GPU:
* **Automatic Mixed Precision:** Force `amp=True` (FP16 precision) across all PyTorch training steps.
* **Gradient Accumulation:** Hard-cap native GPU `batch=16`. Use `accumulate=4` to achieve an effective batch size of 64 without memory spikes.
* **RAM Offloading:** Leverage the 31.6 GB system RAM by setting `cache=True`, `workers=8`, and `pin_memory=True` in the PyTorch dataloader settings.
* **Garbage Collection & Cache Flushes:** Call `torch.cuda.empty_cache()` and `gc.collect()` explicitly before and after model instantiation, epoch evaluations, and exported engine benchmark runs.
* **Pre-Execution VRAM Check:** Implement a guard clause that checks `torch.cuda.mem_get_info()[0]` before starting any training loop. Raise an error if free VRAM is under 1.5 GB.

### C. Iterative ML Development Flow
Never train a full model from scratch on step 1. Follow this strict iteration cycle:
1. **Phase 0 (Sanity / Smoke Test):** Dry-run pipeline on a 100-image subset for 2 epochs to confirm data flow, forward pass, loss calculation, saving, and memory release.
2. **Phase 1 (Baseline Model):** Train full FP32 baseline on complete dataset. Save checkpoint and evaluate baseline metrics.
3. **Phase 2 (Pruning Iterations):** Apply structured channel pruning iteratively (10% step reductions up to 30%), followed by 20-epoch recovery fine-tuning runs.
4. **Phase 3 (Quantization):** Export the pruned model to OpenVINO FP16 and INT8 using post-training calibration (500 images).
5. **Phase 4 (Benchmarking & EFS):** Benchmark all variants (Baseline FP32, Pruned FP32, Pruned INT8) on CPU and GPU. Calculate the Edge Feasibility Score (EFS).

---

## 2. PROJECT DIRECTORY STRUCTURE

```text
icac_tea_disease_research/
├── configs/
│   └── config.yaml             # Single source of truth for paths, hyperparams, thresholds
├── datasets/                   # Figshare dataset directory (images, labels, data.yaml)
│   └── test/
│   └── train/
│   └── valid/
│   └── data.yaml/
│   └── README.md/            
├── exports/                    # OpenVINO .xml/.bin exported models
├── logs/                       # Training logs, TensorBoard / CSV metrics
├── notebooks/                  # Interactive analysis / validation
├── scripts/
│   ├── verify_gpu.py
│   ├── run_sanity.py           # Phase 0
│   ├── train_baseline.py       # Phase 1
│   ├── prune_model.py          # Phase 2
│   ├── quantize_openvino.py    # Phase 3
│   └── benchmark_efs.py        # Phase 4
├── src/
│   ├── __init__.py
│   ├── data/                   # Dataset utils & verifiers
│   ├── models/                 # Model builders & wrappers
│   ├── optimization/           # Structured pruning & OpenVINO exports
│   ├── evaluation/             # Metrics calculation (mAP, Latency, EFS)
│   └── utils/                  # Memory checkers, logging, helpers
├── weights/                    # PyTorch .pt model checkpoints
└── GEMINI.md
```

## 3. DOMAIN SPECIFICATIONS & DATASET TARGETS

* **Target Crop / Dataset:** *Tea-Leaf Disease and Pest Detection Dataset (v1.0)* (Figshare).
* **Classes (8 total):**
1. `Black rot`
2. `Brown blight`
3. `Leaf rust`
4. `Red Spider infested leaf`
5. `Tea Mosquito bug infested leaf`
6. `healthy Tea leaf`
7. `White spot`
8. `disease` (generalized)


* **Image Input Resolution:** `640x640`
* **Baseline Architecture:** `yolo11n.pt` (COCO pre-trained weights)

---

## 4. PHASED IMPLEMENTATION PLAN

### PHASE 0: Environment & Pipeline Smoke Test

* **Goal:** Verify end-to-end data loading, forward pass, loss calculation, metric logging, and GPU memory release using a dummy dataset subset.
* **Tasks:**
1. Create `configs/config.yaml` containing paths, batch size (16), accumulate (4), workers (8), device (0), and epochs.
2. Implement `src/utils/memory.py` with memory logging and guard clauses (`check_vram()`, `flush_memory()`).
3. Create `scripts/run_sanity.py` that loads 50 train / 20 val images, runs 2 epochs of YOLO11n, and confirms `.pt` file export without memory leakage.



### PHASE 1: Baseline YOLO11n Fine-Tuning

* **Goal:** Train the baseline FP32 model on the complete Tea Disease dataset.
* **Training Settings:**
* Epochs: `150` (patience: `30` early stopping)
* Optimizer: `AdamW` (`lr0=0.001`, cosine learning rate decay)
* Augmentations: Mosaic, MixUp, HSV (H:0.015, S:0.7, V:0.4), Flips, Rotation (±15°)
* Hardware Parameters: `batch=16`, `accumulate=4`, `amp=True`, `cache=True`, `workers=8`, `device=0`


* **Output Artifacts:** `weights/baseline_yolo11n_fp32.pt`, baseline validation metrics (`mAP50`, `mAP50-95`, `GFLOPs`, `Params`, `Model Size MB`).

### PHASE 2: Structured Channel Pruning & Fine-Tuning

* **Goal:** Physically reduce network dimensions by surgically removing redundant convolutional filters in the C3k2 modules based on L1-norm scaling factors.
* **Execution Rule:**
1. Perform iterative structured pruning at 10% parameter reduction steps up to a global target of ~30% parameter reduction (~1.8M params remaining from 2.62M baseline).
2. Apply a 20-epoch recovery fine-tuning phase after each pruning iteration.


* **Output Artifacts:** `weights/pruned_yolo11n_fp32.pt`.

### PHASE 3: OpenVINO Post-Training INT8 Quantization (PTQ)

* **Goal:** Convert the pruned PyTorch model into an OpenVINO INT8 engine for CPU acceleration.
* **Execution Rule:**
1. Export PyTorch weights to OpenVINO format.
2. Run asymmetric calibration using a representative subset of 500 training images to calculate INT8 scale factors and zero-points for layer activations.


* **Output Artifacts:** `exports/pruned_yolo11n_int8.xml` and `.bin`.

### PHASE 4: Benchmarking & Edge Feasibility Score (EFS) Calculation

* **Goal:** Benchmark baseline FP32, pruned FP32, and pruned INT8 variants on CPU (Intel i7) and GPU (RTX 3050). Calculate comparative metrics and the Edge Feasibility Score.
* **Metrics Matrix:**
* Precision, Recall, mAP@0.5, mAP@0.5:0.95
* Parameter Count (M)
* Model Footprint Size (MB)
* Floating Point Operations (GFLOPs)
* Inference Latency (ms per frame) & Frames Per Second (FPS) on CPU vs GPU


* **EFS Formula Implementation:**

$$EFS = \frac{mAP_{0.5} \times FPS_{CPU}}{\log_{10}(Model\_Size_{MB} \times GFLOPs)}$$


* **Output Artifacts:** Summary markdown table and LaTeX table generated directly in `logs/benchmark_results.json` and `logs/benchmark_table.tex` for inclusion in the ICAC 2026 IEEE paper.

---

## 5. INSTRUCTIONS FOR ANTIGRAVITY CLI

1. **Strict Modular Code:** Do not dump all logic into single script files. Use the directory structure in Section 2.
2. **Configuration Driven:** Always load parameters from `configs/config.yaml`.
3. **Safety First:** Wrap all PyTorch model instantiations and training calls in `try...except` blocks with defensive memory cleanup in `finally:` blocks.
4. **Progress Logging:** Print explicit timestamped progress and VRAM utilization logs at the start and end of every epoch or evaluation phase.
5. **No Hardcoded Paths:** Use relative paths relative to the project root directory.
6. **No emoticons:** Do not use any emojis or em-dashes within the code. 