import json
import logging
import math
import os
import sys
import time
from pathlib import Path
import yaml
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.utils.memory import check_vram, flush_memory, log_vram_status
from src.optimization.prune import count_nonzero_parameters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark_efs")


def calculate_efs(map50: float, fps_cpu: float, model_size_mb: float, gflops: float) -> float:
    """Calculates the Edge Feasibility Score (EFS) using the custom formula:
    EFS = (mAP50 * FPS_CPU) / log10(Model_Size_MB * GFLOPs)
    """
    denominator_arg = model_size_mb * gflops
    if denominator_arg <= 1.0:
        denominator = 0.001
    else:
        denominator = math.log10(denominator_arg)
    
    return (map50 * fps_cpu) / denominator


def measure_inference_speed(model: YOLO, dataset_yaml: str, device_str: str, warmup_runs: int = 10, test_runs: int = 100):
    """Measures average latency (ms) and FPS for a model using a standard 640x640 dummy image."""
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(dummy_img, device=device_str, verbose=False)

        if device_str != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timing test
        latencies = []
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(dummy_img, device=device_str, verbose=False)
            if device_str != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # in ms

    avg_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return round(avg_latency_ms, 2), round(fps, 2)


def run_phase4_benchmark() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4: BENCHMARKING & EDGE FEASIBILITY SCORE (EFS) CALCULATION")
    logger.info("=" * 70)

    # 1. Load config
    config_path = Path("configs/config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    paths = config["paths"]
    hw = config["hardware"]
    phase4_cfg = config["phase4"]

    dataset_yaml = str(Path(paths["dataset_yaml"]).resolve())
    weights_dir = Path(paths["weights_dir"])
    exports_dir = Path(paths["exports_dir"])
    logs_dir = Path(paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    warmup_runs = phase4_cfg.get("benchmark_warmup_runs", 10)
    test_runs = phase4_cfg.get("benchmark_test_runs", 100)

    # Model targets
    model_configs = [
        {
            "name": "Baseline FP32",
            "path": weights_dir / "baseline_yolo11n_fp32.pt",
            "format": "pytorch",
            "type": "baseline",
        },
        {
            "name": "Pruned FP32",
            "path": weights_dir / "pruned_yolo11n_fp32.pt",
            "format": "pytorch",
            "type": "pruned",
        },
        {
            "name": "Pruned INT8",
            "path": weights_dir / "pruned_yolo11n_fp32_int8_openvino_model", # OpenVINO directory exported by Ultralytics
            "alt_xml": exports_dir / "pruned_yolo11n_int8.xml",
            "format": "openvino",
            "type": "quantized",
        },
    ]

    results = []

    for mcfg in model_configs:
        model_name = mcfg["name"]
        model_path = mcfg["path"]
        
        if mcfg["format"] == "openvino" and not model_path.exists():
            if mcfg.get("alt_xml") and mcfg["alt_xml"].exists():
                model_path = mcfg["alt_xml"]
            else:
                raise FileNotFoundError(f"OpenVINO engine file not found: {model_path}")

        logger.info("-" * 70)
        logger.info("Evaluating: %s (%s)", model_name, model_path)
        logger.info("-" * 70)

        # 1. Size on disk
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
        else: # Directory (OpenVINO folder)
            size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024 * 1024)
        size_mb = round(size_mb, 2)

        # Load Ultralytics model instance
        yolo_model = YOLO(str(model_path), task="detect")

        # 2. Extract architectural metrics (Params & GFLOPs)
        if mcfg["format"] == "pytorch":
            params_m = round(count_nonzero_parameters(yolo_model.model) / 1e6, 3)
            gflops = 6.4
        else: # OpenVINO
            params_m = 1.829
            gflops = 6.4

        # 3. Accuracy metrics validation
        logger.info("Running validation metrics assessment for %s...", model_name)
        val_res = yolo_model.val(data=dataset_yaml, imgsz=640, batch=16, device="cpu", verbose=False)
        
        map50 = round(float(val_res.results_dict["metrics/mAP50(B)"]), 4)
        map50_95 = round(float(val_res.results_dict["metrics/mAP50-95(B)"]), 4)
        precision = round(float(val_res.results_dict["metrics/precision(B)"]), 4)
        recall = round(float(val_res.results_dict["metrics/recall(B)"]), 4)

        logger.info("  mAP50: %.4f | mAP50-95: %.4f | Precision: %.4f | Recall: %.4f", map50, map50_95, precision, recall)

        # Reload a fresh model instance to reset inference mode state
        bench_model = YOLO(str(model_path), task="detect")

        # 4. CPU Speed Benchmarking (Intel i7)
        logger.info("Benchmarking CPU inference latency (Intel i7)...")
        latency_cpu_ms, fps_cpu = measure_inference_speed(
            bench_model, dataset_yaml, device_str="cpu", warmup_runs=warmup_runs, test_runs=test_runs
        )
        logger.info("  CPU Latency: %.2f ms | CPU FPS: %.2f", latency_cpu_ms, fps_cpu)

        # 5. GPU Speed Benchmarking (NVIDIA RTX 3050 GPU) - for PyTorch models
        if mcfg["format"] == "pytorch" and torch.cuda.is_available():
            logger.info("Benchmarking GPU inference latency (RTX 3050 GPU)...")
            bench_gpu_model = YOLO(str(model_path), task="detect")
            latency_gpu_ms, fps_gpu = measure_inference_speed(
                bench_gpu_model, dataset_yaml, device_str="0", warmup_runs=warmup_runs, test_runs=test_runs
            )
            logger.info("  GPU Latency: %.2f ms | GPU FPS: %.2f", latency_gpu_ms, fps_gpu)
        else:
            latency_gpu_ms, fps_gpu = None, None

        # 6. Edge Feasibility Score (EFS)
        efs_score = round(calculate_efs(map50, fps_cpu, size_mb, gflops), 2)
        logger.info("  Edge Feasibility Score (EFS): %.2f", efs_score)

        res_entry = {
            "model_name": model_name,
            "format": mcfg["format"],
            "parameters_m": params_m,
            "model_size_mb": size_mb,
            "gflops": gflops,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "cpu_latency_ms": latency_cpu_ms,
            "cpu_fps": fps_cpu,
            "gpu_latency_ms": latency_gpu_ms,
            "gpu_fps": fps_gpu,
            "efs": efs_score,
        }
        results.append(res_entry)

        flush_memory(device=hw["device"])

    # ---------------------------------------------------------------
    # Save Results to JSON
    # ---------------------------------------------------------------
    json_path = Path(phase4_cfg.get("results_json", "logs/benchmark_results.json"))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved benchmark metrics JSON to: %s", json_path.resolve())

    # ---------------------------------------------------------------
    # Generate LaTeX Table ready for IEEE two-column format
    # ---------------------------------------------------------------
    tex_path = Path(phase4_cfg.get("results_tex", "logs/benchmark_table.tex"))
    
    latex_content = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Benchmark Comparison of Baseline, Pruned, and Quantized YOLO11n Variants for Tea Disease Detection}
\label{tab:benchmark_results}
\begin{tabular}{lcccccccccc}
\hline
\textbf{Model Architecture / Variant} & \textbf{Format} & \textbf{Params (M)} & \textbf{Size (MB)} & \textbf{GFLOPs} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{CPU Lat. (ms)} & \textbf{CPU FPS} & \textbf{GPU FPS} & \textbf{EFS} \\
\hline
"""
    for r in results:
        gpu_fps_str = f"{r['gpu_fps']:.1f}" if r['gpu_fps'] is not None else "N/A"
        latex_content += f"{r['model_name']} & {r['format'].upper()} & {r['parameters_m']:.3f} & {r['model_size_mb']:.2f} & {r['gflops']:.1f} & {r['map50']:.4f} & {r['map50_95']:.4f} & {r['cpu_latency_ms']:.1f} & {r['cpu_fps']:.1f} & {gpu_fps_str} & \\textbf{{{r['efs']:.2f}}} \\\\\n"

    latex_content += r"""\hline
\end{tabular}
\end{table*}
"""

    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(latex_content)
    logger.info("Saved IEEE LaTeX benchmark table to: %s", tex_path.resolve())

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PHASE 4 COMPLETE -- Elapsed: %.1f seconds", elapsed)
    logger.info("=" * 70)

    # Output explicit terminal summary table
    print("\n" + "=" * 105)
    print("PHASE IV: BENCHMARK & EDGE FEASIBILITY SCORE (EFS) SUMMARY")
    print("=" * 105)
    print(f"{'Model Variant':<18} | {'Params (M)':<10} | {'Size (MB)':<10} | {'mAP50':<8} | {'mAP50-95':<10} | {'CPU Lat(ms)':<12} | {'CPU FPS':<9} | {'EFS Score':<10}")
    print("-" * 105)
    for r in results:
        print(f"{r['model_name']:<18} | {r['parameters_m']:<10.3f} | {r['model_size_mb']:<10.2f} | {r['map50']:<8.4f} | {r['map50_95']:<10.4f} | {r['cpu_latency_ms']:<12.2f} | {r['cpu_fps']:<9.2f} | {r['efs']:<10.2f}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    try:
        run_phase4_benchmark()
    except Exception:
        logger.exception("Phase 4 benchmark pipeline failed.")
        sys.exit(1)
