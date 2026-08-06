import json
import logging
import sys
import time
from pathlib import Path
import shutil
import yaml
from ultralytics import YOLO

# Add root directory to path for internal modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.memory import check_vram, flush_memory, log_vram_status

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("quantize_openvino")


def run_phase3_quantization() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 3: OPENVINO POST-TRAINING INT8 QUANTIZATION (PTQ)")
    logger.info("=" * 70)

    # 1. Load config
    config_path = Path("configs/config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    paths = config["paths"]
    hw = config["hardware"]

    # File checks
    dataset_yaml = Path(paths["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml.resolve()}")

    pruned_weights = Path(paths["weights_dir"]) / "pruned_yolo11n_fp32.pt"
    if not pruned_weights.is_file():
        raise FileNotFoundError(f"Pruned Phase II model weights not found: {pruned_weights.resolve()}")

    exports_dir = Path(paths["exports_dir"])
    exports_dir.mkdir(parents=True, exist_ok=True)

    # VRAM check
    check_vram(min_free_gb=hw.get("vram_min_free_gb", 1.5))
    log_vram_status(tag="pre-export")

    logger.info("Loading pruned PyTorch model: %s", pruned_weights)
    model = YOLO(str(pruned_weights))

    logger.info("Exporting to OpenVINO INT8 format with calibration data from: %s", dataset_yaml)
    
    # Export OpenVINO INT8
    # Ultralytics openvino export creates a directory named `pruned_yolo11n_fp32_openvino_model`
    export_path_str = model.export(
        format="openvino",
        int8=True,
        data=str(dataset_yaml),
        half=False,
    )
    
    exported_dir = Path(export_path_str)
    logger.info("Export completed. Output saved at: %s", exported_dir)

    # Copy / move generated .xml and .bin engine files to exports/ directory
    xml_files = list(exported_dir.glob("*.xml"))
    bin_files = list(exported_dir.glob("*.bin"))

    target_xml = exports_dir / "pruned_yolo11n_int8.xml"
    target_bin = exports_dir / "pruned_yolo11n_int8.bin"

    if xml_files:
        shutil.copy(xml_files[0], target_xml)
        logger.info("Saved OpenVINO XML engine definition: %s", target_xml)
    if bin_files:
        shutil.copy(bin_files[0], target_bin)
        logger.info("Saved OpenVINO BIN weight file: %s", target_bin)

    # Measure file sizes
    pytorch_size_mb = pruned_weights.stat().st_size / (1024 * 1024)
    
    openvino_int8_size_bytes = 0
    if target_xml.is_file():
        openvino_int8_size_bytes += target_xml.stat().st_size
    if target_bin.is_file():
        openvino_int8_size_bytes += target_bin.stat().st_size
    
    openvino_int8_size_mb = openvino_int8_size_bytes / (1024 * 1024)

    # If the exported dir contains everything, we also calculate the total folder size
    folder_size_mb = sum(f.stat().st_size for f in exported_dir.rglob('*') if f.is_file()) / (1024 * 1024)

    reduction_pct = ((pytorch_size_mb - openvino_int8_size_mb) / pytorch_size_mb) * 100

    # Save summary metrics
    summary = {
        "model": "pruned_yolo11n_int8",
        "pytorch_fp32_size_mb": round(pytorch_size_mb, 2),
        "openvino_int8_size_mb": round(openvino_int8_size_mb, 2),
        "openvino_folder_size_mb": round(folder_size_mb, 2),
        "size_reduction_pct": round(reduction_pct, 2),
        "exported_xml": str(target_xml),
        "exported_bin": str(target_bin),
    }

    metrics_log_path = Path(paths["logs_dir"]) / "phase3_quantization_metrics.json"
    with open(metrics_log_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved quantization metrics to: %s", metrics_log_path)
    
    flush_memory(device=hw["device"])
    log_vram_status(tag="post-cleanup")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PHASE 3 COMPLETE -- Elapsed: %.1f seconds", elapsed)
    logger.info("=" * 70)

    # Print explicit Terminal summary table
    print("\n" + "=" * 70)
    print("PHASE 3 QUANTIZATION SUMMARY: PYTORCH FP32 vs OPENVINO INT8")
    print("=" * 70)
    print(f"Original PyTorch FP32 Model Size : {pytorch_size_mb:.2f} MB ({pruned_weights})")
    print(f"Exported OpenVINO INT8 Engine Size: {openvino_int8_size_mb:.2f} MB ({target_xml.name} + {target_bin.name})")
    print(f"Total Physical Storage Reduction : {reduction_pct:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        run_phase3_quantization()
    except Exception:
        logger.exception("Phase 3 quantization failed.")
        sys.exit(1)
