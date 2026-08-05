"""
scripts/train_baseline.py -- Phase 1: Baseline YOLO11n Fine-Tuning.

Trains the full FP32 baseline on the complete Tea Disease dataset using
COCO-pretrained yolo11n.pt weights. Respects the 6GB VRAM budget through
AMP, batch capping, and defensive memory management.

Output Artifacts:
    - weights/baseline_yolo11n_fp32.pt
    - Baseline validation metrics (mAP50, mAP50-95, GFLOPs, Params, Model Size)
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# -- Resolve project root so imports work regardless of CWD --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.utils.logging_setup import setup_logging
from src.utils.memory import check_vram, flush_memory, log_vram_status, log_system_ram

logger = logging.getLogger(__name__)


def train_baseline() -> None:
    """Execute the Phase 1 baseline fine-tuning pipeline."""

    # ---------------------------------------------------------------
    # Step 1: Load configuration
    # ---------------------------------------------------------------
    config = load_config("configs/config.yaml")
    hw = config["hardware"]
    p1 = config["phase1"]
    aug = p1["augmentation"]
    dataset_cfg = config["dataset"]
    paths = config["paths"]

    # ---------------------------------------------------------------
    # Step 2: Setup logging
    # ---------------------------------------------------------------
    setup_logging(
        log_dir=paths["logs_dir"],
        log_filename="phase1_baseline.log",
        project_root=str(PROJECT_ROOT),
    )

    logger.info("=" * 70)
    logger.info("PHASE 1: BASELINE YOLO11n FINE-TUNING")
    logger.info("=" * 70)
    start_time = time.time()

    # ---------------------------------------------------------------
    # Step 3: Pre-training diagnostics
    # ---------------------------------------------------------------
    log_system_ram()
    log_vram_status(tag="pre-training")
    check_vram(min_free_gb=hw["vram_min_free_gb"])
    flush_memory(device=hw["device"])

    # ---------------------------------------------------------------
    # Step 4: Verify dataset
    # ---------------------------------------------------------------
    dataset_yaml = Path(paths["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml.resolve()}"
        )
    logger.info("Dataset YAML: %s", dataset_yaml.resolve())

    # Verify image directories exist
    for split_name, split_path in [
        ("train", paths["train_images"]),
        ("valid", paths["valid_images"]),
    ]:
        split_dir = Path(split_path)
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_name} images directory not found: {split_dir.resolve()}"
            )
        img_count = len(list(split_dir.iterdir()))
        logger.info("  %s: %d images in %s", split_name, img_count, split_dir)

    # ---------------------------------------------------------------
    # Step 5: Load model and train
    # ---------------------------------------------------------------
    logger.info("Loading YOLO11n from: %s", paths["base_model"])
    log_vram_status(tag="pre-model-load")

    from ultralytics import YOLO

    model = None
    try:
        model = YOLO(paths["base_model"])
        log_vram_status(tag="post-model-load")

        logger.info("Training configuration:")
        logger.info("  Epochs: %d (patience: %d)", p1["epochs"], p1["patience"])
        logger.info("  Optimizer: %s (lr0: %s, lrf: %s)", p1["optimizer"], p1["lr0"], p1["lrf"])
        # nbs (nominal batch size) controls gradient accumulation:
        # accumulate_steps = nbs // batch = 64 // 16 = 4
        nbs = hw["batch_size"] * hw["accumulate"]
        logger.info("  Batch: %d, NBS: %d (accumulate: %d), AMP: %s", hw["batch_size"], nbs, hw["accumulate"], hw["amp"])
        logger.info("  Image size: %d", dataset_cfg["image_size"])
        logger.info("  Augmentations: mosaic=%.1f, mixup=%.2f, degrees=%.1f", aug["mosaic"], aug["mixup"], aug["degrees"])
        logger.info("  HSV: H=%.3f, S=%.1f, V=%.1f", aug["hsv_h"], aug["hsv_s"], aug["hsv_v"])
        logger.info("  Workers: %d, Cache: %s", hw["workers"], hw["cache"])

        results = model.train(
            data=str(dataset_yaml),
            epochs=p1["epochs"],
            patience=p1["patience"],
            optimizer=p1["optimizer"],
            lr0=p1["lr0"],
            lrf=p1["lrf"],
            cos_lr=p1["cos_lr"],
            warmup_epochs=p1["warmup_epochs"],
            warmup_momentum=p1["warmup_momentum"],
            warmup_bias_lr=p1["warmup_bias_lr"],
            batch=hw["batch_size"],
            nbs=nbs,
            imgsz=dataset_cfg["image_size"],
            device=hw["device"],
            amp=hw["amp"],
            workers=hw["workers"],
            cache=hw["cache"],
            # -- Augmentation --
            mosaic=aug["mosaic"],
            mixup=aug["mixup"],
            hsv_h=aug["hsv_h"],
            hsv_s=aug["hsv_s"],
            hsv_v=aug["hsv_v"],
            flipud=aug["flipud"],
            fliplr=aug["fliplr"],
            degrees=aug["degrees"],
            # -- Output --
            project=p1["project"],
            name=p1["name"],
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=-1,
            plots=True,
        )

        log_vram_status(tag="post-training")

        # -----------------------------------------------------------
        # Step 6: Verify and copy checkpoint
        # -----------------------------------------------------------
        train_dir = Path(model.trainer.save_dir)
        best_pt = train_dir / "weights" / "best.pt"
        last_pt = train_dir / "weights" / "last.pt"
        logger.info("Trainer save directory: %s", train_dir)

        if not (best_pt.is_file() or last_pt.is_file()):
            raise FileNotFoundError(
                f"No checkpoint found in {train_dir / 'weights'}. "
                "Training may have failed."
            )

        # Copy best checkpoint to the canonical weights location
        source_pt = best_pt if best_pt.is_file() else last_pt
        dest_pt = Path(paths["weights_dir"]) / "baseline_yolo11n_fp32.pt"
        dest_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_pt), str(dest_pt))

        size_mb = dest_pt.stat().st_size / (1024 * 1024)
        logger.info(
            "Baseline checkpoint saved: %s (%.2f MB)",
            dest_pt,
            size_mb,
        )

        # -----------------------------------------------------------
        # Step 7: Final validation and metrics summary
        # -----------------------------------------------------------
        logger.info("Running final validation on the full validation set...")
        flush_memory(device=hw["device"])
        log_vram_status(tag="pre-final-val")

        val_model = YOLO(str(dest_pt))
        val_results = val_model.val(
            data=str(dataset_yaml),
            imgsz=dataset_cfg["image_size"],
            device=hw["device"],
            batch=hw["batch_size"],
            workers=hw["workers"],
        )

        log_vram_status(tag="post-final-val")

        # -- Extract and log key metrics --
        from ultralytics.utils.torch_utils import get_flops
        flops = get_flops(val_model.model, dataset_cfg["image_size"])
        metrics = {
            "model": "baseline_yolo11n_fp32",
            "parameters_M": round(sum(p.numel() for p in val_model.model.parameters()) / 1e6, 3),
            "model_size_MB": round(size_mb, 2),
            "GFLOPs": round(float(flops), 1) if flops else 0.0,
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50_95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        }

        logger.info("=" * 70)
        logger.info("BASELINE METRICS SUMMARY")
        logger.info("=" * 70)
        for key, value in metrics.items():
            logger.info("  %-20s: %s", key, value)
        logger.info("=" * 70)

        # Save metrics to JSON
        metrics_path = Path(paths["logs_dir"]) / "phase1_baseline_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Metrics saved to: %s", metrics_path)

        del val_model

    except Exception:
        logger.exception("Phase 1 baseline training FAILED with an exception.")
        raise
    finally:
        # -----------------------------------------------------------
        # Step 8: Defensive memory cleanup
        # -----------------------------------------------------------
        if model is not None:
            del model
        flush_memory(device=hw["device"])
        log_vram_status(tag="post-cleanup")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    logger.info("=" * 70)
    logger.info(
        "PHASE 1 COMPLETE -- Elapsed: %.1f seconds (%.2f hours)", elapsed, hours
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        train_baseline()
    except Exception:
        logger.error("Phase 1 terminated with errors. See log above.")
        sys.exit(1)
