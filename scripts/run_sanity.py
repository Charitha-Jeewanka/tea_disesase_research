"""
scripts/run_sanity.py -- Phase 0: Sanity / Smoke Test.

Validates the end-to-end pipeline:
  1. Configuration loading
  2. VRAM guard check
  3. Dataset subset creation (50 train / 20 val)
  4. YOLO11n 2-epoch training loop
  5. Checkpoint (.pt) export verification
  6. Memory release confirmation

Exit code 0 on success, non-zero on failure.
"""

import logging
import os
import sys
import time
from pathlib import Path

# -- Resolve project root so imports work regardless of CWD --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.config_loader import load_config, get_nested
from src.utils.logging_setup import setup_logging
from src.utils.memory import check_vram, flush_memory, log_vram_status, log_system_ram
from src.data.subset import create_subset_dataset

logger = logging.getLogger(__name__)


def run_sanity_test() -> None:
    """Execute the Phase 0 sanity / smoke test pipeline."""

    # ---------------------------------------------------------------
    # Step 1: Load configuration
    # ---------------------------------------------------------------
    config = load_config("configs/config.yaml")
    hw = config["hardware"]
    p0 = config["phase0"]
    dataset_cfg = config["dataset"]
    paths = config["paths"]

    # ---------------------------------------------------------------
    # Step 2: Setup logging
    # ---------------------------------------------------------------
    setup_logging(
        log_dir=paths["logs_dir"],
        log_filename="phase0_sanity.log",
        project_root=str(PROJECT_ROOT),
    )

    logger.info("=" * 70)
    logger.info("PHASE 0: SANITY / SMOKE TEST")
    logger.info("=" * 70)
    start_time = time.time()

    # ---------------------------------------------------------------
    # Step 3: Memory diagnostics
    # ---------------------------------------------------------------
    log_system_ram()
    log_vram_status(tag="pre-sanity")
    check_vram(min_free_gb=hw["vram_min_free_gb"])
    flush_memory(device=hw["device"])

    # ---------------------------------------------------------------
    # Step 4: Create dataset subset
    # ---------------------------------------------------------------
    logger.info("Creating dataset subset for smoke test...")
    subset_yaml = create_subset_dataset(
        dataset_dir=paths["dataset_dir"],
        output_dir=str(PROJECT_ROOT / "datasets" / "_subset_phase0"),
        train_count=p0["train_subset_size"],
        val_count=p0["val_subset_size"],
        class_names=dataset_cfg["class_names"],
        seed=42,
    )
    logger.info("Subset data.yaml: %s", subset_yaml)

    # ---------------------------------------------------------------
    # Step 5: Load YOLO11n and run 2-epoch training
    # ---------------------------------------------------------------
    logger.info("Loading YOLO11n model from: %s", paths["base_model"])
    log_vram_status(tag="pre-model-load")

    from ultralytics import YOLO

    model = None
    try:
        model = YOLO(paths["base_model"])
        log_vram_status(tag="post-model-load")

        logger.info(
            "Starting training: epochs=%d, batch=%d, imgsz=%d, device=%s, amp=%s",
            p0["epochs"],
            hw["batch_size"],
            dataset_cfg["image_size"],
            hw["device"],
            hw["amp"],
        )

        results = model.train(
            data=subset_yaml,
            epochs=p0["epochs"],
            patience=p0["patience"],
            batch=hw["batch_size"],
            imgsz=dataset_cfg["image_size"],
            device=hw["device"],
            amp=hw["amp"],
            workers=hw["workers"],
            cache=hw["cache"],
            project=p0["project"],
            name=p0["name"],
            exist_ok=True,
            verbose=True,
        )

        log_vram_status(tag="post-training")

        # -----------------------------------------------------------
        # Step 6: Verify checkpoint was saved
        # -----------------------------------------------------------
        # Use the actual save directory from the trainer, because
        # Ultralytics 8.4.x nests project under runs/detect/.
        train_dir = Path(model.trainer.save_dir)
        weights_dir = train_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        logger.info("Trainer save directory: %s", train_dir)

        if best_pt.is_file() or last_pt.is_file():
            saved = best_pt if best_pt.is_file() else last_pt
            size_mb = saved.stat().st_size / (1024 * 1024)
            logger.info(
                "Checkpoint saved successfully: %s (%.2f MB)",
                saved,
                size_mb,
            )
        else:
            logger.error(
                "No checkpoint found in %s. Training may have failed.",
                weights_dir,
            )
            raise FileNotFoundError(
                f"Expected checkpoint in {weights_dir} but none found."
            )

        # -----------------------------------------------------------
        # Step 7: Quick validation pass
        # -----------------------------------------------------------
        logger.info("Running validation on the subset...")
        val_results = model.val(
            data=subset_yaml,
            imgsz=dataset_cfg["image_size"],
            device=hw["device"],
            batch=hw["batch_size"],
        )
        log_vram_status(tag="post-validation")

    except Exception:
        logger.exception("Phase 0 sanity test FAILED with an exception.")
        raise
    finally:
        # -----------------------------------------------------------
        # Step 8: Defensive memory cleanup
        # -----------------------------------------------------------
        del model
        flush_memory(device=hw["device"])
        log_vram_status(tag="post-cleanup")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PHASE 0 COMPLETE -- Elapsed: %.1f seconds", elapsed)
    logger.info("=" * 70)
    logger.info("Sanity test PASSED. Pipeline is operational.")


if __name__ == "__main__":
    try:
        run_sanity_test()
    except Exception:
        logger.error("Phase 0 terminated with errors. See log above.")
        sys.exit(1)
