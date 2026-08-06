"""
scripts/prune_model.py -- Phase 2: Iterative Structured Channel Pruning.

Loads the Phase 1 baseline checkpoint and applies iterative structured
channel pruning with 20-epoch recovery fine-tuning cycles. Targets a
30% reduction in effective (non-zero) parameters.

Each iteration:
  1. Applies L1-norm structured pruning to Conv layers in C3k2/SPPF modules
  2. Runs a 20-epoch recovery fine-tuning cycle with reduced augmentation
  3. Validates and checkpoints

The pruning zeroes out entire output channels. Physical channel removal
occurs during the OpenVINO export in Phase 3.

Respects the 6GB VRAM budget through AMP, batch capping, gradient
accumulation (nbs=64), and defensive memory management.

Output Artifacts:
    - weights/pruned_yolo11n_fp32.pt
    - logs/phase2_pruning_metrics.json
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


def run_phase2_pruning() -> None:
    """Execute the Phase 2 iterative pruning and recovery pipeline."""

    # ---------------------------------------------------------------
    # Step 1: Load configuration
    # ---------------------------------------------------------------
    config = load_config("configs/config.yaml")
    hw = config["hardware"]
    p1 = config["phase1"]
    p2 = config["phase2"]
    dataset_cfg = config["dataset"]
    paths = config["paths"]

    # ---------------------------------------------------------------
    # Step 2: Setup logging
    # ---------------------------------------------------------------
    setup_logging(
        log_dir=paths["logs_dir"],
        log_filename="phase2_pruning.log",
        project_root=str(PROJECT_ROOT),
    )

    logger.info("=" * 70)
    logger.info("PHASE 2: ITERATIVE STRUCTURED CHANNEL PRUNING")
    logger.info("=" * 70)
    start_time = time.time()

    # ---------------------------------------------------------------
    # Step 3: Pre-pipeline diagnostics
    # ---------------------------------------------------------------
    log_system_ram()
    log_vram_status(tag="pre-pruning-init")
    check_vram(min_free_gb=hw["vram_min_free_gb"])
    flush_memory(device=hw["device"])

    # ---------------------------------------------------------------
    # Step 4: Load baseline metrics from Phase 1
    # ---------------------------------------------------------------
    baseline_metrics_path = Path(paths["logs_dir"]) / "phase1_baseline_metrics.json"
    if not baseline_metrics_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 baseline metrics not found: {baseline_metrics_path.resolve()}"
        )

    with open(baseline_metrics_path, "r", encoding="utf-8") as fh:
        baseline_metrics = json.load(fh)

    logger.info("Phase 1 Baseline Metrics:")
    for key, value in baseline_metrics.items():
        logger.info("  %-20s: %s", key, value)

    # ---------------------------------------------------------------
    # Step 5: Verify dataset
    # ---------------------------------------------------------------
    dataset_yaml = Path(paths["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml.resolve()}"
        )

    # ---------------------------------------------------------------
    # Step 6: Load baseline model
    # ---------------------------------------------------------------
    model = None
    try:
        from ultralytics import YOLO
        from src.optimization.prune import (
            prune_model_structured,
            count_parameters,
            count_nonzero_parameters,
        )

        baseline_model_path = Path(paths["weights_dir"]) / "baseline_yolo11n_fp32.pt"
        if not baseline_model_path.is_file():
            raise FileNotFoundError(
                f"Baseline model checkpoint not found: {baseline_model_path.resolve()}"
            )

        # To support resuming, check for the highest iteration pre-finetune checkpoint available on disk
        start_iteration = 0
        latest_checkpoint = None
        for i in range(8, 0, -1):
            chk_path = Path(paths["weights_dir"]) / f"pruned_iter{i}_pre_finetune.pt"
            if chk_path.is_file():
                latest_checkpoint = chk_path
                start_iteration = i
                break
        
        if latest_checkpoint is not None:
            logger.info("Found checkpoint %s. Resuming Phase II from Iteration %d.", latest_checkpoint, start_iteration + 1)
            model = YOLO(str(latest_checkpoint))
        else:
            logger.info("Loading baseline model from: %s", baseline_model_path)
            model = YOLO(str(baseline_model_path))
            
        log_vram_status(tag="post-model-load")

        # Record baseline counts (always using the original config values/files for baseline target)
        baseline_total_params = count_parameters(YOLO(str(baseline_model_path)).model)
        baseline_nonzero = count_nonzero_parameters(YOLO(str(baseline_model_path)).model)
        
        # Hard-coded target parameter count for true 30% reduction from baseline
        target_nonzero = 1813979

        logger.info(
            "Baseline parameters:     %d total, %d nonzero (%.3f M)",
            baseline_total_params,
            baseline_nonzero,
            baseline_nonzero / 1e6,
        )
        logger.info(
            "Target nonzero params:   %d (%.3f M)",
            target_nonzero,
            target_nonzero / 1e6,
        )

        # -- Gradient accumulation setup --
        nbs = hw["batch_size"] * hw["accumulate"]
        logger.info(
            "Recovery training config: batch=%d, nbs=%d (accumulate=%d), amp=%s",
            hw["batch_size"], nbs, hw["accumulate"], hw["amp"],
        )

        # -----------------------------------------------------------
        # Step 7: Iterative pruning loop
        # -----------------------------------------------------------
        iteration = start_iteration
        current_nonzero = count_nonzero_parameters(model.model)
        iteration_log = []
        # The per-layer pruning ratio from config (0.10)
        step_ratio = p2["pruning_step"]

        while current_nonzero > target_nonzero:
            iteration += 1
            iter_start = time.time()

            logger.info("-" * 70)
            logger.info("PRUNING ITERATION %d", iteration)
            logger.info("-" * 70)

            # Step A: Pre-iteration memory check
            check_vram(min_free_gb=hw["vram_min_free_gb"])
            log_vram_status(tag=f"pre-prune-iter-{iteration}")
            flush_memory(device=hw["device"])

            # Step B: Dynamically compute cumulative pruning ratio
            # Iterative pruning with static mask logic: we must increase the global target ratio
            # linearly at each iteration (e.g. Iter 1: 10%, Iter 2: 20%, Iter 3: 30%, Iter 4: 40%)
            global_target = p2["pruning_target"]
            effective_ratio = min(iteration * step_ratio, global_target)

            logger.info(
                "Current nonzero: %d, target: %d, "
                "cumulative pruning target: %.1f%%, "
                "effective pruning ratio: %.2f",
                current_nonzero,
                target_nonzero,
                (iteration * step_ratio) * 100.0,
                effective_ratio,
            )

            # Step C: Prune
            pruned_model, prune_stats = prune_model_structured(
                model.model,
                pruning_ratio=effective_ratio,
                min_channels=8,
                baseline_params=baseline_nonzero,
            )
            model.model = pruned_model

            current_nonzero = count_nonzero_parameters(model.model)
            cumulative_reduction = 1.0 - (current_nonzero / baseline_nonzero)

            logger.info(
                "Iteration %d pruning result: %d nonzero params "
                "(%.3f M, %.1f%% cumulative reduction)",
                iteration,
                current_nonzero,
                current_nonzero / 1e6,
                cumulative_reduction * 100,
            )

            # Step D: Save intermediate checkpoint (before recovery)
            intermediate_path = (
                Path(paths["weights_dir"])
                / f"pruned_iter{iteration}_pre_finetune.pt"
            )
            model.save(str(intermediate_path))
            logger.info("Saved pre-finetune checkpoint: %s", intermediate_path)

            # Step E: Recovery fine-tuning
            logger.info(
                "Starting %d-epoch recovery fine-tuning for iteration %d...",
                p2["recovery_epochs"],
                iteration,
            )
            flush_memory(device=hw["device"])
            log_vram_status(tag=f"pre-recovery-iter-{iteration}")

            results = model.train(
                data=str(dataset_yaml),
                epochs=p2["recovery_epochs"],
                patience=p2["recovery_patience"],
                optimizer=p1["optimizer"],
                lr0=p1["lr0"] * 0.1,  # Reduced LR for recovery
                lrf=p1["lrf"],
                cos_lr=p1["cos_lr"],
                batch=hw["batch_size"],
                nbs=nbs,
                imgsz=dataset_cfg["image_size"],
                device=hw["device"],
                amp=hw["amp"],
                workers=hw["workers"],
                cache=hw["cache"],
                # Disable heavy augmentations for recovery
                mosaic=0.0,
                mixup=0.0,
                # Output
                project=p2["project"],
                name=f"{p2['name']}_iter{iteration}",
                exist_ok=True,
                verbose=True,
                save=True,
                plots=True,
            )

            # Step F: Post-recovery reload best checkpoint
            flush_memory(device=hw["device"])
            log_vram_status(tag=f"post-recovery-iter-{iteration}")

            recovery_dir = Path(model.trainer.save_dir)
            best_pt = recovery_dir / "weights" / "best.pt"
            last_pt = recovery_dir / "weights" / "last.pt"

            if best_pt.is_file():
                model = YOLO(str(best_pt))
                logger.info("Loaded best recovery checkpoint: %s", best_pt)
            elif last_pt.is_file():
                model = YOLO(str(last_pt))
                logger.info("Loaded last recovery checkpoint: %s", last_pt)
            else:
                logger.warning(
                    "No recovery checkpoint found in %s. Continuing with current model.",
                    recovery_dir / "weights",
                )

            current_nonzero = count_nonzero_parameters(model.model)
            cumulative_reduction = 1.0 - (current_nonzero / baseline_nonzero)

            iter_elapsed = time.time() - iter_start
            logger.info(
                "Post-recovery iteration %d: %d nonzero params (%.3f M), "
                "cumulative reduction: %.1f%%, elapsed: %.0f s",
                iteration,
                current_nonzero,
                current_nonzero / 1e6,
                cumulative_reduction * 100,
                iter_elapsed,
            )

            iteration_log.append({
                "iteration": iteration,
                "nonzero_params": current_nonzero,
                "nonzero_params_M": round(current_nonzero / 1e6, 3),
                "cumulative_reduction_pct": round(cumulative_reduction * 100, 1),
                "effective_ratio": round(effective_ratio, 3),
                "channels_pruned": prune_stats.get("channels_pruned", 0),
                "elapsed_s": round(iter_elapsed, 1),
            })

            # Check termination conditions
            if current_nonzero <= target_nonzero:
                logger.info("Target nonzero parameter count reached. Stopping pruning loop.")
                break

            if iteration >= 8:
                logger.warning("Maximum pruning iterations (8) reached. Stopping.")
                break

        # -----------------------------------------------------------
        # Step 8: Final validation
        # -----------------------------------------------------------
        logger.info("=" * 70)
        logger.info("FINAL VALIDATION ON PRUNED MODEL")
        logger.info("=" * 70)
        flush_memory(device=hw["device"])
        log_vram_status(tag="pre-final-val")

        val_results = model.val(
            data=str(dataset_yaml),
            imgsz=dataset_cfg["image_size"],
            device=hw["device"],
            batch=hw["batch_size"],
            workers=hw["workers"],
        )

        log_vram_status(tag="post-final-val")

        # -----------------------------------------------------------
        # Step 9: Save final pruned model
        # -----------------------------------------------------------
        final_model_path = Path(paths["weights_dir"]) / "pruned_yolo11n_fp32.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Use the best recovery checkpoint from the last iteration
        source_pt = None
        if best_pt.is_file():
            source_pt = best_pt
        elif last_pt.is_file():
            source_pt = last_pt

        if source_pt is not None:
            shutil.copy2(str(source_pt), str(final_model_path))
        else:
            model.save(str(final_model_path))

        size_mb = final_model_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Final pruned model saved: %s (%.2f MB)", final_model_path, size_mb
        )

        # -----------------------------------------------------------
        # Step 10: Compute and save final metrics
        # -----------------------------------------------------------
        try:
            from ultralytics.utils.torch_utils import get_flops
            gflops = round(float(get_flops(model.model, dataset_cfg["image_size"])), 1)
        except Exception:
            logger.warning("Could not compute GFLOPs. Falling back to 0.0.")
            gflops = 0.0

        final_total_params = count_parameters(model.model)
        final_nonzero = count_nonzero_parameters(model.model)

        final_metrics = {
            "model": "pruned_yolo11n_fp32",
            "parameters_M": round(final_total_params / 1e6, 3),
            "nonzero_parameters_M": round(final_nonzero / 1e6, 3),
            "model_size_MB": round(size_mb, 2),
            "GFLOPs": gflops,
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50_95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
            "pruning_iterations": iteration,
            "total_reduction_pct": round(
                100.0 * (1.0 - final_nonzero / baseline_nonzero), 1
            ),
            "iteration_log": iteration_log,
        }

        metrics_path = Path(paths["logs_dir"]) / "phase2_pruning_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(final_metrics, fh, indent=2)
        logger.info("Pruning metrics saved to: %s", metrics_path)

        # -----------------------------------------------------------
        # Step 11: Comparison summary
        # -----------------------------------------------------------
        b = baseline_metrics
        p = final_metrics

        def pct_delta(new_val, old_val):
            if old_val == 0:
                return 0.0
            return ((new_val - old_val) / old_val) * 100.0

        d_params = pct_delta(p["nonzero_parameters_M"], b["parameters_M"])
        d_gflops = pct_delta(p["GFLOPs"], b["GFLOPs"])
        d_size = pct_delta(p["model_size_MB"], b["model_size_MB"])
        d_map50 = pct_delta(p["mAP50"], b["mAP50"])
        d_map = pct_delta(p["mAP50_95"], b["mAP50_95"])
        d_prec = pct_delta(p["precision"], b["precision"])
        d_rec = pct_delta(p["recall"], b["recall"])

        summary = (
            "\n"
            "=" * 70 + "\n"
            "PHASE 2 PRUNING RESULTS vs PHASE 1 BASELINE\n"
            "=" * 70 + "\n"
            "  Metric               | Baseline     | Pruned       | Delta\n"
            "-" * 70 + "\n"
            f"  Parameters (M)       | {b['parameters_M']:<12.3f} | {p['nonzero_parameters_M']:<12.3f} | {d_params:+.1f}%\n"
            f"  GFLOPs               | {b['GFLOPs']:<12.1f} | {p['GFLOPs']:<12.1f} | {d_gflops:+.1f}%\n"
            f"  Model Size (MB)      | {b['model_size_MB']:<12.2f} | {p['model_size_MB']:<12.2f} | {d_size:+.1f}%\n"
            f"  mAP@0.5              | {b['mAP50']:<12.4f} | {p['mAP50']:<12.4f} | {d_map50:+.2f}%\n"
            f"  mAP@0.5:0.95         | {b['mAP50_95']:<12.4f} | {p['mAP50_95']:<12.4f} | {d_map:+.2f}%\n"
            f"  Precision            | {b['precision']:<12.4f} | {p['precision']:<12.4f} | {d_prec:+.2f}%\n"
            f"  Recall               | {b['recall']:<12.4f} | {p['recall']:<12.4f} | {d_rec:+.2f}%\n"
            "=" * 70
        )
        logger.info(summary)

        del val_results

    except Exception:
        logger.exception("Phase 2 pruning pipeline FAILED with an exception.")
        raise
    finally:
        # -----------------------------------------------------------
        # Defensive memory cleanup
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
        "PHASE 2 COMPLETE -- Elapsed: %.1f seconds (%.2f hours)", elapsed, hours
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        run_phase2_pruning()
    except Exception:
        logger.error("Phase 2 terminated with errors. See log above.")
        sys.exit(1)
