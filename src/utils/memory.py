"""
src/utils/memory.py -- Defensive GPU/VRAM memory management utilities.

Provides guard clauses, memory logging, and cache flush routines
to prevent OOM failures on the 6GB VRAM RTX 3050.
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_vram_info() -> dict:
    """
    Return a dictionary with current VRAM utilisation statistics.

    Returns
    -------
    dict
        Keys: 'total_gb', 'free_gb', 'used_gb', 'utilisation_pct'.
        All values are floats. Returns zeroed dict if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Returning zeroed VRAM info.")
        return {
            "total_gb": 0.0,
            "free_gb": 0.0,
            "used_gb": 0.0,
            "utilisation_pct": 0.0,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    total_gb = total_bytes / (1024 ** 3)
    free_gb = free_bytes / (1024 ** 3)
    used_gb = used_bytes / (1024 ** 3)
    utilisation_pct = (used_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0

    return {
        "total_gb": round(total_gb, 3),
        "free_gb": round(free_gb, 3),
        "used_gb": round(used_gb, 3),
        "utilisation_pct": round(utilisation_pct, 1),
    }


def log_vram_status(tag: str = "") -> dict:
    """
    Log current VRAM utilisation with an optional context tag.

    Parameters
    ----------
    tag : str
        A short label describing the checkpoint (e.g. 'pre-training', 'post-epoch-3').

    Returns
    -------
    dict
        The VRAM info dictionary.
    """
    info = get_vram_info()
    prefix = f"[VRAM {tag}]" if tag else "[VRAM]"
    logger.info(
        "%s Total: %.3f GB | Used: %.3f GB | Free: %.3f GB | Util: %.1f%%",
        prefix,
        info["total_gb"],
        info["used_gb"],
        info["free_gb"],
        info["utilisation_pct"],
    )
    return info


def check_vram(min_free_gb: float = 1.5) -> None:
    """
    Pre-execution guard clause that raises RuntimeError if free VRAM
    is below the specified threshold.

    Parameters
    ----------
    min_free_gb : float
        Minimum free VRAM in gigabytes required to proceed.

    Raises
    ------
    RuntimeError
        If available VRAM is below the threshold.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Skipping VRAM guard check."
        )
        return

    info = get_vram_info()
    if info["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {info['free_gb']:.3f} GB free, "
            f"but {min_free_gb:.1f} GB required. "
            f"Total: {info['total_gb']:.3f} GB, Used: {info['used_gb']:.3f} GB. "
            f"Free up GPU memory before proceeding."
        )
    logger.info(
        "VRAM guard passed: %.3f GB free (minimum: %.1f GB).",
        info["free_gb"],
        min_free_gb,
    )


def flush_memory(device: Optional[int] = None) -> None:
    """
    Aggressively free GPU and CPU memory.

    Performs Python garbage collection followed by CUDA cache clearing.

    Parameters
    ----------
    device : int or None
        CUDA device index. If None, clears cache on the current device.
    """
    gc.collect()
    if torch.cuda.is_available():
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.debug("GPU memory cache flushed successfully.")
    else:
        logger.debug("CUDA not available. Only Python GC executed.")


def log_system_ram() -> dict:
    """
    Log system RAM utilisation using psutil if available.

    Returns
    -------
    dict
        Keys: 'total_gb', 'available_gb', 'used_pct'.
        Returns empty dict if psutil is not installed.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        info = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_pct": mem.percent,
        }
        logger.info(
            "[RAM] Total: %.2f GB | Available: %.2f GB | Used: %.1f%%",
            info["total_gb"],
            info["available_gb"],
            info["used_pct"],
        )
        return info
    except ImportError:
        logger.warning("psutil not installed. System RAM logging unavailable.")
        return {}
