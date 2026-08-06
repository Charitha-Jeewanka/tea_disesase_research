"""
src/optimization/prune.py -- Structured channel pruning for YOLO11n.

Implements L1-norm structured channel pruning targeting Conv2d layers
with BatchNorm2d inside C3k2 modules and SPPF pooling layers of the
YOLO11n architecture, while preserving the Detect head and stem layers.

Uses PyTorch's built-in torch.nn.utils.prune for structured pruning
at the output-channel dimension (dim=0), which zeros out entire filters.
The pruning masks are made permanent after each step so subsequent
fine-tuning adapts the remaining channels.

For physical channel removal, the pruned model is exported to OpenVINO
in Phase 3, where the runtime optimizer folds away zero-weight channels.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C3k2, SPPF
except ImportError:
    raise ImportError(
        "ultralytics is required but could not import YOLO module types. "
        "Install it with: pip install ultralytics"
    )

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """
    Return the total number of parameters in the model.

    Counts all parameters regardless of requires_grad status, since
    checkpoint-loaded models may have requires_grad=False.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    int
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """
    Return the number of non-zero parameters in the model.

    This is the effective parameter count after structured pruning,
    since pruned channels have their weights zeroed out.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    int
        Number of non-zero parameters.
    """
    total = 0
    for p in model.parameters():
        total += torch.count_nonzero(p).item()
    return total


def count_zero_parameters(model: nn.Module) -> int:
    """
    Return the number of zeroed-out parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    int
        Number of zero parameters.
    """
    total = 0
    for p in model.parameters():
        total += (p == 0).sum().item()
    return total


def _is_descendant_of_detect(model: nn.Module, target_module: nn.Module) -> bool:
    """
    Check whether target_module is a descendant of any Detect module.

    Parameters
    ----------
    model : nn.Module
        Root model to search within.
    target_module : nn.Module
        The module to check ancestry for.

    Returns
    -------
    bool
        True if target_module is inside a Detect module.
    """
    for module in model.modules():
        if isinstance(module, Detect):
            for child in module.modules():
                if child is target_module:
                    return True
    return False


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return a list of (name, Conv module) tuples for all prunable Conv
    layers inside C3k2 and SPPF modules, excluding the Detect head,
    first Conv stem layer, and narrow layers (channels <= 32).

    Only includes Ultralytics Conv modules (Conv2d + BN + activation)
    that reside within C3k2 or SPPF parent modules.

    Parameters
    ----------
    model : nn.Module
        The YOLO model (typically model.model for Ultralytics YOLO).

    Returns
    -------
    list
        List of (name, Conv module) tuples eligible for pruning.
    """
    # Collect C3k2 and SPPF modules not inside Detect
    prunable_parents = set()
    for module in model.modules():
        if isinstance(module, (C3k2, SPPF)):
            if not _is_descendant_of_detect(model, module):
                prunable_parents.add(id(module))

    prunable_layers = []
    for name, module in model.named_modules():
        # Blacklist the very first Conv stem layer (usually model.0)
        if name.startswith("model.0"):
            continue

        if isinstance(module, (C3k2, SPPF)) and id(module) in prunable_parents:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, Conv):
                    conv2d = sub_module.conv
                    # Width Threshold: completely skip narrow layers
                    if conv2d.in_channels <= 32 or conv2d.out_channels <= 32:
                        continue
                    
                    if hasattr(sub_module, "bn") and isinstance(sub_module.bn, nn.BatchNorm2d):
                        full_name = f"{name}.{sub_name}" if sub_name else name
                        prunable_layers.append((full_name, sub_module))

    logger.info(
        "Found %d prunable Conv layers inside C3k2/SPPF modules after filtering narrow layers and stem.",
        len(prunable_layers),
    )
    return prunable_layers


def compute_bn_threshold(model: nn.Module, pruning_ratio: float) -> float:
    """
    Compute the global BN gamma L1-norm threshold for the given pruning ratio.

    Collects absolute BN gamma values from all prunable layers and returns
    the value at the pruning_ratio percentile.

    Parameters
    ----------
    model : nn.Module
        The YOLO model.
    pruning_ratio : float
        Fraction of channels to prune (0.0 to 1.0).

    Returns
    -------
    float
        The L1-norm threshold value.
    """
    gammas = []
    prunable_layers = get_prunable_layers(model)

    for _name, module in prunable_layers:
        if hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
            gammas.append(module.bn.weight.data.abs().clone().cpu())

    if not gammas:
        logger.warning("No BN gammas found in prunable layers. Returning 0.0.")
        return 0.0

    all_gammas = torch.cat(gammas)
    sorted_gammas, _ = torch.sort(all_gammas)

    threshold_index = int(len(sorted_gammas) * pruning_ratio)
    threshold_index = max(0, min(threshold_index, len(sorted_gammas) - 1))

    threshold = sorted_gammas[threshold_index].item()
    logger.info(
        "BN gamma threshold for %.1f%% pruning: %.6f (from %d total channels)",
        pruning_ratio * 100,
        threshold,
        len(all_gammas),
    )
    return threshold


def get_channel_survival_mask(
    bn_layer: nn.BatchNorm2d,
    threshold: float,
    min_channels: int = 8,
) -> torch.Tensor:
    """
    Return a boolean mask indicating which channels survive pruning.

    Parameters
    ----------
    bn_layer : nn.BatchNorm2d
        The batch normalization layer.
    threshold : float
        Channels with BN gamma below this are pruned.
    min_channels : int
        Minimum channels that must survive per layer.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (num_features,). True = survives.
    """
    gamma_mags = bn_layer.weight.data.abs()
    mask = gamma_mags >= threshold

    surviving_count = mask.sum().item()
    if surviving_count < min_channels:
        _, top_indices = torch.topk(gamma_mags, min(min_channels, len(gamma_mags)))
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[top_indices] = True

    return mask


def _apply_structured_pruning_to_conv(
    conv_module: nn.Conv2d,
    prune_ratio: float,
    min_channels: int = 8,
) -> int:
    """
    Apply L1-norm structured pruning to a Conv2d layer at the
    output channel dimension (dim=0), then make the pruning permanent.

    Parameters
    ----------
    conv_module : nn.Conv2d
        The convolutional layer to prune.
    prune_ratio : float
        Fraction of output channels to zero out (0.0 to 1.0).
    min_channels : int
        Minimum active channels to preserve.

    Returns
    -------
    int
        Number of channels that were zeroed out.
    """
    num_channels = conv_module.out_channels
    num_to_prune = int(num_channels * prune_ratio)

    # Ensure we keep at least min_channels
    if num_channels - num_to_prune < min_channels:
        num_to_prune = max(0, num_channels - min_channels)

    if num_to_prune == 0:
        return 0

    # Apply L1 structured pruning on output channels (dim=0)
    prune.ln_structured(conv_module, name="weight", amount=num_to_prune, n=1, dim=0)

    # Make the pruning permanent (remove the mask, apply to weights)
    prune.remove(conv_module, "weight")

    return num_to_prune


def _zero_bn_for_pruned_channels(
    bn_module: nn.BatchNorm2d,
    conv_module: nn.Conv2d,
) -> None:
    """
    Zero out BN parameters for channels whose Conv2d filters are all zeros.

    This ensures consistency between the Conv2d and its paired BN layer
    after structured pruning.

    Parameters
    ----------
    bn_module : nn.BatchNorm2d
        The batch normalization layer paired with conv_module.
    conv_module : nn.Conv2d
        The pruned Conv2d layer.
    """
    with torch.no_grad():
        # Find which output channels are fully zeroed in the conv
        weight = conv_module.weight.data
        # Sum across input channels, height, width
        channel_norms = weight.view(weight.size(0), -1).abs().sum(dim=1)
        zero_mask = channel_norms == 0

        # Zero out corresponding BN parameters
        bn_module.weight.data[zero_mask] = 0.0
        bn_module.bias.data[zero_mask] = 0.0
        bn_module.running_mean[zero_mask] = 0.0
        bn_module.running_var[zero_mask] = 1.0  # Avoid division by zero


def prune_model_structured(
    model: nn.Module,
    pruning_ratio: float = 0.15,
    min_channels: int = 8,
    baseline_params: Optional[int] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Perform one step of structured channel pruning on the YOLO model.

    Uses L1-norm structured pruning via torch.nn.utils.prune on the
    Conv2d layers inside C3k2 and SPPF modules. Entire output channels
    (filters) with the smallest L1-norm are zeroed out. BN parameters
    for the pruned channels are also zeroed for consistency.

    The Detect head and narrow/stem layers are excluded from pruning.

    After pruning, recovery fine-tuning should be run to adapt the
    model to the reduced capacity. Physical channel removal occurs
    during OpenVINO export in Phase 3.

    Parameters
    ----------
    model : nn.Module
        The YOLO model to prune (the .model attribute of an Ultralytics
        YOLO object, i.e. the nn.Sequential backbone+head).
    pruning_ratio : float
        Fraction of output channels to prune per layer (0.0 to 1.0).
        Typically 0.15 to 0.20 for wider blocks.
    min_channels : int
        Minimum channels to preserve per layer.
    baseline_params : int or None
        Original baseline total parameter count for cumulative
        tracking. If None, uses the pre-pruning non-zero count.

    Returns
    -------
    tuple
        (model, stats_dict) where stats_dict contains:
        - original_params: total params before pruning
        - original_nonzero: non-zero params before pruning
        - pruned_nonzero: non-zero params after pruning
        - reduction_pct: percentage reduction in non-zero params (this step)
        - cumulative_reduction_pct: total reduction from baseline
        - layers_pruned: number of Conv layers that had channels zeroed
        - channels_pruned: total number of channels zeroed across all layers
    """
    original_total = count_parameters(model)
    original_nonzero = count_nonzero_parameters(model)

    if baseline_params is None:
        baseline_params = original_nonzero

    logger.info(
        "Starting structured pruning: ratio=%.2f, total_params=%d, "
        "nonzero_params=%d, baseline=%d",
        pruning_ratio,
        original_total,
        original_nonzero,
        baseline_params,
    )

    # Get prunable layers (narrow layers and stem are automatically filtered out)
    prunable_layers = get_prunable_layers(model)

    layers_pruned = 0
    total_channels_pruned = 0

    for name, conv_module in prunable_layers:
        conv = conv_module.conv  # The actual nn.Conv2d
        bn = conv_module.bn     # The nn.BatchNorm2d

        # For wider blocks (e.g. 128, 256 out_channels), we prune at pruning_ratio.
        # For intermediate wider layers (e.g. 64 out_channels), we can prune slightly less (e.g. 10%)
        # to ensure stability, but default to the provided pruning_ratio for simplicity and effectiveness.
        effective_ratio = pruning_ratio
        
        channels_pruned = _apply_structured_pruning_to_conv(
            conv, 
            effective_ratio, 
            min_channels=min_channels
        )
        if channels_pruned > 0:
            _zero_bn_for_pruned_channels(bn, conv)
            layers_pruned += 1
            total_channels_pruned += channels_pruned
            logger.debug(
                "  Pruned %d/%d channels from %s",
                channels_pruned,
                conv.out_channels,
                name,
            )

    # Compute post-pruning statistics
    pruned_nonzero = count_nonzero_parameters(model)
    step_reduction_pct = (
        100.0 * (original_nonzero - pruned_nonzero) / original_nonzero
        if original_nonzero > 0
        else 0.0
    )
    cumulative_reduction_pct = (
        100.0 * (baseline_params - pruned_nonzero) / baseline_params
        if baseline_params > 0
        else 0.0
    )

    stats = {
        "original_params": original_total,
        "original_nonzero": original_nonzero,
        "pruned_nonzero": pruned_nonzero,
        "reduction_pct": round(step_reduction_pct, 2),
        "cumulative_reduction_pct": round(cumulative_reduction_pct, 2),
        "layers_pruned": layers_pruned,
        "channels_pruned": total_channels_pruned,
    }

    logger.info(
        "Pruning step complete: %d -> %d nonzero params "
        "(step: -%.2f%%, cumulative: -%.2f%%), "
        "%d layers affected, %d channels zeroed",
        original_nonzero,
        pruned_nonzero,
        step_reduction_pct,
        cumulative_reduction_pct,
        layers_pruned,
        total_channels_pruned,
    )

    return model, stats
