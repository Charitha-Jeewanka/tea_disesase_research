"""
src/data/subset.py -- Create temporary dataset subsets for smoke testing.

Copies a random subset of images and their matching labels into a
temporary directory, producing a valid data.yaml for Ultralytics.
"""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)


def _collect_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> List[Dict[str, Path]]:
    """
    Match image files to their label files by stem name.

    Parameters
    ----------
    images_dir : Path
        Directory containing image files.
    labels_dir : Path
        Directory containing label .txt files.

    Returns
    -------
    list of dict
        Each dict has keys 'image' and 'label' with Path values.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.is_file():
                pairs.append({"image": img_path, "label": label_path})
    return pairs


def create_subset_dataset(
    dataset_dir: str,
    output_dir: str,
    train_count: int = 50,
    val_count: int = 20,
    class_names: list = None,
    seed: int = 42,
) -> str:
    """
    Create a small subset dataset for smoke testing.

    Parameters
    ----------
    dataset_dir : str
        Path to the full dataset directory (containing train/ and valid/).
    output_dir : str
        Path where the subset will be written.
    train_count : int
        Number of training samples to include.
    val_count : int
        Number of validation samples to include.
    class_names : list
        List of class names for the data.yaml.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Absolute path to the generated subset data.yaml.
    """
    random.seed(seed)

    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    # -- Clean previous subset if it exists --
    if output_path.exists():
        shutil.rmtree(output_path)
        logger.info("Removed previous subset at: %s", output_path)

    splits = {
        "train": {
            "images_src": dataset_path / "train" / "images",
            "labels_src": dataset_path / "train" / "labels",
            "count": train_count,
        },
        "val": {
            "images_src": dataset_path / "valid" / "images",
            "labels_src": dataset_path / "valid" / "labels",
            "count": val_count,
        },
    }

    for split_name, split_info in splits.items():
        images_dst = output_path / split_name / "images"
        labels_dst = output_path / split_name / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)

        pairs = _collect_image_label_pairs(
            split_info["images_src"],
            split_info["labels_src"],
        )

        if len(pairs) < split_info["count"]:
            logger.warning(
                "Requested %d %s samples but only %d available. Using all.",
                split_info["count"],
                split_name,
                len(pairs),
            )
            selected = pairs
        else:
            selected = random.sample(pairs, split_info["count"])

        for pair in selected:
            shutil.copy2(pair["image"], images_dst / pair["image"].name)
            shutil.copy2(pair["label"], labels_dst / pair["label"].name)

        logger.info(
            "Subset %s: copied %d image-label pairs.",
            split_name,
            len(selected),
        )

    # -- Generate data.yaml for the subset --
    nc = len(class_names) if class_names else 8
    data_yaml = {
        "train": str((output_path / "train" / "images").resolve()),
        "val": str((output_path / "val" / "images").resolve()),
        "nc": nc,
        "names": class_names or [],
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data_yaml, fh, default_flow_style=False, allow_unicode=True)

    logger.info("Subset data.yaml written to: %s", yaml_path.resolve())
    return str(yaml_path.resolve())
