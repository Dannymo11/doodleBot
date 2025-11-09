"""
QuickDraw loaders for raster and vector sketches.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_raster_samples(
    root: Path | str,
    labels: Iterable[str],
    *,
    samples_per_label: int | None = None,
    normalize: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load rasterized QuickDraw sketches.

    Parameters
    ----------
    root:
        Directory containing ``*.npy`` / ``*.npz`` QuickDraw bitmaps
        (e.g., ``full_numpy_bitmap_cat.npy``).
    labels:
        Iterable of label strings to load. Order determines label ids.
    samples_per_label:
        Optional cap per label to avoid loading the entire category.
    normalize:
        When True, scale pixel intensities to [0, 1], otherwise cast using ``dtype``.
    dtype:
        Target dtype after normalization/casting.

    Returns
    -------
    images: np.ndarray
        Array of shape ``(N, H, W)`` with raster sketches.
    label_ids: np.ndarray
        Integer label ids aligned with ``images``.
    label_names: list[str]
        Canonicalized label names matching ``label_ids``.
    """

    root = Path(root)
    images: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    label_names: list[str] = []

    for idx, raw_label in enumerate(labels):
        slug = _slugify(raw_label)
        file_path = _resolve_file(
            root,
            slug,
            preferred=[
                f"full_numpy_bitmap_{slug}.npy",
                f"{slug}.npy",
                f"full_numpy_bitmap_{slug}.npz",
                f"{slug}.npz",
            ],
            fallback_suffixes=(".npy", ".npz"),
        )
        data = _load_numpy_file(file_path)
        data = _maybe_reshape_bitmap(data)
        if samples_per_label is not None:
            data = data[:samples_per_label]
        if normalize:
            data = data.astype(dtype, copy=False) / np.float32(255.0)
        else:
            data = data.astype(dtype, copy=False)
        if data.size == 0:
            logger.warning("No raster data found for label '%s' at %s", raw_label, file_path)
            continue
        images.append(data)
        label_ids.append(np.full(len(data), idx, dtype=np.int64))
        label_names.append(raw_label)

    if not images:
        raise RuntimeError("No raster samples were loaded. Check label names and dataset path.")

    stacked_images = np.concatenate(images, axis=0)
    stacked_labels = np.concatenate(label_ids, axis=0)
    return stacked_images, stacked_labels, label_names


def load_vector_samples(
    root: Path | str,
    labels: Iterable[str],
    *,
    samples_per_label: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load vector stroke sequences along with synthetic timing metadata.

    Parameters
    ----------
    root:
        Directory containing QuickDraw ``*.ndjson`` (or ``*.json``) stroke files.
    labels:
        Iterable of label strings to load. Order determines label ids.
    samples_per_label:
        Optional cap per label.

    Returns
    -------
    strokes: np.ndarray
        Ragged array (dtype=object) where each element is a list of ``(N_i, 2)`` arrays.
    times: np.ndarray
        Ragged array (dtype=object) mirroring ``strokes`` with cumulative point indices.
    label_ids: np.ndarray
        Integer label ids aligned with the returned samples.
    label_names: list[str]
        Canonicalized label names matching ``label_ids``.
    """

    root = Path(root)
    stroke_samples: list[list[np.ndarray]] = []
    time_samples: list[list[np.ndarray]] = []
    label_ids: list[int] = []
    label_names: list[str] = []

    for idx, raw_label in enumerate(labels):
        slug = _slugify(raw_label)
        file_path = _resolve_file(
            root,
            slug,
            preferred=[
                f"{slug}.ndjson",
                f"{slug}.ndjson.gz",
                f"{slug}.json",
                f"{slug}.json.gz",
            ],
            fallback_suffixes=(".ndjson", ".json", ".ndjson.gz", ".json.gz"),
        )
        count = 0
        with _open_text_file(file_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                drawing = entry.get("drawing") or entry.get("strokes")
                if drawing is None:
                    logger.debug("Skipping entry without 'drawing' for label '%s'.", raw_label)
                    continue
                strokes: list[np.ndarray] = []
                times: list[np.ndarray] = []
                cumulative_time = 0
                for stroke in drawing:
                    if len(stroke) < 2:
                        continue
                    x_coords, y_coords = stroke[:2]
                    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
                    strokes.append(coords)
                    time_indices = cumulative_time + np.arange(coords.shape[0], dtype=np.float32)
                    times.append(time_indices)
                    cumulative_time = float(time_indices[-1] + 1.0)
                if not strokes:
                    continue
                stroke_samples.append(strokes)
                time_samples.append(times)
                label_ids.append(idx)
                count += 1
                if samples_per_label is not None and count >= samples_per_label:
                    break
        if count == 0:
            logger.warning("No vector data found for label '%s' at %s", raw_label, file_path)
            continue
        label_names.append(raw_label)

    if not stroke_samples:
        raise RuntimeError("No vector samples were loaded. Check label names and dataset path.")

    strokes_array = np.array(stroke_samples, dtype=object)
    times_array = np.array(time_samples, dtype=object)
    label_array = np.array(label_ids, dtype=np.int64)
    return strokes_array, times_array, label_array, label_names


def _slugify(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")


def _resolve_file(
    root: Path,
    slug: str,
    *,
    preferred: Sequence[str],
    fallback_suffixes: Sequence[str],
) -> Path:
    for candidate in preferred:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path

    for suffix in fallback_suffixes:
        matches = sorted(root.glob(f"*{slug}*{suffix}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not locate QuickDraw file for slug '{slug}' in {root}")


def _load_numpy_file(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return data[data.files[0]]
    return np.load(path, allow_pickle=True)


def _maybe_reshape_bitmap(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        side = int(np.sqrt(array.shape[1]))
        if side * side == array.shape[1]:
            return array.reshape(array.shape[0], side, side)
    return array


def _open_text_file(path: Path):
    if path.suffix.endswith("gz"):  # handles '.gz'
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")
