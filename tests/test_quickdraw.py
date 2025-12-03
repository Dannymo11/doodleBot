import json
from pathlib import Path

import numpy as np

from src.data.quickdraw import load_raster_samples, load_vector_samples


def _write_ndjson(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_raster_samples(tmp_path):
    cat_data = (np.arange(28 * 28 * 3) % 255).astype(np.uint8).reshape(3, 28 * 28)
    dog_data = np.full((2, 28 * 28), 128, dtype=np.uint8)
    np.save(tmp_path / "full_numpy_bitmap_cat.npy", cat_data)
    np.save(tmp_path / "dog.npy", dog_data)

    images, label_ids, label_names = load_raster_samples(
        tmp_path,
        labels=["cat", "dog"],
        samples_per_label=2,
        normalize=True,
    )

    assert images.shape == (4, 28, 28)
    assert np.all((0.0 <= images) & (images <= 1.0))
    assert label_ids.tolist() == [0, 0, 1, 1]
    assert label_names == ["cat", "dog"]


def test_load_vector_samples(tmp_path):
    cat_records = [
        {"word": "cat", "drawing": [[[0, 1], [0, 1]]]},
        {"word": "cat", "drawing": [[[2, 3, 4], [3, 4, 5]], [[5, 6], [6, 7]]]},
    ]
    dog_records = [
        {"word": "dog", "drawing": [[[1, 2], [2, 3]]]},
    ]

    _write_ndjson(tmp_path / "cat.ndjson", cat_records)
    _write_ndjson(tmp_path / "dog.ndjson", dog_records)

    strokes, times, label_ids, label_names = load_vector_samples(
        tmp_path,
        labels=["cat", "dog"],
        samples_per_label=2,
    )

    assert len(strokes) == len(times) == len(label_ids) == 3
    assert label_ids.tolist() == [0, 0, 1]
    assert label_names == ["cat", "dog"]
    # ensure each stroke entry is a list of numpy arrays
    assert all(isinstance(seq, list) for seq in strokes)
    assert all(arr.shape[1] == 2 for seq in strokes for arr in seq)
