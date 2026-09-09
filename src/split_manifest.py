"""Utilities for loading and validating the frozen 3-class split manifest."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


SPLIT_COLUMN = "split_multiclass_3class"
REQUIRED_COLUMNS = {
    "filename",
    "filepath",
    "original_subfolder",
    "macro_class_index",
    "macro_class_name",
    SPLIT_COLUMN,
}
VALID_SPLITS = {"train", "val", "test"}
VALID_CLASSES = {0, 1, 2}


def load_split_manifest(manifest_path="data/splits.csv", project_root=None):
    """Return validated manifest rows with absolute, existing file paths.

    The manifest is the single source of truth. Training code must not recreate a
    split, since doing so can silently invalidate reported held-out results.
    """
    root = Path(project_root or Path.cwd()).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Frozen split manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Split manifest is empty")

    seen_paths = set()
    for line_number, row in enumerate(rows, start=2):
        split = row[SPLIT_COLUMN].strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r} on manifest line {line_number}")
        try:
            class_index = int(row["macro_class_index"])
        except ValueError as exc:
            raise ValueError(f"Invalid class index on manifest line {line_number}") from exc
        if class_index not in VALID_CLASSES:
            raise ValueError(f"Unexpected class {class_index} on manifest line {line_number}")

        audio_path = Path(row["filepath"])
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        audio_path = audio_path.resolve()
        key = str(audio_path).casefold()
        if key in seen_paths:
            raise ValueError(f"Duplicate audio path in split manifest: {audio_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Manifest audio file does not exist: {audio_path}")
        seen_paths.add(key)
        row["filepath"] = str(audio_path)
        row["macro_class_index"] = class_index
        row[SPLIT_COLUMN] = split

    split_counts = Counter(row[SPLIT_COLUMN] for row in rows)
    absent = VALID_SPLITS.difference(split_counts)
    if absent:
        raise ValueError(f"Split manifest has no samples for: {sorted(absent)}")

    for split in VALID_SPLITS:
        present_classes = {
            row["macro_class_index"] for row in rows if row[SPLIT_COLUMN] == split
        }
        if present_classes != VALID_CLASSES:
            raise ValueError(
                f"Split {split!r} has classes {sorted(present_classes)}; "
                f"expected {sorted(VALID_CLASSES)}"
            )
    return rows


def rows_for_split(rows, split):
    """Return parallel filepath, label, and source-folder lists for one split."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    selected = [row for row in rows if row[SPLIT_COLUMN] == split]
    return (
        [row["filepath"] for row in selected],
        [row["macro_class_index"] for row in selected],
        [row["original_subfolder"] for row in selected],
    )
