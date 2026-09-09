import unittest
from collections import Counter
import json

from src.split_manifest import load_split_manifest, rows_for_split


class SplitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_split_manifest("data/splits.csv")

    def test_expected_size_and_partition_counts(self):
        self.assertEqual(len(self.rows), 2148)
        counts = Counter(row["split_multiclass_3class"] for row in self.rows)
        self.assertEqual(counts, {"train": 1374, "val": 344, "test": 430})

    def test_paths_are_unique(self):
        paths = [row["filepath"].casefold() for row in self.rows]
        self.assertEqual(len(paths), len(set(paths)))

    def test_every_partition_contains_all_three_classes(self):
        for split in ("train", "val", "test"):
            _, labels, _ = rows_for_split(self.rows, split)
            self.assertEqual(set(labels), {0, 1, 2})

    def test_experiment_config_uses_same_frozen_manifest(self):
        with open("configs/experiment_3class.json", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["model"]["num_classes"], 3)
        self.assertEqual(config["data_split"]["manifest"], "data/splits.csv")
        self.assertEqual(
            config["data_split"]["split_column"], "split_multiclass_3class"
        )
        self.assertTrue(config["data_split"]["immutable_during_training"])


if __name__ == "__main__":
    unittest.main()
