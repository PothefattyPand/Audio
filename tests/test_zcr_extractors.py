"""Run the real numerical function bodies without importing Torch/plotting.

Only the selected function and its shared-statistic import are compiled from
source. NumPy and SciPy are real dependencies, not mocks. This isolates the
feature arithmetic; it is not a full module-import or model integration test.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load_numerical_function(relative_path, function_name):
    path = ROOT / relative_path
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node for node in source.body
        if (isinstance(node, ast.ImportFrom) and node.module == "src.signal_statistics")
        or (isinstance(node, ast.FunctionDef) and node.name == function_name)
    ]
    namespace = {"np": np, "stats": stats}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


compute_detailed_features = load_numerical_function("eda_analysis.py", "compute_detailed_features")
extract_signal_statistical_features = load_numerical_function(
    "src/audio_features.py", "extract_signal_statistical_features"
)


class ZCRExtractorTests(unittest.TestCase):
    def test_feature_vectors_count_sign_crossings(self):
        audio = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        eda = compute_detailed_features(audio)
        baseline = extract_signal_statistical_features(audio)
        self.assertEqual(len(eda), 25)
        self.assertEqual(baseline.shape, (24,))
        self.assertEqual(baseline.dtype, np.float32)
        self.assertEqual(eda[12], 1.0)
        self.assertEqual(baseline[13], 1.0)

    def test_touching_zero_from_positive_side_is_not_a_crossing(self):
        audio = np.array([1.0, 0.0, 1.0])
        self.assertEqual(compute_detailed_features(audio)[12], 0.0)
        self.assertEqual(extract_signal_statistical_features(audio)[13], 0.0)

    def test_extractors_agree_and_do_not_modify_input(self):
        audio = np.array([-0.8, -0.2, 0.0, 0.4, -0.1, 0.0, 0.7])
        original = audio.copy()
        expected = 3 / 6
        self.assertEqual(compute_detailed_features(audio)[12], expected)
        self.assertAlmostEqual(extract_signal_statistical_features(audio)[13], expected)
        np.testing.assert_array_equal(audio, original)


if __name__ == "__main__":
    unittest.main()
