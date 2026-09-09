import unittest

import numpy as np

from src.signal_statistics import zero_crossing_rate


class ZeroCrossingRateTests(unittest.TestCase):
    def test_known_sign_changes(self):
        cases = [
            ([-1, 1, -1, 1], 1.0),
            ([-1, -1, 1, 1], 1 / 3),
            ([1, 2, 3], 0.0),
            ([-1, -2, -3], 0.0),
            ([0, 0, 0], 0.0),
        ]
        for audio, expected in cases:
            with self.subTest(audio=audio):
                self.assertAlmostEqual(zero_crossing_rate(audio), expected)

    def test_zeros_are_nonnegative_including_signed_zero(self):
        for audio, expected in [
            ([1, 0, 1], 0.0),
            ([-1, 0, 1], 0.5),
            ([-1, 0, -1], 1.0),
            ([0.0, -0.0, 0.0], 0.0),
            ([-1, -0.0, 1], 0.5),
        ]:
            with self.subTest(audio=audio):
                self.assertEqual(zero_crossing_rate(audio), expected)

    def test_no_adjacent_pairs(self):
        for audio in ([], [0.0], [1.0], [-1.0]):
            with self.subTest(audio=audio):
                self.assertEqual(zero_crossing_rate(audio), 0.0)

    def test_rejects_nonfinite_samples(self):
        for audio in ([np.nan], [0, np.inf], [-np.inf, 1]):
            with self.subTest(audio=audio):
                with self.assertRaisesRegex(ValueError, "finite"):
                    zero_crossing_rate(audio)

    def test_rejects_non_mono_and_non_real_samples(self):
        for audio in (1.0, [[1, -1]], np.zeros((2, 3)), [1j], ["1", "-1"]):
            with self.subTest(audio=audio):
                with self.assertRaises(ValueError):
                    zero_crossing_rate(audio)

    def test_matches_scalar_reference_without_modifying_input(self):
        rng = np.random.default_rng(42)
        audio = rng.integers(-5, 6, size=202)[::2]  # Includes zeros and non-contiguous storage.
        original = audio.copy()
        crossings = sum((a < 0) != (b < 0) for a, b in zip(audio[:-1], audio[1:]))
        expected = crossings / (len(audio) - 1)
        self.assertEqual(zero_crossing_rate(audio), expected)
        np.testing.assert_array_equal(audio, original)

    def test_dtype_and_positive_gain_do_not_change_rate(self):
        for dtype in (np.int16, np.float32, np.float64):
            audio = np.array([-2, -1, 0, 3, -4], dtype=dtype)
            self.assertEqual(zero_crossing_rate(audio), 0.5)
            self.assertEqual(zero_crossing_rate(audio * 2), 0.5)
        self.assertEqual(zero_crossing_rate(np.array([0, 255, 0], dtype=np.uint8)), 0.0)


if __name__ == "__main__":
    unittest.main()
