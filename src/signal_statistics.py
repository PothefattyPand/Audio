"""Stateless waveform statistics shared by EDA and classical ML features."""
import numpy as np


def zero_crossing_rate(audio):
    """Fraction of adjacent samples that cross between negative and nonnegative.

    Input must be a finite, real, one-dimensional mono waveform. Both +0.0 and
    -0.0 are nonnegative. There is no amplitude threshold, padding, or resampling.
    The denominator is N - 1 (sample pairs), not duration or sample count.
    Empty and one-sample waveforms have no pairs and return 0.0.

    This function does not fit parameters or modify the input.
    """
    values = np.asarray(audio)
    if values.ndim != 1:
        raise ValueError("zero_crossing_rate expects a one-dimensional mono waveform")
    if values.dtype.kind not in "iuf":
        raise ValueError("zero_crossing_rate expects real numeric samples")
    if not np.isfinite(values).all():
        raise ValueError("zero_crossing_rate requires finite samples")
    if values.size < 2:
        return 0.0
    negative = values < 0
    return float(np.count_nonzero(negative[1:] != negative[:-1]) / (values.size - 1))
