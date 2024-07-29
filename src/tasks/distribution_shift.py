from typing import Any
import numpy as np
from scipy.stats import ks_2samp


# ==========================================================================
# Utils functions
# ==========================================================================


def _scale_range(input_array, min_val: int, max_val: int):
    """
    Scales the values of an input array to a specific range.
    """
    input_array += -(np.min(input_array))
    input_array /= np.max(input_array) / (max_val - min_val)
    input_array += min_val
    return input_array


def _psi_expected_actual(expected_array, actual_array, buckets):
    """
    Calculates the expected and actual percentages in each bucket.
    """
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    expected_percents = np.histogram(expected_array, np.percentile(expected_array, breakpoints))[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, np.percentile(expected_array, breakpoints))[0] / len(actual_array)
    return expected_percents, actual_percents


def ks_test_score(train_data: np.ndarray, test_data: np.ndarray) -> Any:
    """
    This function calculates the Kolmogorov-Smirnov test statistic between two arrays.
    The KS test is a statistical test used to compare the cumulative distribution functions of two datasets.
    It is widely used in statistics and engineering to determine if two datasets come from the same parent distribution.
    """
    ks_stat, p_value = ks_2samp(train_data, test_data)
    return ks_stat


def psi(expected_array: np.ndarray, actual_array: np.ndarray, buckets: int = 10) -> float:
    """
    This function calculates the Percentage of the Sample (PSI) which is a measure of the degree
    of divergence between two probability distributions.
    It is used in anomaly detection and distribution shift analysis.
    It is calculated as the sum of the differences between the expected and actual percentages in each bucket.
    The PSI value ranges from 0 to 1, where a value of 0 indicates that the two distributions are identical,
    and a value of 1 indicates that they are very different.
    """
    expected_array = _scale_range(expected_array, 0, 10)
    actual_array = _scale_range(actual_array, 0, 10)

    expected_percents, actual_percents = _psi_expected_actual(expected_array, actual_array, buckets)
    epsilon = 1e-10
    psi_value = np.sum(
        (expected_percents - actual_percents) * np.log((expected_percents + epsilon) / (actual_percents + epsilon))
    )
    return float(psi_value)


# ==========================================================================
# Exported functions
# ==========================================================================


def distribution_shift_scoring(samples, method="psi") -> Any:
    """
    This function calculates the distribution shift score
    """

    # load first distribution serving as comparison
    initial_distribution = samples

    # compute the distribution shfit scores
    if method == "psi":
        score = psi(initial_distribution, samples)

    elif method == "ks":
        score = ks_test_score(initial_distribution, samples)

    else:
        raise NotImplementedError

    return score
