from scipy.stats import ks_2samp
import numpy as np

# ==========================================================================
# Module variables
# ==========================================================================
MIN_VAL = 0
MAX_VAL = 10
EPSILON = 1e-10


# ==========================================================================
# Distribution Comparison Metrics (PSI, KS)
# Compare the overall shapes or cumulative properties of two distributions.
# Focused on whether two distributions are similar or come from the same population.
# ==========================================================================


def _scale_range(input_array, min_val: int, max_val: int) -> np.ndarray:
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

    Parameters:
    - expected_array (np.ndarray): Expected distribution array.
    - actual_array (np.ndarray): Actual distribution array.
    - buckets (int): Number of buckets to divide the distributions into.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Percentages in each bucket for expected and actual arrays.
    """
    if not isinstance(expected_array, np.ndarray):
        expected_array = np.array(expected_array)
    if not isinstance(actual_array, np.ndarray):
        actual_array = np.array(actual_array)

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    expected_percents = np.histogram(expected_array, np.percentile(expected_array, breakpoints))[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, np.percentile(expected_array, breakpoints))[0] / len(actual_array)
    return expected_percents, actual_percents


def ks_test_score(train_data: np.ndarray, test_data: np.ndarray) -> float:
    """
    This function calculates the Kolmogorov-Smirnov test statistic between two arrays.
    The KS test is a statistical test used to compare the cumulative distribution functions of two datasets.
    It is widely used in statistics and engineering to determine if two datasets come from the same parent distribution.
    """
    ks_stat, _ = ks_2samp(train_data, test_data)
    return ks_stat


def psi(expected_array: np.ndarray, actual_array: np.ndarray, buckets: int = 10):
    """
    This function calculates the Percentage of the Sample (PSI) which is a measure of the degree
    of divergence between two probability distributions.
    It is used in anomaly detection and distribution shift analysis.
    It is calculated as the sum of the differences between the expected and actual percentages in each bucket.
    The PSI value ranges from 0 to 1, where a value of 0 indicates that the two distributions are identical,
    and a value of 1 indicates that they are very different.
    """
    expected_array = _scale_range(expected_array, MIN_VAL, MAX_VAL)
    actual_array = _scale_range(actual_array, MIN_VAL, MAX_VAL)

    expected_percents, actual_percents = _psi_expected_actual(expected_array, actual_array, buckets)
    psi_value = np.sum(
        (expected_percents - actual_percents) * np.log((expected_percents + EPSILON) / (actual_percents + EPSILON))
    )
    return psi_value


# ==========================================================================
# Divergence metric (KL, JS)
# Quantify the difference between two probability distributions.
# focused on how one distribution diverges from another.
# ==========================================================================


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two distributions.
    """
    p = np.clip(p, EPSILON, 1)
    q = np.clip(q, EPSILON, 1)
    return np.sum(p * np.log(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculates the Jensen-Shannon (JS) divergence between two distributions.
    """
    p = np.clip(p, EPSILON, 1)
    q = np.clip(q, EPSILON, 1)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


# Using alibi (specialized library for drift detection)
# from alibi_detect.cd import CVMDrift
# import numpy as np

# Initialize the detector
# cvm_drift = CVMDrift(train_data, p_val=0.05)
# preds = cvm_drift.predict(test_data)  # Detect drift
