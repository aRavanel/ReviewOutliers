import numpy as np

# module imports
from src.config import logger

# ==========================================================================
# Utils functions
# ==========================================================================

from src.utils.maths.divergence_metrics import (
    psi,
    ks_test_score,
    kl_divergence,
    js_divergence,
)


# ==========================================================================
# Exported functions
# ==========================================================================


def distribution_shift_scoring(samples: np.ndarray, method: str = "psi") -> float:
    """
    Calculates the distribution shift score using the specified method.

    Parameters:
    - samples (np.ndarray): Array of sample data.
    - method (str): Method to use for scoring ('psi' or 'ks'). Default is 'psi'.

    Returns:
    - float: Distribution shift score.
    """
    logger.debug("calling distribution_shift_scoring")

    if method not in {"psi", "ks"}:
        raise ValueError("Method must be either 'psi' or 'ks'.")

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
