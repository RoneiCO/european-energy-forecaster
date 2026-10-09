import numpy as np
from numpy.typing import ArrayLike


def bootstrap_ratio(
    numerator: ArrayLike, denominator: ArrayLike, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """95% interval for sum(numerator) / sum(denominator), resampling whole days.

    Hourly errors within a day move together (one windy day, one spike), so the
    day, not the hour, is the unit that gets resampled.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape:
        raise ValueError("numerator and denominator must cover the same days")
    rng = np.random.default_rng(seed)
    days = rng.integers(
        0, len(num), size=(n_boot, len(num))
    )  # n_boot rows of resampled day indices
    ratios = num[days].sum(axis=1) / den[days].sum(axis=1)
    low, mid, high = np.percentile(ratios, [2.5, 50, 97.5])
    return float(low), float(mid), float(high)
