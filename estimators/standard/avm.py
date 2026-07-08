from typing import List, Optional, Tuple

import numpy as np

from estimators import base


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using the Absolute Value Method (AVM).
  Robust for cumulative series (random walks).
  """
  series = np.array(time_series)
  N = len(series)
  max_window = N // 4

  if max_window < 4:
    return 0.5, None

  # Window sizes (logarithmic spacing could be better, but linear is
  # fine for basic)
  # Using unique integer window sizes
  min_window = 2

  # Generate ~20 points log-spaced
  window_sizes = np.unique(
    np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
  )
  # Remove duplicates and sort
  window_sizes = window_sizes[window_sizes >= min_window]

  if len(window_sizes) < 3:
    return 0.5, None

  abs_moments = []

  for m in window_sizes:
    # Truncate series to make it evenly divisible by m
    length = N - (N % m)
    truncated = series[:length]

    # Reshape into blocks and calculate the mean of each block
    blocks = truncated.reshape(-1, m)

    # Calculate the absolute first moment of the aggregated series.
    # This implementation follows the standard AVM definition:
    # 1. Aggregate series (block means).
    # 2. Compute mean of absolute values of these block means.
    block_means = np.mean(blocks, axis=1)
    mean_abs_moment = np.mean(np.abs(block_means))
    # Avoid log(0) if the signal is constant or blocks sum to exactly 0
    if mean_abs_moment == 0:
      mean_abs_moment = np.float64(1e-12)
    abs_moments.append(mean_abs_moment)

  # Log-log regression
  # Relation: Moment ~ m^(H - 1)

  log_m = np.log(window_sizes)
  log_moments = np.log(abs_moments)

  slope, _ = np.polyfit(log_m, log_moments, 1)

  # H = slope + 1
  H = slope + 1.0

  return float(np.clip(H, 0.0, 1.0)), None
