from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import periodogram

from estimators import base


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using Whittle's approximate Maximum Likelihood
  Estimation in the frequency domain.
  Assumes fGn (fractional Gaussian noise) behavior.
  """
  series = np.array(time_series)

  # Compute periodogram
  freqs, Pxx = periodogram(series)

  # Exclude 0 frequency
  freqs = freqs[1:]
  Pxx = Pxx[1:]

  # Filter out potential zeros
  valid = Pxx > 0
  freqs = freqs[valid]
  Pxx = Pxx[valid]

  if len(freqs) < 2:
    return 0.5, None

  # Whittle Likelihood Approximation
  # Spectral density f(lambda) ~ |lambda|^(1-2H)
  def negative_log_likelihood(H: float) -> float:
    # Model spectrum (unscaled)
    # Note: Using small epsilon or managing frequency range to avoid numerical
    # issues
    exponent = 1 - 2 * H
    spectral_density = freqs**exponent

    # Whittle Profile Likelihood (simplified):
    # L(H) = - sum(log S_shape) - N * log( mean(I / S_shape) )

    sum_log_S = np.sum(np.log(spectral_density))
    scaled_periodogram = Pxx / spectral_density
    mean_scaled = np.mean(scaled_periodogram)

    # We want to enable minimization of NEGATIVE likelihood
    # So we return: sum_log_S + N * log(mean_scaled)
    N = len(freqs)
    return float(sum_log_S + N * np.log(mean_scaled))

  # Optimize H in [0.01, 0.99]
  res = minimize_scalar(
    negative_log_likelihood, bounds=(0.01, 0.99), method="bounded"
  )

  if res.success:
    return float(res.x), None

  return 0.5, None
