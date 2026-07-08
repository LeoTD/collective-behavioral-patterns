import numpy as np
import scipy.signal

from estimators import base


def estimate_hurst(
  time_series: list[float], params: base.EstimatorParams | None = None
) -> tuple[float, base.EstimatorResultDetails | None]:
  """
  Estimates the Hurst exponent using the periodogram method.
  """
  # 1. Compute Periodogram
  freqs, Pxx = scipy.signal.periodogram(time_series)

  # 2. Log-Log Regression
  # Exclude DC component (freq=0)
  freqs = freqs[1:]
  Pxx = Pxx[1:]

  # Filter out zeros to avoid log(0)
  valid = Pxx > 0
  freqs = freqs[valid]
  Pxx = Pxx[valid]

  if len(freqs) < 2:
    return 0.5, None

  log_freqs = np.log(freqs)
  log_power = np.log(Pxx)

  # Slope of log(Pxx) vs log(freq) is -beta
  # For fGn (stationary increments), beta = 2H - 1
  # Spectral density S(f) ~ |f|^(-beta)
  # log(S) = -beta * log(f) + C

  slope, _ = np.polyfit(log_freqs, log_power, 1)
  beta = -slope

  # relation: beta = 2H - 1  =>  2H = beta + 1 => H = (beta + 1) / 2
  # if -1 < beta < 1 (0 < H < 1)
  H = (beta + 1) / 2.0

  return float(np.clip(H, 0.0, 1.0)), None
