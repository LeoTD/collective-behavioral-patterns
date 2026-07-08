from typing import List, Optional, Tuple

import numpy as np
import pywt  # type: ignore

from estimators import base


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using Wavelet transform (Discrete).
  """
  series = np.array(time_series)

  # Choose wavelet
  wavelet = "db4"
  mode = "per"

  # Decompose using max level
  max_level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
  if max_level < 3:
    return 0.5, None

  coeffs = pywt.wavedec(series, wavelet, mode=mode, level=max_level)

  # Detail coefficients are coeffs[1:] (coeffs[0] is approx)
  # Standard relation for fGn: Variance ~ scale^(2H-1)

  variances = []
  levels = []

  # PyWavelets returns [cA_n, cD_n, cD_n-1, ..., cD_1]
  # where cD_n is detail at level n (coarsest scale).
  # cD_1 is detail at level 1 (finest scale).

  # We evaluate slope of log2(Var) vs level j.
  # For fGn: Slope = 2H - 1.

  for i in range(1, len(coeffs)):
    # Calculate variance of detail coefficients at this level
    detail = coeffs[i]
    var = np.var(detail)

    # Scale level mapping:
    # PyWavelets coeffs: [cA_n, cD_n, cD_n-1, ..., cD_1]
    # i=1 -> cD_n (Level n)
    # i=len-1 -> cD_1 (Level 1)
    # level = len(coeffs) - i

    if var > 0:
      variances.append(var)
      levels.append(len(coeffs) - i)

  if len(levels) < 3:
    return 0.5, None

  log_vars = np.log2(variances)
  log_scales = np.array(levels)

  # Slope ~ 2H - 1
  slope, _ = np.polyfit(log_scales, log_vars, 1)
  # H = (slope + 1) / 2

  H = (slope + 1) / 2.0

  return float(np.clip(H, 0.0, 1.0)), None
