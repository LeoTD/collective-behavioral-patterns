from typing import List, Optional, Tuple

import antropy as ant  # type: ignore
import numpy as np

from estimators import base
from estimators.hdfc import hdfc_types


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using Higuchi Fractal Dimension (HFD).
  H = 2 - D for cumulative processes (like fBm).
  """
  series = np.array(time_series)

  try:
    # Calculate Higuchi Fractal Dimension
    k_max = 10
    if params:
      if isinstance(params, hdfc_types.HFDParameters):
        k_max = params.k_max
      elif isinstance(params, dict):
        k_max = params.get("k_max", 10)
      elif hasattr(params, "k_max"):
        k_max = getattr(params, "k_max")

    D = ant.higuchi_fd(series, kmax=k_max)

    # H = 2 - D
    # Theoretical relationship for cumulative processes (like fBm).

    H = 2.0 - D
    return float(np.clip(H, 0.0, 1.0)), None
  except Exception:
    return 0.5, None
