from typing import List, Optional, Tuple

import nolds  # type: ignore
import numpy as np

from estimators import base


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using Detrended Fluctuation Analysis (DFA).
  Uses `nolds` library.
  """
  series = np.array(time_series)

  try:
    # Check for sufficient length
    if len(series) < 50:
      return 0.5, None

    h = nolds.dfa(series)
    return float(np.clip(h, 0.0, 1.0)), None
  except Exception:
    return 0.5, None
