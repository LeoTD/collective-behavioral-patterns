from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import (  # type: ignore
  LinearRegression,
  RANSACRegressor,
)

from estimators import base
from estimators.hdfc import hdfc_types


def _get_rs_params(
  params: Optional[base.EstimatorParams],
) -> hdfc_types.RSParams:
  if params is None:
    return hdfc_types.RSParams()
  if isinstance(params, hdfc_types.RSParams):
    return params
  if isinstance(params, hdfc_types.HDFCParameters):
    return params.rs
  # Fallback
  if hasattr(params, "rs_min_scale") and hasattr(params, "rs_scale_subsample"):
    return params  # type: ignore
  return hdfc_types.RSParams()




def estimate_hurst(
  time_series: Union[List[float], np.ndarray],
  params: Optional[base.EstimatorParams] = None,
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst exponent using Rescaled Range (R/S) analysis.
  Uses a robust RANSAC-based regression method (similar to HDFC's internal R/S).
  """
  rs_params = _get_rs_params(params)

  if len(time_series) < rs_params.rs_min_window:
    return 0.5, None

  ts_np = np.asarray(time_series)

  # Pre-processing: Detrend and Diff if requested (for Random Walk inputs)
  if rs_params.rs_diff:
    x_indices = np.arange(len(ts_np))
    trend_coeffs = np.polyfit(x_indices, ts_np, 1)
    detrended_ts = ts_np - np.polyval(trend_coeffs, x_indices)
    # Diffing converts Walk to Noise (increments)
    ts = np.diff(detrended_ts)
  else:
    # Assume input is already Noise (increments)
    ts = ts_np

  N = len(ts)

  min_scale = rs_params.rs_min_scale
  if min_scale < 2:
    min_scale = 2

  max_scale = N // 2
  if max_scale <= min_scale:
    return 0.5, None

  scales = np.unique(
    np.logspace(
      np.log10(min_scale), np.log10(max_scale), num=rs_params.rs_scale_subsample
    ).astype(int)
  )
  scales = scales[scales >= 2]  # Ensure valid

  n_values = []
  rs_values = []

  for n in scales:
    all_chunks_list = []
    N_ts_diff = len(ts)

    # Overlapping windows logic from HDFC
    if N_ts_diff < 200:
      num_offsets = 1
    elif N_ts_diff < 400:
      num_offsets = 3
    else:
      num_offsets = 5

    for i in range(num_offsets):
      offset = (i * n) // num_offsets
      if N - offset < n:
        continue

      series_subset = ts[offset:]
      num_chunks = len(series_subset) // n
      if num_chunks > 0:
        chunks = series_subset[: num_chunks * n].reshape(num_chunks, n)
        all_chunks_list.append(chunks)

    if not all_chunks_list:
      continue

    all_chunks = np.vstack(all_chunks_list)
    # R/S Calculation
    mean_centered_chunks = all_chunks - np.mean(
      all_chunks, axis=1, keepdims=True
    )
    z = np.cumsum(mean_centered_chunks, axis=1)
    R = np.max(z, axis=1) - np.min(z, axis=1)
    S = np.std(all_chunks, axis=1, ddof=1)

    valid_indices = S != 0

    if np.any(valid_indices):
      avg_rs = np.mean(R[valid_indices] / S[valid_indices])
      if avg_rs > 0:
        n_values.append(n)
        rs_values.append(avg_rs)

  if len(n_values) < 3:
    return 0.5, None

  # Robust Regression (RANSAC with Polynomial Features)
  log_n_arr = np.log(np.array(n_values))
  log_rs_arr = np.log(np.array(rs_values))
  sample_weights = np.linspace(0.5, 1.5, len(log_n_arr))

  hurst_candidates = []
  fit_qualities = []
  total_log_n_range_overall = log_n_arr[-1] - log_n_arr[0]

  # Try polynomial degrees 1, 2, 3 (HDFC-style)
  for degree in [1, 2, 3]:
    if len(n_values) < degree + 1:
      continue

    X_poly_features = []
    for d in range(degree, 0, -1):
      X_poly_features.append(log_n_arr**d)
    X_poly = np.column_stack(X_poly_features)

    try:
      ransac = RANSACRegressor(LinearRegression(), random_state=42)
      ransac.fit(X_poly, log_rs_arr, sample_weight=sample_weights)

      inlier_mask = ransac.inlier_mask_
      if inlier_mask is None or not np.any(inlier_mask):
        continue

      inlier_log_n = log_n_arr[inlier_mask]
      inlier_log_rs = log_rs_arr[inlier_mask]
      inlier_X_poly = X_poly[inlier_mask]

      # Use score on inliers
      r_squared_overall = ransac.estimator_.score(inlier_X_poly, inlier_log_rs)

      if len(inlier_log_n) < 2:
        continue

      # Local Hurst estimation

      sort_indices = np.argsort(inlier_log_n)
      inlier_log_n_sorted = inlier_log_n[sort_indices]

      local_hurst_candidates = []
      local_weights = []

      num_local_segments = min(len(inlier_log_n_sorted) // 2, 4)
      if num_local_segments < 1:
        num_local_segments = 1

      current_min = inlier_log_n_sorted[0]
      current_max = inlier_log_n_sorted[-1]

      x_min_f = np.array(
        [current_min**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)
      x_max_f = np.array(
        [current_max**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)

      if num_local_segments == 1:
        p_min = ransac.estimator_.predict(x_min_f)
        p_max = ransac.estimator_.predict(x_max_f)
        if current_max > current_min + 1e-9:
          global_slope = (p_max - p_min) / (current_max - current_min)
          local_hurst_candidates.append(global_slope[0])
          local_weights.append(len(inlier_log_n_sorted))
      else:
        for i in range(num_local_segments):
          s_idx = len(inlier_log_n_sorted) * i // num_local_segments
          e_idx = len(inlier_log_n_sorted) * (i + 1) // num_local_segments

          if e_idx - s_idx < 2:
            if s_idx > 0:
              s_idx -= 1
            if e_idx < len(inlier_log_n_sorted):
              e_idx += 1
            if e_idx - s_idx < 2:
              continue

          seg_log_n = inlier_log_n_sorted[s_idx:e_idx]
          s_min = seg_log_n[0]
          s_max = seg_log_n[-1]

          if s_max - s_min < 1e-9:
            continue

          x_s_min = np.array([s_min**d for d in range(degree, 0, -1)]).reshape(
            1, -1
          )
          x_s_max = np.array([s_max**d for d in range(degree, 0, -1)]).reshape(
            1, -1
          )

          p_s_min = ransac.estimator_.predict(x_s_min)
          p_s_max = ransac.estimator_.predict(x_s_max)

          l_h = (p_s_max - p_s_min) / (s_max - s_min)
          local_hurst_candidates.append(l_h[0])
          local_weights.append(len(seg_log_n))

      if not local_hurst_candidates:
        continue

      current_hurst = np.average(local_hurst_candidates, weights=local_weights)

      if total_log_n_range_overall > 1e-9:
        inlier_spread_ratio = (
          current_max - current_min
        ) / total_log_n_range_overall
      else:
        inlier_spread_ratio = 0.0

      quality = max(0.0, r_squared_overall) * inlier_spread_ratio
      hurst_candidates.append(current_hurst)
      fit_qualities.append(quality)

    except ValueError:
      continue

  if not hurst_candidates:
    return 0.5, None

  hurst_candidates_np = np.array(hurst_candidates)
  fit_qualities_np = np.array(fit_qualities)
  valid_mask = fit_qualities_np > 1e-9

  if not np.any(valid_mask):
    final_hurst = np.median(hurst_candidates_np)
  else:
    final_hurst = np.average(
      hurst_candidates_np[valid_mask], weights=fit_qualities_np[valid_mask]
    )

  return float(np.clip(final_hurst, 0.0, 1.0)), None
