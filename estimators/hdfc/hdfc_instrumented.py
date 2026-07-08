from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA  # type: ignore
from sklearn.linear_model import (  # type: ignore
  LinearRegression,
  RANSACRegressor,
)
from sklearn.neighbors import NearestNeighbors  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from estimators import base, utils
from estimators.hdfc import hdfc_types


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:
  """
  Estimates the Hurst coefficient using a "Hyper-Dimensional Fractal
  Consensus" method.
  (Instrumented Version)
  """
  # Cast base params to HDFCParameters
  if params is None:
    hdfc_params = hdfc_types.HDFCParameters()
  elif isinstance(params, hdfc_types.HDFCParameters):
    hdfc_params = params
  else:
    # Fallback to default HDFCParameters
    hdfc_params = hdfc_types.HDFCParameters()

  ts = np.asarray(time_series)
  timer = utils.ExecutionTimer()

  with timer.measure("total"):
    # Fallback for short series
    if len(time_series) < 100:
      with timer.measure("original"):
        h_original = _estimate_hurst_rs(time_series, hdfc_params)

      # Return simple result for short series
      # Populate minimal fields for HDFCResultDetails
      return float(h_original), hdfc_types.HDFCResultDetails(
        rs_h=float(h_original),
        svd_h=None,
        svd_quality=None,
        surrogates_h_mean=None,
        surrogates_h_std=None,
        reordered_h=None,
        consensus=hdfc_types.ConsensusDetails(
          stability_factor=0.0,
          consistency_surr=0.0,
          consistency_reordered=0.0,
          consistency_svd=0.0,
          potentials=[],
          weights=[],
          sigma_h=0.0,
          raw_h=0.0,
        ),
        stability_factor=0.0,
        timing=hdfc_types.TimerDetails(timings=timer.timings),
      )

    # 1. Original Analysis
    h_original = 0.5
    if hdfc_params.rs.enabled:
      with timer.measure("original"):
        h_original = _estimate_hurst_rs(time_series, hdfc_params)

    # --- Surrogate Generation and Ensemble Averaging ---
    h_surrogate_mean = None
    h_surrogate_std = None

    if hdfc_params.surrogates.enabled:
      with timer.measure("surrogates"):
        h_surrogates = []

        fft_original = np.fft.rfft(ts)
        amplitudes = np.abs(fft_original)
        phases = np.angle(fft_original)
        N = len(ts)

        for _ in range(hdfc_params.surrogates.surrogate_count):
          epsilon = 1e-6
          start_idx = 1
          end_idx = len(amplitudes)
          if N % 2 == 0:
            end_idx -= 1

          if start_idx >= end_idx:
            continue

          amp_for_weighting = amplitudes[start_idx:end_idx]
          phase_noise_std = hdfc_params.surrogates.surrogate_alpha / (
            amp_for_weighting + epsilon
          )

          gaussian_noise = np.random.normal(scale=phase_noise_std)

          p_tunnel = hdfc_params.surrogates.surrogate_p_tunnel
          tunneling_mask = np.random.rand(len(amp_for_weighting)) < p_tunnel

          tunnel_phases = np.random.choice(
            [
              np.pi / 2,
              np.pi,
              3 * np.pi / 2,
              -np.pi / 2,
              -np.pi,
              -3 * np.pi / 2,
            ],
            size=int(np.sum(tunneling_mask)),
          )

          combined_noise = gaussian_noise
          combined_noise[tunneling_mask] = tunnel_phases

          perturbed_phases = np.copy(phases)
          perturbed_phases[start_idx:end_idx] += combined_noise

          fft_surrogate = amplitudes * np.exp(1j * perturbed_phases)
          surrogate_ts = np.fft.irfft(fft_surrogate, n=N)

          h_s = _estimate_hurst_rs(surrogate_ts.tolist(), hdfc_params)
          h_surrogates.append(h_s)

        if not h_surrogates:
          h_surrogate_mean = h_original
          h_surrogate_std = 0.0
        else:
          h_surrogate_mean = float(np.mean(h_surrogates))
          h_surrogate_std = float(np.std(h_surrogates))

    # --- Temporal Wormhole Shuffling ---
    h_wormhole = None
    if hdfc_params.reordered.enabled:
      with timer.measure("reordered"):
        wormhole_ts = _create_wormhole_series(time_series, hdfc_params)
        h_wormhole = _estimate_hurst_rs(wormhole_ts, hdfc_params)

    # --- SVD Spectral Scaling ---
    h_svd = None
    svd_quality = None
    if hdfc_params.svd.enabled:
      with timer.measure("svd"):
        h_svd, svd_quality = _estimate_hurst_svd(time_series, hdfc_params)

    # --- Hyper-Dimensional Fractal Consensus Aggregation ---
    with timer.measure("consensus"):
      ts_np_orig = np.asarray(time_series)

      mean_sq = np.mean(ts_np_orig**2)
      var_score = np.var(ts_np_orig) / (mean_sq + 1e-9) if mean_sq > 0 else 0.0
      var_score_normalized = np.clip(var_score / 0.5, 0.0, 1.0)

      # Check for near-constant signal to avoid precision loss in skew
      # calculation
      if var_score < 1e-12:
        skew_score = 0.0
      else:
        try:
          skew_score = np.abs(skew(ts_np_orig, nan_policy="omit"))
          if np.isnan(skew_score):
            skew_score = 0.0
        except Exception:
          skew_score = 0.0

      skew_score_normalized = np.clip(skew_score / 2.0, 0.0, 1.0)

      stability_factor = (1.0 - var_score_normalized) * (
        1.0 - skew_score_normalized
      )

      max_plausible_std = hdfc_params.consensus.consensus_max_plausible_std
      max_plausible_divergence = (
        hdfc_params.consensus.consensus_max_plausible_div
      )

      consistency_surrogate = 0.0
      if h_surrogate_std is not None:
        consistency_surrogate = np.exp(
          -h_surrogate_std / (max_plausible_std + 1e-9)
        )

      consistency_wormhole = 0.0
      if h_wormhole is not None:
        wormhole_divergence = np.abs(h_wormhole - h_original)
        consistency_wormhole = np.exp(
          -wormhole_divergence / (max_plausible_divergence + 1e-9)
        )

      consistency_svd = 0.0
      if h_svd is not None and svd_quality is not None:
        svd_divergence = np.abs(h_svd - h_original)
        consistency_svd = svd_quality * np.exp(
          -svd_divergence / (max_plausible_divergence + 1e-9)
        )

      base_potential_epsilon = hdfc_params.consensus.consensus_base_epsilon

      estimates = [h_original]
      potentials = [stability_factor + base_potential_epsilon]

      if h_surrogate_mean is not None:
        estimates.append(h_surrogate_mean)
        potentials.append(consistency_surrogate + base_potential_epsilon)

      if h_wormhole is not None:
        estimates.append(h_wormhole)
        potentials.append(consistency_wormhole + base_potential_epsilon)

      if h_svd is not None:
        estimates.append(h_svd)
        potentials.append(consistency_svd + base_potential_epsilon)

      estimates_np = np.array(estimates)
      potentials_np = np.array(potentials)

      overall_hurst_dispersion = np.std(estimates_np)
      sigma_h_coherence = (
        hdfc_params.consensus.consensus_sigma_base
        + overall_hurst_dispersion * hdfc_params.consensus.consensus_sigma_scale
      )
      _sigma_sq_safe = (sigma_h_coherence**2) * 2 + 1e-9

      diff_matrix_sq = (estimates_np[:, None] - estimates_np[None, :]) ** 2
      sum_sq_diffs = np.sum(diff_matrix_sq, axis=1)
      C_factors = np.exp(-sum_sq_diffs / _sigma_sq_safe)

      W_raw = potentials_np * C_factors
      sum_W = np.sum(W_raw) + 1e-9
      W_norm = W_raw / sum_W

      final_h = np.sum(W_norm * estimates_np)
      final_h = np.clip(final_h, 0.0, 1.0)

      consensus_details = hdfc_types.ConsensusDetails(
        stability_factor=float(stability_factor),
        consistency_surr=float(consistency_surrogate),
        consistency_reordered=float(consistency_wormhole),
        consistency_svd=float(consistency_svd),
        potentials=potentials_np.tolist(),
        weights=W_norm.tolist(),
        sigma_h=float(sigma_h_coherence),
        raw_h=float(np.sum(W_norm * estimates_np)),
      )

  details = hdfc_types.HDFCResultDetails(
    rs_h=float(h_original),
    svd_h=float(h_svd) if h_svd is not None else None,
    svd_quality=float(svd_quality) if svd_quality is not None else None,
    surrogates_h_mean=float(h_surrogate_mean)
    if h_surrogate_mean is not None
    else None,
    surrogates_h_std=float(h_surrogate_std)
    if h_surrogate_std is not None
    else None,
    reordered_h=float(h_wormhole) if h_wormhole is not None else None,
    consensus=consensus_details,
    stability_factor=float(stability_factor),
    timing=hdfc_types.TimerDetails(timings=timer.timings),
  )

  return float(final_h), details


def _create_wormhole_series(
  time_series: List[float], params: hdfc_types.HDFCParameters
) -> List[float]:
  segment_len = params.reordered.reordered_segment_len
  step = params.reordered.reordered_step

  ts_np = np.asarray(time_series)
  N = len(ts_np)

  if N < segment_len * 2:
    return time_series

  num_segments = (N - segment_len) // step + 1
  segments = np.array(
    [ts_np[i * step : i * step + segment_len] for i in range(num_segments)]
  )

  segment_stds = np.std(segments, axis=1)
  constant_segments_mask = segment_stds == 0

  if np.any(constant_segments_mask):
    segments[constant_segments_mask] += np.random.normal(
      0, 1e-6, size=segments[constant_segments_mask].shape
    )

  scaler = StandardScaler()
  scaled_segments = scaler.fit_transform(segments)

  n_components_pca = min(5, segment_len)

  if num_segments < n_components_pca:
    with np.errstate(divide="ignore", invalid="ignore"):
      means = np.mean(segments, axis=1)
      stds = np.std(segments, axis=1)
      skews = skew(segments, axis=1)
      skews = np.nan_to_num(skews, nan=0.0)

    features = np.column_stack(
      [
        (means - np.mean(means)) / (np.std(means) + 1e-9),
        (stds - np.mean(stds)) / (np.std(stds) + 1e-9),
        (skews - np.mean(skews)) / (np.std(skews) + 1e-9),
      ]
    )
  else:
    pca = PCA(n_components=n_components_pca, random_state=42)
    features = pca.fit_transform(scaled_segments)
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features = (features - features_mean) / (features_std + 1e-9)

  k = min(params.reordered.reordered_k_neighbors, num_segments - 1)
  if k < 1:
    return time_series

  nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(features)
  distances, indices = nn.kneighbors(features)

  path = []
  if not features.shape[0]:
    return time_series

  start_idx = np.argmin(np.sum(np.abs(features), axis=1))
  current_idx = start_idx
  visited = np.zeros(num_segments, dtype=bool)

  for _ in range(num_segments):
    path.append(current_idx)
    visited[current_idx] = True
    next_idx_found = False
    for neighbor_idx in indices[current_idx]:
      if not visited[neighbor_idx]:
        current_idx = neighbor_idx
        next_idx_found = True
        break

    if not next_idx_found:
      unvisited_indices = np.where(~visited)[0]
      if len(unvisited_indices) > 0:
        current_idx = unvisited_indices[0]
      else:
        break

  wormhole_ts = segments[path].flatten()
  return wormhole_ts.tolist()


def _estimate_hurst_svd(
  time_series: List[float], params: hdfc_types.HDFCParameters
) -> Tuple[float, float]:
  ts = np.asarray(time_series)
  N = len(ts)
  if N < params.svd.svd_min_window:
    return 0.5, 0.0

  L = int(np.clip(N / 4, 10, 64))
  h_est, quality, _, _ = _estimate_hurst_svd_single_L(ts, L, params)
  return h_est, quality


def _estimate_hurst_svd_single_L(
  ts: np.ndarray, L: int, params: hdfc_types.HDFCParameters
) -> Tuple[float, float, np.ndarray, np.ndarray]:
  N = len(ts)
  empty_res = (0.5, 0.0, np.array([]), np.array([]))

  if L < 8 or L >= N:
    return empty_res

  K = N - L + 1
  if K < 2:
    return empty_res

  X = np.array([ts[i : i + L] for i in range(K)])
  X_centered = X - np.mean(X, axis=1, keepdims=True)

  try:
    s = np.linalg.svd(X_centered, compute_uv=False)
  except np.linalg.LinAlgError:
    return empty_res

  if s[0] == 0:
    return empty_res
  s = s / s[0]

  num_sv = len(s)
  min_points_for_fit = 3
  k_start_candidates_0based = [1, 2, 3]

  best_h_est = 0.5
  best_r2 = 0.0
  best_log_k = np.array([])
  best_log_s = np.array([])

  for current_start_idx in k_start_candidates_0based:
    proportional_end_candidates_0based = [
      int(num_sv * 0.4),
      int(num_sv * 0.6),
      int(num_sv * 0.8),
      num_sv,
    ]

    k_end_candidates = sorted(
      list(
        set(
          [
            k
            for k in proportional_end_candidates_0based
            if k >= current_start_idx + min_points_for_fit and k <= num_sv
          ]
        )
      )
    )

    for current_end_idx in k_end_candidates:
      k_vals = np.arange(current_start_idx, current_end_idx) + 1
      log_k = np.log(k_vals).reshape(-1, 1)
      log_s = np.log(s[current_start_idx:current_end_idx] + 1e-12)

      try:
        ransac = RANSACRegressor(
          LinearRegression(), random_state=42, min_samples=2
        )
        ransac.fit(log_k, log_s)

        inlier_mask = ransac.inlier_mask_
        if np.sum(inlier_mask) < min_points_for_fit:
          continue

        r2_current = ransac.estimator_.score(
          log_k[inlier_mask], log_s[inlier_mask]
        )

        if r2_current > best_r2:
          best_r2 = r2_current
          best_h_est = -ransac.estimator_.coef_[0]
      except (ValueError, IndexError):
        continue

  if best_r2 == 0.0:
    return empty_res

  return (
    np.clip(best_h_est, 0.0, 1.0),
    max(0.0, best_r2),
    best_log_k,
    best_log_s,
  )


def _estimate_hurst_rs(
  time_series: List[float], params: hdfc_types.HDFCParameters
) -> float:
  if len(time_series) < params.rs.rs_min_window:
    return 0.5

  ts_np = np.asarray(time_series)
  x_indices = np.arange(len(ts_np))
  trend_coeffs = np.polyfit(x_indices, ts_np, 1)
  detrended_ts = ts_np - np.polyval(trend_coeffs, x_indices)
  ts = np.diff(detrended_ts)
  N = len(ts)

  min_scale = params.rs.rs_min_scale
  max_scale = N // 2
  if max_scale <= min_scale:
    return 0.5

  scales = np.unique(
    np.logspace(
      np.log10(min_scale), np.log10(max_scale), num=params.rs.rs_scale_subsample
    ).astype(int)
  )

  n_values = []
  rs_values = []

  for n in scales:
    all_chunks_list = []
    N_ts_diff = len(ts)

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
    mean_centered_chunks = all_chunks - np.mean(
      all_chunks, axis=1, keepdims=True
    )
    z = np.cumsum(mean_centered_chunks, axis=1)
    R = np.max(z, axis=1) - np.min(z, axis=1)
    S = np.std(all_chunks, axis=1)
    valid_indices = S != 0

    if np.any(valid_indices):
      avg_rs = np.mean(R[valid_indices] / S[valid_indices])
      if avg_rs > 0:
        n_values.append(n)
        rs_values.append(avg_rs)

  if len(n_values) < 3:
    return 0.5

  log_n_arr = np.log(np.array(n_values))
  log_rs_arr = np.log(np.array(rs_values))
  sample_weights = np.linspace(0.5, 1.5, len(log_n_arr))

  hurst_candidates = []
  fit_qualities = []
  total_log_n_range_overall = log_n_arr[-1] - log_n_arr[0]

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
      if not np.any(inlier_mask):
        continue

      inlier_log_n = log_n_arr[inlier_mask]
      inlier_log_rs = log_rs_arr[inlier_mask]
      inlier_X_poly = X_poly[inlier_mask]
      r_squared_overall = ransac.estimator_.score(inlier_X_poly, inlier_log_rs)

      if len(inlier_log_n) < 2:
        continue

      sort_indices = np.argsort(inlier_log_n)
      inlier_log_n_sorted = inlier_log_n[sort_indices]

      local_hurst_candidates = []
      local_weights = []

      num_local_segments = min(len(inlier_log_n_sorted) // 2, 4)
      if num_local_segments < 1:
        num_local_segments = 1

      current_min_log_n_inliers = inlier_log_n_sorted[0]
      current_max_log_n_inliers = inlier_log_n_sorted[-1]

      x_min_features = np.array(
        [current_min_log_n_inliers**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)
      x_max_features = np.array(
        [current_max_log_n_inliers**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)

      if num_local_segments == 1:
        pred_rs_min = ransac.estimator_.predict(x_min_features)
        pred_rs_max = ransac.estimator_.predict(x_max_features)
        if current_max_log_n_inliers > current_min_log_n_inliers + 1e-9:
          local_h = (pred_rs_max - pred_rs_min) / (
            current_max_log_n_inliers - current_min_log_n_inliers
          )
          local_hurst_candidates.append(local_h[0])
          local_weights.append(len(inlier_log_n_sorted))
      else:
        for i in range(num_local_segments):
          idx_start = len(inlier_log_n_sorted) * i // num_local_segments
          idx_end = len(inlier_log_n_sorted) * (i + 1) // num_local_segments

          if idx_end - idx_start < 2:
            if idx_start > 0:
              idx_start -= 1
            if idx_end < len(inlier_log_n_sorted):
              idx_end += 1
            if idx_end - idx_start < 2:
              continue

          seg_log_n = inlier_log_n_sorted[idx_start:idx_end]
          seg_min = seg_log_n[0]
          seg_max = seg_log_n[-1]
          if seg_max - seg_min < 1e-9:
            continue

          x_seg_min = np.array(
            [seg_min**d for d in range(degree, 0, -1)]
          ).reshape(1, -1)
          x_seg_max = np.array(
            [seg_max**d for d in range(degree, 0, -1)]
          ).reshape(1, -1)

          p_min = ransac.estimator_.predict(x_seg_min)
          p_max = ransac.estimator_.predict(x_seg_max)

          l_h = (p_max - p_min) / (seg_max - seg_min)
          local_hurst_candidates.append(l_h[0])
          local_weights.append(len(seg_log_n))

      if not local_hurst_candidates:
        continue

      current_hurst = np.average(local_hurst_candidates, weights=local_weights)

      if total_log_n_range_overall > 1e-9 and len(inlier_log_n_sorted) > 1:
        inlier_spread_ratio = (
          current_max_log_n_inliers - current_min_log_n_inliers
        ) / total_log_n_range_overall
      else:
        inlier_spread_ratio = 0.0

      quality = max(0.0, r_squared_overall) * inlier_spread_ratio
      hurst_candidates.append(current_hurst)
      fit_qualities.append(quality)

    except ValueError:
      continue

  if not hurst_candidates:
    return 0.5

  hurst_candidates_np = np.array(hurst_candidates)
  fit_qualities_np = np.array(fit_qualities)
  valid_mask = fit_qualities_np > 1e-9

  if not np.any(valid_mask):
    return float(np.clip(np.median(hurst_candidates_np), 0.0, 1.0))

  valid_hursts = hurst_candidates_np[valid_mask]
  valid_weights = fit_qualities_np[valid_mask]
  final_hurst = np.average(valid_hursts, weights=valid_weights)

  return float(np.clip(final_hurst, 0.0, 1.0))
