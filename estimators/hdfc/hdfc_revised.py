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
from estimators.standard import rs


def estimate_hurst(
  time_series: List[float], params: Optional[base.EstimatorParams] = None
) -> Tuple[float, Optional[base.EstimatorResultDetails]]:

  # Cast base params to HDFCParameters
  if params is None:
    hdfc_params = hdfc_types.HDFCParameters()
  elif isinstance(params, hdfc_types.HDFCParameters):
    hdfc_params = params
  else:
    hdfc_params = hdfc_types.HDFCParameters()

  ts = np.array(time_series)
  timer = utils.ExecutionTimer()

  with timer.measure("total"):
    # 1. Original Analysis (Fast Baseline).
    h_original = 0.5
    # Define dummy details for fallback
    dummy_consensus = hdfc_types.ConsensusDetails(
      stability_factor=0.0,
      consistency_surr=0.0,
      consistency_reordered=0.0,
      consistency_svd=0.0,
      potentials=[],
      weights=[],
      sigma_h=0.0,
      raw_h=0.5,
    )
    dummy_timer = hdfc_types.TimerDetails(timings={})

    # Fallback to standard R/S analysis for short time series.
    if len(ts) < 100:
      if hdfc_params.rs.enabled:
        with timer.measure("original"):
          h_original, _ = rs.estimate_hurst(ts, params=hdfc_params)

      return float(h_original), hdfc_types.HDFCResultDetails(
        rs_h=float(h_original),
        svd_h=None,
        svd_quality=None,
        surrogates_h_mean=None,
        surrogates_h_std=None,
        reordered_h=None,
        consensus=dummy_consensus,
        stability_factor=1.0,
        timing=dummy_timer,
      )

    # 1. Original Analysis (Fast Baseline).
    h_original = 0.5
    if hdfc_params.rs.enabled:
      with timer.measure("original"):
        h_original, _ = rs.estimate_hurst(ts, params=hdfc_params)

    # 2. SVD Spectral Scaling (Fast-ish, Good Variance Proxy).
    h_svd = None
    svd_quality = None
    early_exit_triggered = False

    if hdfc_params.svd.enabled:
      with timer.measure("svd"):
        h_svd, svd_quality = _estimate_hurst_svd(ts, hdfc_params)

        # Early Exit check
        if (
          svd_quality is not None
          and (1.0 - svd_quality) < hdfc_params.svd.svd_early_exit_threshold
        ):
          early_exit_triggered = True

    # 3. Expensive Estimators (Surrogates, Reordered).
    # Run only if NOT early exiting.
    h_surrogate_mean = None
    h_surrogate_std = None
    h_reordered = None

    if not early_exit_triggered:
      if hdfc_params.surrogates.enabled:
        with timer.measure("surrogates"):
          surrogates = _generate_phase_shuffled_surrogates(ts, hdfc_params)
          h_surrogates_list = [
            rs.estimate_hurst(s, params=hdfc_params)[0] for s in surrogates
          ]

          if not h_surrogates_list:
            h_surrogate_mean = h_original
            h_surrogate_std = 0.0
          else:
            h_surrogate_mean = float(np.mean(h_surrogates_list))
            h_surrogate_std = float(np.std(h_surrogates_list))

      if hdfc_params.reordered.enabled:
        with timer.measure("reordered"):
          reordered_ts = _generate_similarity_reordered_series(ts, hdfc_params)
          h_reordered, _ = rs.estimate_hurst(reordered_ts, params=hdfc_params)

    # 5. Weighted Consensus.
    with timer.measure("consensus"):
      consensus_h, consensus_details_obj = _compute_fractal_consensus(
        ts,
        h_original,
        h_surrogate_mean,
        h_surrogate_std,
        h_reordered,
        h_svd,
        svd_quality,
        hdfc_params,
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
    reordered_h=float(h_reordered) if h_reordered is not None else None,
    consensus=consensus_details_obj,
    stability_factor=consensus_details_obj.stability_factor,
    timing=hdfc_types.TimerDetails(timings=timer.timings),
  )

  return float(consensus_h), details


def _compute_fractal_consensus(
  ts: np.ndarray,
  h_original: float,
  h_surrogate_mean: Optional[float],
  h_surrogate_std: Optional[float],
  h_reordered: Optional[float],
  h_svd: Optional[float],
  svd_quality: Optional[float],
  params: hdfc_types.HDFCParameters,
) -> Tuple[float, hdfc_types.ConsensusDetails]:
  """Computes the consensus of fractal estimates."""

  # 1. Dynamic Signal Characterization.
  mean_sq = np.mean(ts**2)
  var_score = float(np.var(ts) / (mean_sq + 1e-9) if mean_sq > 0 else 0.0)
  var_score_norm = float(np.clip(var_score / 0.5, 0.0, 1.0))

  if np.nanstd(ts) < 1e-9:
    skew_score = 0.0
  else:
    skew_score = float(np.abs(skew(ts, nan_policy="omit")))
    if np.isnan(skew_score):
      skew_score = 0.0
  skew_score_norm = float(np.clip(skew_score / 2.0, 0.0, 1.0))

  stability_factor = (1.0 - var_score_norm) * (1.0 - skew_score_norm)

  # 2. Confidence/Consistency Scores.
  max_plausible_std = params.consensus.consensus_max_plausible_std
  max_plausible_div = params.consensus.consensus_max_plausible_div

  consistency_surr = 0.0
  if h_surrogate_std is not None:
    consistency_surr = np.exp(-h_surrogate_std / (max_plausible_std + 1e-9))

  consistency_reordered = 0.0
  if h_reordered is not None:
    reordered_div = np.abs(h_reordered - h_original)
    consistency_reordered = np.exp(-reordered_div / (max_plausible_div + 1e-9))

  consistency_svd = 0.0
  if h_svd is not None and svd_quality is not None:
    svd_div = np.abs(h_svd - h_original)
    consistency_svd = svd_quality * np.exp(
      -svd_div / (max_plausible_div + 1e-9)
    )

  # 3. Potential Field Aggregation.
  base_epsilon = params.consensus.consensus_base_epsilon

  # Collect valid estimates and their potentials
  valid_estimates = [h_original]
  valid_potentials = [stability_factor + base_epsilon]

  if h_surrogate_mean is not None and h_surrogate_std is not None:
    valid_estimates.append(h_surrogate_mean)
    valid_potentials.append(consistency_surr + base_epsilon)

  if h_reordered is not None:
    valid_estimates.append(h_reordered)
    valid_potentials.append(consistency_reordered + base_epsilon)

  if h_svd is not None and svd_quality is not None:
    valid_estimates.append(h_svd)
    valid_potentials.append(consistency_svd + base_epsilon)

  estimates = np.array(valid_estimates)
  potentials = np.array(valid_potentials)

  # Dynamic coherence scale.
  sigma_h = (
    params.consensus.consensus_sigma_base
    + np.std(estimates) * params.consensus.consensus_sigma_scale
  )
  sigma_sq = (sigma_h**2) * 2 + 1e-9

  # Pairwise distances.
  diff_sq = (estimates[:, None] - estimates[None, :]) ** 2
  sum_diff_sq = np.sum(diff_sq, axis=1)

  # Coherence factors.
  c_factors = np.exp(-sum_diff_sq / sigma_sq)

  # Final weighting.
  weights = potentials * c_factors
  weights_norm = weights / (np.sum(weights) + 1e-9)

  final_h = np.sum(weights_norm * estimates)
  final_h = float(np.clip(final_h, 0.0, 1.0))

  details = hdfc_types.ConsensusDetails(
    stability_factor=float(stability_factor),
    consistency_surr=float(consistency_surr),
    consistency_reordered=float(consistency_reordered),
    consistency_svd=float(consistency_svd),
    potentials=potentials.tolist(),
    weights=weights_norm.tolist(),
    sigma_h=float(sigma_h),
    raw_h=float(final_h),
  )

  return final_h, details


def _generate_phase_shuffled_surrogates(
  ts: np.ndarray, params: hdfc_types.HDFCParameters
) -> List[np.ndarray]:
  n = len(ts)
  fft = np.fft.rfft(ts)
  amplitudes = np.abs(fft)
  phases = np.angle(fft)

  surrogates = []

  start_idx = 1
  end_idx = len(amplitudes)
  if n % 2 == 0:
    end_idx -= 1

  if start_idx >= end_idx:
    return [ts] * params.surrogates.surrogate_count

  amp_weight = amplitudes[start_idx:end_idx]
  epsilon = 1e-6
  alpha = params.surrogates.surrogate_alpha
  phase_noise_std = alpha / (amp_weight + epsilon)

  for _ in range(params.surrogates.surrogate_count):
    noise = np.random.normal(scale=phase_noise_std)

    p_tunnel = params.surrogates.surrogate_p_tunnel
    tunnel_mask = np.random.rand(len(amp_weight)) < p_tunnel

    if np.any(tunnel_mask):
      tunnel_phases = np.random.choice(
        [np.pi / 2, np.pi, 3 * np.pi / 2, -np.pi / 2, -np.pi, -3 * np.pi / 2],
        size=int(np.sum(tunnel_mask)),
      )
      noise[tunnel_mask] = tunnel_phases

    new_phases = phases.copy()
    new_phases[start_idx:end_idx] += noise

    fft_surr = amplitudes * np.exp(1j * new_phases)
    surr_ts = np.fft.irfft(fft_surr, n=n)
    surrogates.append(surr_ts)

  return surrogates


def _generate_similarity_reordered_series(
  ts: np.ndarray, params: hdfc_types.HDFCParameters
) -> np.ndarray:
  n = len(ts)
  segment_len = params.reordered.reordered_segment_len
  step = params.reordered.reordered_step

  if n < segment_len * 2:
    return ts

  shape = ((n - segment_len) // step + 1, segment_len)
  strides = (ts.strides[0] * step, ts.strides[0])
  segments = np.lib.stride_tricks.as_strided(
    ts, shape=shape, strides=strides
  ).copy()

  num_segments = segments.shape[0]

  stds = np.std(segments, axis=1, keepdims=True)
  no_var_mask = stds.flatten() == 0
  if np.any(no_var_mask):
    segments[no_var_mask] += np.random.normal(
      0, 1e-6, size=(np.sum(no_var_mask), segment_len)
    )

  scaler = StandardScaler()
  scaled_segments = scaler.fit_transform(segments)

  n_components = min(5, segment_len)
  if num_segments < n_components:
    m = np.mean(segments, axis=1)
    s = np.std(segments, axis=1)
    k = skew(segments, axis=1)
    k = np.nan_to_num(k)
    features = np.column_stack([m, s, k])
    features = StandardScaler().fit_transform(features)
  else:
    pca = PCA(n_components=n_components, random_state=42)
    features = pca.fit_transform(scaled_segments)
    features = StandardScaler().fit_transform(features)

  k_neighbors = min(params.reordered.reordered_k_neighbors, num_segments - 1)
  if k_neighbors < 1:
    return ts

  nn = NearestNeighbors(n_neighbors=k_neighbors).fit(features)
  distances, indices = nn.kneighbors(features)

  path = np.zeros(num_segments, dtype=int)
  visited = np.zeros(num_segments, dtype=bool)

  current_idx = np.argmin(np.sum(np.abs(features), axis=1))

  for i in range(num_segments):
    path[i] = current_idx
    visited[current_idx] = True

    found = False
    for nbr in indices[current_idx]:
      if not visited[nbr]:
        current_idx = nbr
        found = True
        break

    if not found and i < num_segments - 1:
      unvisited = np.where(~visited)[0]
      if len(unvisited) > 0:
        current_idx = unvisited[0]
      else:
        break
  return segments[path].flatten()


def _estimate_hurst_svd(
  time_series: np.ndarray, params: hdfc_types.HDFCParameters
) -> Tuple[Optional[float], Optional[float]]:
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
          best_h_est = -ransac.estimator_.coef_[0] - 0.5
          best_log_k = log_k[inlier_mask].flatten()
          best_log_s = log_s[inlier_mask].flatten()
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
