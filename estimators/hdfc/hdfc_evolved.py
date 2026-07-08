# mypy: ignore-errors
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

from estimators.base import EstimatorParams, EstimatorResultDetails


def estimate_hurst(
  time_series: List[float],
  params: Optional[EstimatorParams] = None,
) -> Tuple[float, Optional[EstimatorResultDetails]]:
  """
  Estimates the Hurst coefficient using a "Hyper-Dimensional Fractal Consensus" method.
  This involves three parallel analyses:
  1.  Direct R/S analysis on the original series.
  2.  Analysis of "Phase-Space Quantum Tunneling Surrogates", which are versions of the
      series with perturbed phase information, testing robustness to noise.
  3.  Analysis of a "Temporal Wormhole" series, a version shuffled based on
      structural similarity, testing robustness to temporal ordering.

  The final estimate is a gravitationally-inspired, dynamically weighted consensus
  of these three perspectives, aiming for extreme robustness against noise,
  non-stationarity, and multi-fractal signal variations.
  """

  # Calculate Hurst for the original, unaltered time series.
  h_original = _estimate_hurst_core(time_series)

  # For very short series, the FFT-based surrogate generation is unstable
  # and computationally not worthwhile. Return the direct estimate.
  if len(time_series) < 100:
    return h_original, None

  # --- Surrogate Generation and Ensemble Averaging ---

  num_surrogates = 5
  # Small perturbation factor for phase noise. This value is chosen to be small
  # enough to preserve the overall signal structure while introducing plausible variations.
  alpha = 0.05

  h_surrogates = []
  ts_np = np.asarray(time_series)
  N = len(ts_np)

  # Use rfft for real-valued input; it's more efficient and handles symmetry.
  fft_original = np.fft.rfft(ts_np)
  amplitudes = np.abs(fft_original)
  phases = np.angle(fft_original)

  for _ in range(num_surrogates):
    # Create phase noise with std inversely proportional to frequency amplitude.
    # This perturbs noisy (low-amplitude) components more than strong ones.
    # A slightly larger epsilon for numerical stability when amplitudes are near zero.
    # This prevents extremely large phase perturbations for components that might
    # be negligible or pure noise, ensuring more meaningful surrogate generation.
    epsilon = 1e-6  # Increased from 1e-9 for better stability with very small amplitudes

    # Determine indices for phase perturbation.
    # We don't perturb the DC component (index 0) or the Nyquist component
    # (last index if N is even), as their phases are fixed (0 or pi) and generally
    # should not be perturbed randomly without affecting fundamental signal properties.
    start_idx = 1
    end_idx = len(amplitudes)
    if N % 2 == 0:
      end_idx -= 1  # Exclude Nyquist component

    if start_idx >= end_idx:  # If there are no components to perturb, skip.
      continue

    # Calculate noise standard deviation only for the frequencies we'll perturb.
    amp_for_weighting = amplitudes[start_idx:end_idx]
    phase_noise_std = alpha / (amp_for_weighting + epsilon)
    # Generate two types of perturbations for "Inter-Dimensional Phase-Space Quantum Tunneling Surrogates":
    # 1. Standard "Quantum Fluctuation" (Gaussian noise).
    gaussian_noise = np.random.normal(scale=phase_noise_std)

    # 2. "Phase-Tunneling Event": With a small probability,
    #    a phase 'tunnels' to a discrete, highly perturbed state.
    #    This introduces sparse, large-magnitude phase shifts
    #    to robustly test the Hurst estimation against extreme phase reconfigurations.
    p_tunnel = (
      0.05  # Probability of a phase tunneling event for any given frequency.
    )
    tunneling_mask = np.random.rand(len(amp_for_weighting)) < p_tunnel

    # Possible discrete "tunneling" phase shifts (e.g., pi/2, pi, 3pi/2, or their negatives).
    # Randomly choose one of these "extreme" shifts for tunneled components.
    tunnel_phases = np.random.choice(  # type: ignore
      [np.pi / 2, np.pi, 3 * np.pi / 2, -np.pi / 2, -np.pi, -3 * np.pi / 2],
      size=np.sum(tunneling_mask),
    )

    # Combine the noise. Tunneling events override Gaussian noise where they occur,
    # creating a hybrid perturbation.
    combined_noise = gaussian_noise
    combined_noise[tunneling_mask] = tunnel_phases

    # Apply the combined noise to the phases.
    perturbed_phases = np.copy(phases)
    perturbed_phases[start_idx:end_idx] += combined_noise

    # Reconstruct the surrogate spectrum from original amplitudes and perturbed phases.
    fft_surrogate = amplitudes * np.exp(1j * perturbed_phases)

    # Perform inverse FFT to get the surrogate time series.
    # n=N ensures the output length matches the original input length.
    surrogate_ts = np.fft.irfft(fft_surrogate, n=N)

    # Calculate Hurst for this plausible "alternative history" of the signal.
    h_s = _estimate_hurst_core(surrogate_ts.tolist())
    h_surrogates.append(h_s)

  if not h_surrogates:
    return h_original, None

  # --- Temporal Wormhole Shuffling ---
  # Create a shuffled series based on structural similarity, not time.
  # This tests the robustness of the fractal pattern to temporal reordering.
  wormhole_ts = _create_wormhole_series(time_series)
  h_wormhole = _estimate_hurst_core(wormhole_ts)

  # --- SVD Spectral Scaling ---
  # Analyze the decay of singular values of the trajectory matrix (Tensor Decomposition approach).
  # This provides an estimate based on the covariance structure's complexity ("Entropy of Mixing").
  h_svd, svd_quality = _estimate_hurst_svd(time_series)

  # --- Hyper-Dimensional Fractal Consensus Aggregation ---
  # Combines original, phase-perturbed surrogates, and temporally-shuffled series.

  # Dynamic Signal Characterization: Assess intrinsic properties of the time series
  # to dynamically adjust confidence and weighting.
  ts_np_orig = np.asarray(time_series)

  # 1. Variability Score: High variability suggests richer fractal structure
  #    but also potential for noise, affecting direct R/S stability.
  #    Normalize variance by the mean squared to make it scale-invariant.
  mean_sq = np.mean(ts_np_orig**2)
  var_score = np.var(ts_np_orig) / (mean_sq + 1e-9) if mean_sq > 0 else 0.0
  # Normalize var_score to a [0,1] range, assuming typical biological signals
  # rarely exceed a normalized variance of 0.5 (for a signal oscillating around zero).
  var_score_normalized = np.clip(var_score / 0.5, 0.0, 1.0)

  # 2. Skewness Score: High skewness (non-Gaussianity) implies bursts or
  #    asymmetries, which might make direct R/S less representative of long-term scaling,
  #    and increase the value of phase-shuffled surrogates.
  skew_score = np.abs(skew(ts_np_orig))
  # Normalize skew_score to a [0,1] range. Absolute skewness of 2.0 is considered quite high.
  skew_score_normalized = np.clip(skew_score / 2.0, 0.0, 1.0)

  # Combine these into a "Fractal Information Stability" score.
  # If the signal is very flat (low var_score) or highly skewed (high skew_score),
  # the direct R/S might be less reliable, suggesting more reliance on ensemble.
  stability_factor = (1.0 - var_score_normalized) * (
    1.0 - skew_score_normalized
  )

  h_surrogate_mean = np.mean(h_surrogates)
  h_surrogate_std = np.std(h_surrogates)

  # Calculate confidence/consistency for each estimation method.
  max_plausible_std = 0.25  # For phase surrogates
  max_plausible_divergence = 0.2  # For wormhole/SVD estimate vs original

  # Confidence in phase surrogates: high if their estimates are tightly clustered.
  consistency_surrogate = np.exp(-h_surrogate_std / (max_plausible_std + 1e-9))

  # Confidence in wormhole estimate: high if shuffling doesn't drastically change H.
  wormhole_divergence = np.abs(h_wormhole - h_original)
  consistency_wormhole = np.exp(
    -wormhole_divergence / (max_plausible_divergence + 1e-9)
  )

  # Confidence in SVD estimate: Based on the R^2 of the power law fit.
  # We also penalize if SVD is wildly different from Original, assuming Original is baseline.
  svd_divergence = np.abs(h_svd - h_original)
  consistency_svd = svd_quality * np.exp(
    -svd_divergence / (max_plausible_divergence + 1e-9)
  )

  # --- "Coherence Potential Field Aggregation (CPFA)" ---
  # Treats each estimate as a node in a fully connected potential field.
  # Generalized to N estimates.

  # List of (H_value, Intrinsic_Reliability)
  base_potential_epsilon = 0.01

  estimates = [h_original, h_surrogate_mean, h_wormhole, h_svd]
  potentials = [
    stability_factor + base_potential_epsilon,
    consistency_surrogate + base_potential_epsilon,
    consistency_wormhole + base_potential_epsilon,
    consistency_svd + base_potential_epsilon,
  ]

  estimates_np = np.array(estimates)
  potentials_np = np.array(potentials)

  # Calculate dynamic coherence scale
  overall_hurst_dispersion = np.std(estimates_np)
  sigma_h_coherence = 0.05 + overall_hurst_dispersion * 0.75
  _sigma_sq_safe = (sigma_h_coherence**2) * 2 + 1e-9

  # Calculate Coherence Boost Factors (C_i)
  # Compute pairwise squared distances matrix
  diff_matrix_sq = (estimates_np[:, None] - estimates_np[None, :]) ** 2

  # Sum of squared differences for each node against all others
  # For a product of probabilities (exponential sum in log space): C_i = exp(-sum(d_ij^2))
  sum_sq_diffs = np.sum(diff_matrix_sq, axis=1)

  # Coherence score: High if close to all others on average.
  C_factors = np.exp(-sum_sq_diffs / _sigma_sq_safe)

  # Final Weights
  W_raw = potentials_np * C_factors

  # Normalize
  sum_W = np.sum(W_raw) + 1e-9
  W_norm = W_raw / sum_W

  # Weighted Average
  final_h = np.sum(W_norm * estimates_np)

  # @davyrisso: Adding details for debugging & analysis.

  # The final result is clipped to the valid [0, 1] range.
  return float(np.clip(final_h, 0.0, 1.0)), None


def _create_wormhole_series(
  time_series: List[float], segment_len: int = 20, step: int = 5
) -> List[float]:
  """
  Reorders a time series by finding a path through structurally similar segments,
  creating a "Temporal Wormhole Shuffle." This tests if the fractal properties are
  intrinsic to the signal's patterns, independent of their chronological sequence.
  """
  ts_np = np.asarray(time_series)
  N = len(ts_np)

  # Not enough data to shuffle meaningfully.
  if N < segment_len * 2:
    return time_series

  # 1. Create overlapping segments
  num_segments = (N - segment_len) // step + 1
  segments = np.array(
    [ts_np[i * step : i * step + segment_len] for i in range(num_segments)]
  )

  # 2. Characterize segments with "Dynamical Mode Features" using PCA.
  # This extracts the most dominant structural patterns from the segments,
  # treating them as a matrix and applying a form of matrix decomposition.

  # Handle segments that might be constant (std=0) before scaling for PCA.
  # Replace constant segments with a small random noise or handle them separately.
  segment_stds = np.std(segments, axis=1)
  # Identify constant segments.
  constant_segments_mask = segment_stds == 0

  # Replace constant segments with small noise to prevent issues with StandardScaler.
  # This also helps ensure that all segments contribute meaningfully to PCA,
  # even if their original variance is zero, by giving them a unique, but small, fingerprint.
  if np.any(constant_segments_mask):
    segments[constant_segments_mask] += np.random.normal(
      0, 1e-6, size=segments[constant_segments_mask].shape
    )

  # Standardize each segment (mean 0, variance 1) so PCA is not dominated by amplitude.
  # This ensures that PCA focuses on the *shape* of the segments.
  scaler = StandardScaler()
  scaled_segments = scaler.fit_transform(segments)

  # Apply PCA to reduce dimensionality and extract core "dynamical modes".
  # Choosing `n_components` to capture a significant portion of variance,
  # but not too many to avoid noise, typically 3-5 is reasonable for segment_len=20.
  # Using `min(5, segment_len)` ensures we don't ask for more components than features.
  n_components_pca = min(5, segment_len)

  # If there are fewer segments than components or too few unique features,
  # PCA might fail. Handle this gracefully.
  if num_segments < n_components_pca:
    # Fallback to simple features if PCA is not viable.
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

    # A further normalization step for the PCA features might be beneficial
    # if the variance across components is still very high, to ensure
    # that the NearestNeighbors algorithm gives equal weight to all principal components.
    # This acts as a "second-layer" normalization for enhanced robustness.
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features = (features - features_mean) / (features_std + 1e-9)

  # 3. Find nearest neighbors for pathfinding
  k = min(20, num_segments - 1)
  if k < 1:
    return time_series

  nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(features)
  distances, indices = nn.kneighbors(features)

  # 4. Construct the new path, starting from a stable (low-variance) segment
  path = []
  # 4. Construct the new path, starting from a stable (low-variance) segment.
  # The 'stability' here is defined by segments that are well-represented by PCA.
  # A simple choice is the segment with the smallest L2 norm of its PCA features
  # (i.e., closest to the origin in the PCA space), implying a common/average structure.
  # Alternatively, starting from a segment whose original standard deviation is minimal.
  path = []  # Initialize path for segment sequence.

  if not features.shape[0]:  # If no features were generated, return original.
    return time_series

  # Start from the segment with the smallest sum of absolute PCA feature values,
  # indicating a "most average" or "least extreme" structural pattern.
  start_idx = np.argmin(np.sum(np.abs(features), axis=1))
  current_idx = start_idx
  visited = np.zeros(num_segments, dtype=bool)

  for _ in range(num_segments):
    path.append(current_idx)
    visited[current_idx] = True

    # Find the next unvisited nearest neighbor
    next_idx_found = False
    for neighbor_idx in indices[current_idx]:
      if not visited[neighbor_idx]:
        current_idx = neighbor_idx
        next_idx_found = True
        break

    # If all neighbors are visited, jump to any unvisited node
    if not next_idx_found:
      unvisited_indices = np.where(~visited)[0]
      if len(unvisited_indices) > 0:
        current_idx = unvisited_indices[0]
      else:
        break  # All segments visited

  # 5. Reconstruct the time series by concatenating segments along the new path
  wormhole_ts = segments[path].flatten()
  return wormhole_ts.tolist()


def _estimate_hurst_svd(time_series: List[float]) -> Tuple[float, float]:
  """
  Estimates the Hurst exponent using a simplified Singular Value Decomposition method.
  This version forgoes the complex multi-scale tensor optimization for a more direct
  approach using a single, robustly chosen embedding dimension (L). This improves
  simplicity and computational efficiency.

  The Hurst exponent is estimated from the power-law decay of singular values
  of the trajectory matrix of the time series. H is the negative slope of
  log(singular value) vs log(rank).

  Returns (H_estimate, Fit_Quality).
  """
  ts = np.asarray(time_series)
  N = len(ts)
  if N < 50:
    return 0.5, 0.0

  # Choose a single, robust embedding dimension L.
  # It adapts to the time series length but is capped for stability.
  # A value around N/4 is common, but we cap it to avoid overly large matrices.
  L = int(np.clip(N / 4, 10, 64))

  # The core estimation is now delegated to the single-L helper function.
  h_est, quality, _, _ = _estimate_hurst_svd_single_L(ts, L)

  return h_est, quality


def _estimate_hurst_svd_single_L(
  ts: np.ndarray, L: int
) -> Tuple[float, float, np.ndarray, np.ndarray]:
  """
  Helper function to analyze singular value decay for a single embedding dimension L.
  Returns (H_estimate, Fit_Quality, log_k_inliers, log_s_inliers).
  """
  N = len(ts)
  empty_res = (0.5, 0.0, np.array([]), np.array([]))

  if L < 8 or L >= N:
    return empty_res

  K = N - L + 1
  if K < 2:
    return empty_res

  # 1. Construct Trajectory Matrix (Hankel matrix)
  X = np.array([ts[i : i + L] for i in range(K)])

  # 2. Center the trajectories
  X_centered = X - np.mean(X, axis=1, keepdims=True)

  try:
    # Compute singular values
    s = np.linalg.svd(X_centered, compute_uv=False)
  except np.linalg.LinAlgError:
    return empty_res

  if s[0] == 0:
    return empty_res
  s = s / s[0]

  # 3. Adaptive "Spectral Segment Optimization" to find the scaling region
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
      # Prepare data: log(k) vs log(s)
      k_vals = np.arange(current_start_idx, current_end_idx) + 1
      log_k = np.log(k_vals).reshape(-1, 1)
      log_s = np.log(s[current_start_idx:current_end_idx] + 1e-12)

      try:
        ransac = RANSACRegressor(
          LinearRegression(), random_state=42, min_samples=2
        )
        ransac.fit(log_k, log_s)

        # Check for sufficient inliers
        inlier_mask = ransac.inlier_mask_
        if np.sum(inlier_mask) < min_points_for_fit:
          continue

        # Calculate R^2 on inliers
        r2_current = ransac.estimator_.score(
          log_k[inlier_mask], log_s[inlier_mask]
        )

        if r2_current > best_r2:
          best_r2 = r2_current
          best_h_est = -ransac.estimator_.coef_[0]
          best_log_k = log_k[inlier_mask].flatten()
          best_log_s = log_s[inlier_mask].flatten()

      except (ValueError, IndexError):
        continue

  # If no initial fit was found, return empty results.
  if best_r2 == 0.0:
    return empty_res

  # The result from the initial broad search is returned directly, without iterative refinement.
  # This simplifies the logic and reduces computation time.
  return (
    np.clip(best_h_est, 0.0, 1.0),
    max(0.0, best_r2),
    best_log_k,
    best_log_s,
  )


def _estimate_hurst_core(time_series: List[float]) -> float:
  """
  Estimates the Hurst coefficient (H) for a given time series using Rescaled Range (R/S) analysis.

  The Hurst exponent is a measure of the long-term memory of time series.
  It relates to the autocorrelations of the time series, and the rate at
  which these decrease as the lag between pairs of values increases.

  Returns:
      float: Estimated Hurst coefficient (should be between 0 and 1).
  """

  # --- R/S analysis implementation ---
  # Computes a robust estimate based on the Rescaled Range (R/S) method.
  # The Hurst exponent is derived from the slope of the log-log plot
  # of the rescaled range versus the time lag.

  if len(time_series) < 50:
    return 0.5  # Not enough data for reliable R/S analysis

  # First, detrend the time series to remove any slow-varying linear trend.
  # This is important for biological data which can have instrumental drift,
  # and makes the subsequent analysis on the increments more robust.
  ts_np = np.asarray(time_series)
  x_indices = np.arange(len(ts_np))
  trend_coeffs = np.polyfit(x_indices, ts_np, 1)
  detrended_ts = ts_np - np.polyval(trend_coeffs, x_indices)

  # Compute increments of the detrended time series.
  ts = np.diff(detrended_ts)
  N = len(ts)

  # Create sub-series lengths (scales)
  # We use a logarithmic spacing for scales
  min_scale = 10
  max_scale = N // 2
  if max_scale <= min_scale:
    return 0.5  # Not enough data for a range of scales

  # Generate a set of logarithmically spaced scales (chunk sizes) for the regression.
  # This choice balances computational efficiency and the robustness of the regression fit.
  # Generate a set of logarithmically spaced scales (chunk sizes) for the regression.
  # Increasing the number of scales provides a denser sampling of the R/S curve,
  # leading to a more robust and accurate regression fit.
  # Increased num from 20 to 30 for better resolution in regression.
  scales = np.unique(
    np.logspace(np.log10(min_scale), np.log10(max_scale), num=30).astype(int)
  )

  # Store scale (n) and mean R/S values. Only valid, positive R/S values are included.
  n_values = []
  rs_values = []

  for n in scales:
    # "Fractal Tiling Chunking": Instead of just two sets of chunks (from start and end),
    # we create multiple sets of non-overlapping chunks using different start offsets.
    # This "tiling" of the time series uses the data more thoroughly and provides a more
    # stable R/S estimate for each scale, which is crucial for robust regression later.
    all_chunks_list = []

    # Adaptively determine the number of offsets for tiling.
    # For shorter series, fewer offsets are used to ensure sufficient data per chunk.
    # For longer series, more offsets (up to a prime number, 5) enhance robustness
    # by providing a more diverse sampling of the time series data, reducing risk
    # of harmonic interference and improving stability of R/S estimates.
    N_ts_diff = len(ts)  # Length of the differenced time series
    if N_ts_diff < 200:
      num_offsets = 1
    elif N_ts_diff < 400:
      num_offsets = 3  # Use another prime number for intermediate length
    else:
      num_offsets = 5  # Max prime offsets for sufficiently long series

    for i in range(num_offsets):
      # Calculate a unique offset for each tiling pass.
      offset = (i * n) // num_offsets
      if N - offset < n:  # Ensure there's at least one full chunk possible.
        continue

      # Create a view of the time series starting from the calculated offset.
      series_subset = ts[offset:]
      num_chunks = len(series_subset) // n

      if num_chunks > 0:
        # Extract non-overlapping chunks from this subset and add to our list.
        chunks = series_subset[: num_chunks * n].reshape(num_chunks, n)
        all_chunks_list.append(chunks)

    if not all_chunks_list:
      continue  # No valid chunks could be formed for this scale 'n'.

    # Aggregate all chunks from all tilings into a single array for R/S calculation.
    all_chunks = np.vstack(all_chunks_list)

    # Mean-center each chunk and calculate its cumulative sum (profile)
    mean_centered_chunks = all_chunks - np.mean(
      all_chunks, axis=1, keepdims=True
    )
    z = np.cumsum(mean_centered_chunks, axis=1)

    # Calculate the range (R) and standard deviation (S) for each chunk
    R = np.max(z, axis=1) - np.min(z, axis=1)
    S = np.std(all_chunks, axis=1)

    # Calculate R/S for each chunk. Only consider chunks with non-zero standard deviation.
    # This prevents division by zero and ensures valid R/S values.
    valid_indices = S != 0

    if np.any(valid_indices):
      # Calculate R/S only for valid chunks, then take the mean for this scale.
      # Only use this point for regression if avg_rs is strictly positive.
      avg_rs = np.mean(R[valid_indices] / S[valid_indices])
      if avg_rs > 0:
        n_values.append(n)
        rs_values.append(avg_rs)

  # Perform regression on log(scale) vs log(R/S) to find the Hurst exponent H.

  # Ensure we have enough data points for robust regression.
  # For RANSAC and polynomial fitting, a minimum of 3 points is typically recommended
  # to allow for some outlier detection and to provide robustness beyond a simple 2-point fit.
  if len(n_values) < 3:
    return 0.5  # Not enough valid data points for robust analysis, return default Hurst (random walk).

  # Convert lists to numpy arrays and take logarithms once.
  log_n_arr = np.log(np.array(n_values))
  log_rs_arr = np.log(np.array(rs_values))

  # Create sample weights to emphasize larger scales (long-term memory),
  # as the Hurst exponent is primarily a measure of long-range dependence.
  # A linear ramp gives progressively more weight to points with larger 'n'.
  sample_weights = np.linspace(0.5, 1.5, len(log_n_arr))

  # For 3 or more points, apply the "Adaptive Scale-Degree Fusion" logic.
  # This involves trying multiple polynomial degrees and aggregating their derived Hurst values.

  hurst_candidates = []
  fit_qualities = []  # Store fit quality (e.g., R-squared of inliers) for weighted averaging

  # Pre-calculate the total possible log_n range from all R/S points for spread normalization.
  total_log_n_range_overall = log_n_arr[-1] - log_n_arr[0]

  # Iterate through different polynomial degrees (linear, quadratic, cubic).
  # This allows the estimator to adapt to potentially varying scaling behaviors.
  for degree in [1, 2, 3]:
    # We need at least (degree + 1) points to fit a polynomial of 'degree'.
    if len(n_values) < degree + 1:
      continue

    # Prepare features for polynomial regression: X = [log_n^degree, ..., log_n^1]
    # This creates the design matrix for the polynomial fit.
    X_poly_features = []
    for d in range(degree, 0, -1):
      X_poly_features.append(log_n_arr**d)  # Use log_n_arr
    X_poly = np.column_stack(X_poly_features)

    try:
      # Fit a polynomial to the log-log plot using RANSAC.
      # RANSAC makes the fit robust to outliers in the (log(scale), log(R/S)) data points.
      # We use sample weights to give more importance to the larger scales.
      ransac = RANSACRegressor(LinearRegression(), random_state=42)
      ransac.fit(X_poly, log_rs_arr, sample_weight=sample_weights)

      # --- "Multi-Point Adaptive Chord Ensemble (MACE)" ---
      # Instead of a single chord, we calculate multiple chords across the inlier range
      # and average their slopes, weighted by local segment fit quality.

      inlier_mask = ransac.inlier_mask_
      if not np.any(inlier_mask):
        continue  # No inliers, skip this degree.

      inlier_log_n = log_n_arr[inlier_mask]
      inlier_log_rs = log_rs_arr[inlier_mask]
      inlier_X_poly = X_poly[inlier_mask]

      # Calculate R-squared for the overall inlier set for this specific degree.
      r_squared_overall = ransac.estimator_.score(inlier_X_poly, inlier_log_rs)

      # Ensure there are enough inliers for robust processing.
      if len(inlier_log_n) < 2:
        continue

      # Sort inliers by log_n to ensure proper segment division.
      sort_indices = np.argsort(inlier_log_n)
      inlier_log_n_sorted = inlier_log_n[sort_indices]
      # inlier_log_rs_sorted = inlier_log_rs[sort_indices]

      local_hurst_candidates = []
      local_weights = []

      # Determine number of segments for local chords.
      # Use more segments for more data, up to a reasonable maximum.
      # Each segment needs at least 2 points for a chord.
      num_local_segments = min(
        len(inlier_log_n_sorted) // 2, 4
      )  # Up to 4 segments, ensure at least 2 points per segment.
      if num_local_segments < 1:
        num_local_segments = 1  # Fallback to single segment (full inlier range)

      # Define global min/max log_n from inliers for overall spread calculation.
      current_min_log_n_inliers = inlier_log_n_sorted[0]
      current_max_log_n_inliers = inlier_log_n_sorted[-1]

      # Use `current_min_log_n_inliers` and `current_max_log_n_inliers` for features,
      # even if only one segment, to maintain consistency.
      x_min_features_inliers = np.array(
        [current_min_log_n_inliers**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)
      x_max_features_inliers = np.array(
        [current_max_log_n_inliers**d for d in range(degree, 0, -1)]
      ).reshape(1, -1)

      if num_local_segments == 1:
        # If only one segment, treat it as the full inlier range.
        # Calculate chord from the overall inlier min to max using the RANSAC estimator.
        predicted_rs_min = ransac.estimator_.predict(x_min_features_inliers)
        predicted_rs_max = ransac.estimator_.predict(x_max_features_inliers)

        if (
          current_max_log_n_inliers > current_min_log_n_inliers + 1e-9
        ):  # Avoid tiny range
          local_h = (predicted_rs_max - predicted_rs_min) / (
            current_max_log_n_inliers - current_min_log_n_inliers
          )
          local_hurst_candidates.append(local_h[0])
          local_weights.append(len(inlier_log_n_sorted))
      else:
        for i in range(num_local_segments):
          # Adaptive segmentation. Ensure segments have enough points for a chord.
          # Overlap segments if needed to guarantee minimum length.
          segment_start_idx = len(inlier_log_n_sorted) * i // num_local_segments
          segment_end_idx = (
            len(inlier_log_n_sorted) * (i + 1) // num_local_segments
          )

          # Ensure minimum of 2 points per segment for a valid chord, potentially overlapping.
          if segment_end_idx - segment_start_idx < 2:
            # Try to extend segment to include 2 points.
            # This logic could be more sophisticated for perfect non-overlapping,
            # but simple extension for minimum chord length is pragmatic.
            if segment_start_idx > 0:
              segment_start_idx -= 1
            if segment_end_idx < len(inlier_log_n_sorted):
              segment_end_idx += 1
            if (
              segment_end_idx - segment_start_idx < 2
            ):  # Still not enough, skip.
              continue

          segment_log_n = inlier_log_n_sorted[segment_start_idx:segment_end_idx]
          segment_min_log_n = segment_log_n[0]
          segment_max_log_n = segment_log_n[-1]

          # Check for zero range in segment to avoid division by zero.
          if segment_max_log_n - segment_min_log_n < 1e-9:
            continue

          # Construct feature vectors for segment endpoints using the *overall* RANSAC model.
          x_segment_min_features = np.array(
            [segment_min_log_n**d for d in range(degree, 0, -1)]
          ).reshape(1, -1)
          x_segment_max_features = np.array(
            [segment_max_log_n**d for d in range(degree, 0, -1)]
          ).reshape(1, -1)

          predicted_segment_log_rs_min = ransac.estimator_.predict(
            x_segment_min_features
          )
          predicted_segment_log_rs_max = ransac.estimator_.predict(
            x_segment_max_features
          )

          local_h = (
            predicted_segment_log_rs_max - predicted_segment_log_rs_min
          ) / (segment_max_log_n - segment_min_log_n)

          # Weight by the number of data points in the segment. More points, more reliable local estimate.
          local_hurst_candidates.append(local_h[0])
          local_weights.append(len(segment_log_n))

      if not local_hurst_candidates:
        continue  # No valid local chords, skip this degree.

      # Calculate the weighted average Hurst exponent for this degree from local segments.
      current_hurst = np.average(local_hurst_candidates, weights=local_weights)

      # Calculate quality score for this degree.
      # Combine the overall R-squared with the "inlier spread ratio".
      # A high-quality fit should have good correlation AND cover a significant portion of the scales.
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
      continue  # Skip if regression fails

  # Aggregation of Multi-Degree Estimates using "Dynamical Consensus".
  if not hurst_candidates:
    return 0.5  # Fallback if all degrees fail

  hurst_candidates_np = np.array(hurst_candidates)
  fit_qualities_np = np.array(fit_qualities)

  # Filter out very poor fits (quality ~ 0).
  valid_mask = fit_qualities_np > 1e-9

  if not np.any(valid_mask):
    return float(np.clip(np.median(hurst_candidates_np), 0.0, 1.0))

  valid_hursts = hurst_candidates_np[valid_mask]
  valid_weights = fit_qualities_np[valid_mask]

  # Weighted average of the valid Hurst candidates.
  final_hurst = np.average(valid_hursts, weights=valid_weights)

  return float(np.clip(final_hurst, 0.0, 1.0))
