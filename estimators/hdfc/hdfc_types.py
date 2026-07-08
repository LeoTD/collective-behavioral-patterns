from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json

from estimators import base


@dataclass_json
@dataclass
class SubEstimatorParams(base.EstimatorParams):
  enabled: bool = True


@dataclass_json
@dataclass
class RSParams(SubEstimatorParams):
  rs_min_scale: int = 10
  rs_scale_subsample: int = 30  # Number of log-spaced scales
  rs_min_window: int = 50  # Minimum window size for valid R/S
  rs_diff: bool = (
    True  # If True, difference the series (Standardized for Random Walks)
  )


@dataclass_json
@dataclass
class SurrogatesParams(SubEstimatorParams):
  surrogate_count: int = 5
  surrogate_alpha: float = 0.05  # Noise scaling factor
  surrogate_p_tunnel: float = 0.05  # Phase flipping probability


@dataclass_json
@dataclass
class ReorderedParams(SubEstimatorParams):
  reordered_segment_len: int = 20
  reordered_step: int = 5
  reordered_k_neighbors: int = 20


@dataclass_json
@dataclass
class SVDParams(SubEstimatorParams):
  svd_min_window: int = 50
  svd_fit_fraction: float = (
    0.5  # Fraction of singular values to fit (0.5 = first half)
  )
  svd_early_exit_threshold: float = (
    0.05  # If SVD std < threshold, return immediately
  )


@dataclass_json
@dataclass
class ConsensusParams(base.EstimatorParams):
  consensus_max_plausible_std: float = 0.25
  consensus_max_plausible_div: float = 0.2
  consensus_base_epsilon: float = 0.01
  consensus_sigma_base: float = 0.05
  consensus_sigma_scale: float = 0.75


@dataclass_json
@dataclass
class HDFCParameters(base.EstimatorParams):
  rs: RSParams = field(default_factory=RSParams)
  surrogates: SurrogatesParams = field(default_factory=SurrogatesParams)
  reordered: ReorderedParams = field(default_factory=ReorderedParams)
  svd: SVDParams = field(default_factory=SVDParams)
  consensus: ConsensusParams = field(default_factory=ConsensusParams)


@dataclass_json
@dataclass
class HFDParameters(base.EstimatorParams):
  k_max: int = 10


@dataclass_json
@dataclass
class ConsensusDetails(base.EstimatorResultDetails):
  stability_factor: float
  consistency_surr: float
  consistency_reordered: float
  consistency_svd: float
  potentials: List[float]
  weights: List[float]
  sigma_h: float
  raw_h: float


@dataclass_json
@dataclass
class TimerDetails(base.EstimatorResultDetails):
  timings: Dict[str, float]


@dataclass_json
@dataclass
class HDFCResultDetails(base.EstimatorResultDetails):
  rs_h: float
  svd_h: Optional[float]
  svd_quality: Optional[float]
  surrogates_h_mean: Optional[float]
  surrogates_h_std: Optional[float]
  reordered_h: Optional[float]
  consensus: ConsensusDetails
  stability_factor: float
  timing: TimerDetails
