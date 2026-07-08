from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Protocol, Tuple, runtime_checkable

from dataclasses_json import dataclass_json


class InputType(Enum):
  """Specifies the type of input data an estimator expects."""
  INCREMENTS = auto()   # Stationary noise (e.g., fGn)
  CUMULATIVE = auto()   # Random walk / Trajectory (e.g., fBm)
  ANY = auto()          # Can handle either (implementation dependent)


@dataclass_json
@dataclass
class EstimatorParams:
  pass


@dataclass_json
@dataclass
class EstimatorResultDetails:
  pass


@runtime_checkable
class Estimator(Protocol):
  def __call__(
    self, time_series: List[float], params: Optional[EstimatorParams] = None
  ) -> Tuple[float, Optional[EstimatorResultDetails]]: ...
