import time
from contextlib import contextmanager
from typing import Dict, Generator


class ExecutionTimer:
  def __init__(self) -> None:
    self.timings: Dict[str, float] = {}

  @contextmanager
  def measure(self, name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
      yield
    finally:
      end = time.perf_counter()
      self.timings[name] = end - start
