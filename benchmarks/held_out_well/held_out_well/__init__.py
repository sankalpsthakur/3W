"""Held-out-well benchmark for the 3W dataset (protocol 3W-HW v1).

Scenario: models are developed on some wells and deployed on wells they
have never seen. The split is by well; see ``README.md``.
"""

from .benchmark import Benchmark
from .constants import PROTOCOL_ID
from .scoring import report, select_threshold

__all__ = ["Benchmark", "PROTOCOL_ID", "report", "select_threshold"]
