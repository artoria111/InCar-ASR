"""InCar-ASR reproducible evaluation and deployment toolkit."""

from .backends import BackendResult, MockBackend, OnnxBackend, SubprocessBackend
from .commands import CommandMatch, CommandMatcher, build_default_catalog
from .contracts import ModelContract
from .evaluation import run_evaluation

__all__ = [
    "BackendResult",
    "CommandMatch",
    "CommandMatcher",
    "MockBackend",
    "ModelContract",
    "OnnxBackend",
    "SubprocessBackend",
    "build_default_catalog",
    "run_evaluation",
]

__version__ = "0.2.0"
