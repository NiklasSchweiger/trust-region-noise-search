from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from abc import ABC, abstractmethod

from ..loggers.base import ExperimentLogger


class Benchmark(ABC):
    """Base class for defining a benchmark suite (e.g., prompts or targets)."""

    def __init__(self, num_runs: int = 1, randomize: bool = False):
        self.num_runs = num_runs
        self.randomize = randomize

    @abstractmethod
    def prompts(self) -> List[str]:
        """Return a list of prompts/targets to optimize.

        This legacy method is used to build default tasks. Override tasks()
        directly for non-text modalities or richer task definitions.
        """
        raise NotImplementedError

    # New, more general task interface
    def tasks(self) -> List[Dict[str, Any]]:
        """Return a list of task dicts.

        Each task should minimally include a 'context' dict. Optional fields:
        - 'modality': e.g., 'image', 'image_noise', 'molecule', ...
        - any additional fields used by the ProblemBuilder
        """
        tasks: List[Dict[str, Any]] = []
        try:
            prompts = self.prompts()
            for p in prompts:
                tasks.append({
                    "modality": "image",
                    "context": {"prompt": p},
                })
        except NotImplementedError:
            pass
        return tasks

    def iter_tasks(self) -> Iterable[Dict[str, Any]]:
        for i, task in enumerate(self.tasks()[: self.num_runs]):
            # Attach index for convenience
            task_with_index = dict(task)
            task_with_index.setdefault("context", {})
            task_with_index["context"]["index"] = i
            yield task_with_index

    # Minimal base: require concrete benchmarks to implement run
    @abstractmethod
    def run(self, algorithm: Any, problem_builder: Any, *, algorithm_kwargs: Optional[Dict[str, Any]] = None, logger: Optional[ExperimentLogger] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


