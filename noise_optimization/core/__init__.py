"""Core abstractions for search, problems, benchmarks, tracking, noise injection, and algorithms."""

from .problems.base import Problem
from .benchmarks.base import Benchmark
from .loggers.base import ExperimentLogger, WandbLogger
from .rewards.base import (
    RewardFunction,
    ImageRewardFunction,
    TextPromptRewardFunction,
    CompositeReward,
    RewardConfig,
    MoleculeRewardFunction,
)
from .rewards import (
    RewardFunctionRegistry,
    get_reward_function,
    list_reward_functions,
)

from .solvers.base import Solver, SolveResult
from .solvers.random_search import RandomSearchSolver
from .solvers.zero_order import ZeroOrderSolver
from .solvers.trs import TrustRegionSolver

__all__ = [
    # Core abstractions
    "Problem",
    "Benchmark",
    "ExperimentLogger",
    "WandbLogger",

    # Reward functions
    "RewardFunction",
    "ImageRewardFunction",
    "TextPromptRewardFunction",
    "MoleculeRewardFunction",
    "CompositeReward",
    "RewardConfig",
    "RewardFunctionRegistry",
    "get_reward_function",
    "list_reward_functions",

    # Solvers
    "Solver",
    "SolveResult",
    "RandomSearchSolver",
    "ZeroOrderSolver",
    "TrustRegionSolver",
]


# Lazy loading functions for application-specific modules
def _lazy_import_image_problem():
    from .problems.image import ImageGenerationProblem
    return ImageGenerationProblem


def _lazy_import_molecule_problem():
    from .problems.molecule import QM9OptimizationProblem
    return QM9OptimizationProblem


def _lazy_import_t2i_logger():
    from .loggers.t2i import T2IWandbLogger
    return T2IWandbLogger


def _lazy_import_molecule_logger():
    from .loggers.molecule_logger import MoleculeLogger
    return MoleculeLogger


def _lazy_import_molecule_visualizer():
    from .loggers.molecule_visualizer import MoleculeVisualizer, visualize_best_molecules
    return MoleculeVisualizer, visualize_best_molecules


def _lazy_import_t2i_models():
    from .models.t2i import (
        SamplingPipeline,
        SDSamplingPipeline,
        SDXLSamplingPipeline,
        SD3SamplingPipeline,
        PixArtAlphaSamplingPipeline,
        PixArtSigmaSamplingPipeline,
        LCMSamplingPipeline,
        FluxSamplingPipeline,
    )
    return {
        "SamplingPipeline": SamplingPipeline,
        "SDSamplingPipeline": SDSamplingPipeline,
        "SDXLSamplingPipeline": SDXLSamplingPipeline,
        "SD3SamplingPipeline": SD3SamplingPipeline,
        "PixArtAlphaSamplingPipeline": PixArtAlphaSamplingPipeline,
        "PixArtSigmaSamplingPipeline": PixArtSigmaSamplingPipeline,
        "LCMSamplingPipeline": LCMSamplingPipeline,
        "FluxSamplingPipeline": FluxSamplingPipeline,
    }


def _lazy_import_t2i_benchmarks():
    from .benchmarks.t2i import (
        PromptFileBenchmark,
        DrawBenchBenchmark,
        SimpleAnimalsBenchmark,
        COCOCaptionsBenchmark,
        ListPromptsBenchmark,
    )
    return {
        "PromptFileBenchmark": PromptFileBenchmark,
        "DrawBenchBenchmark": DrawBenchBenchmark,
        "SimpleAnimalsBenchmark": SimpleAnimalsBenchmark,
        "COCOCaptionsBenchmark": COCOCaptionsBenchmark,
        "ListPromptsBenchmark": ListPromptsBenchmark,
    }


def _lazy_import_molecule_benchmarks():
    from .benchmarks.protein import ProteinLengthBenchmark
    return {
        "ProteinLengthBenchmark": ProteinLengthBenchmark,
    }


def _lazy_import_qm9_benchmark():
    from .benchmarks.molecule import QM9PropertyTargetsBenchmark
    return QM9PropertyTargetsBenchmark


