# Lazy imports to avoid pulling in application-specific dependencies
# Import only when actually needed

__all__ = [
    "MoleculeOptimizationProblem",  # alias (QM9)
    "QM9OptimizationProblem",
    "ProteinOptimizationProblem",
]


def __getattr__(name: str):
    """Lazy import for problem classes to avoid eager loading of dependencies."""
    if name == "MoleculeOptimizationProblem":
        from .molecule import MoleculeOptimizationProblem
        return MoleculeOptimizationProblem
    if name == "QM9OptimizationProblem":
        from .molecule import QM9OptimizationProblem
        return QM9OptimizationProblem
    if name == "ProteinOptimizationProblem":
        from .protein import ProteinOptimizationProblem
        return ProteinOptimizationProblem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
