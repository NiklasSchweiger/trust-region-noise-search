from .t2i import (
    PromptFileBenchmark,
    DrawBenchBenchmark,
    SimpleAnimalsBenchmark,
    COCOCaptionsBenchmark,
    ListPromptsBenchmark,
    CustomPromptBenchmark,
)
from .protein import ProteinLengthBenchmark
# Molecule benchmarks are imported lazily to avoid pulling in QM9 dependencies at startup

__all__ = [
    "PromptFileBenchmark",
    "DrawBenchBenchmark",
    "SimpleAnimalsBenchmark",
    "COCOCaptionsBenchmark",
    "ListPromptsBenchmark",
    "CustomPromptBenchmark",
    "ProteinLengthBenchmark",
    "QM9PropertyTargetsBenchmark",
    "QM9CustomTargetBenchmark",
    "QM9ScalarPropertyBenchmark",
    "PropertyTargetsBenchmark",
]


def __getattr__(name: str):
    """Lazy import for molecule benchmarks to avoid eager loading of QM9 dependencies."""
    if name in ("QM9PropertyTargetsBenchmark", "QM9ScalarPropertyBenchmark",
                "QM9CustomTargetBenchmark", "PropertyTargetsBenchmark"):
        from . import molecule as _mol
        return getattr(_mol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


