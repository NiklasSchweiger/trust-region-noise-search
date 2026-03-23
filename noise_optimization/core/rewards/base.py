from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import numpy as np

if TYPE_CHECKING:  # For type hints only
    from PIL.Image import Image as PILImageType
else:
    PILImageType = Any

try:  # Optional dependency for image-based rewards
    from PIL import Image as _PILImageModule
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully
    _PILImageModule = None
    PIL_AVAILABLE = False


class RewardFunction(ABC):
    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_input_format(self) -> str:
        raise NotImplementedError

    def get_output_range(self) -> tuple[float, float]:
        return (-float("inf"), float("inf"))

    def __call__(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return self.evaluate(candidates, context)


class ImageRewardFunction(RewardFunction):
    def __init__(self, name: str, device: Optional[str] = None, expects_pil: bool = False, differentiable: bool = False):
        super().__init__(name, device)
        self.expects_pil = expects_pil
        self.differentiable = differentiable

    def get_input_format(self) -> str:
        return "pil" if self.expects_pil else "tensor"

    def _convert_to_expected_format(self, images: Any) -> Any:
        if self.expects_pil:
            if not PIL_AVAILABLE:
                raise ImportError(
                    "Pillow (PIL) is required for rewards that expect PIL images. "
                    "Install Pillow or select a tensor-based reward."
                )
            # If we already have PIL images, keep them as-is
            if PIL_AVAILABLE and isinstance(images, _PILImageModule.Image):
                return images
            if (
                PIL_AVAILABLE
                and isinstance(images, (list, tuple))
                and len(images) > 0
                and all(isinstance(img, _PILImageModule.Image) for img in images)
            ):
                return list(images)

            if isinstance(images, torch.Tensor):
                # If differentiable mode is enabled, we cannot convert to PIL (it breaks gradients)
                if self.differentiable:
                    raise ValueError(
                        f"Cannot convert tensors to PIL images when differentiable=True. "
                        f"Reward {self.name} requires PIL but cannot be used in differentiable mode. "
                        f"Consider using a tensor-based reward (e.g., CLIPRewardHF) instead."
                    )
                try:
                    from torchvision.transforms.functional import to_pil_image
                    t = images.detach().cpu()
                    # Handle channel-last tensors from diffusers (B,H,W,C) or (H,W,C)
                    if t.dim() == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
                        t = t.permute(0, 3, 1, 2)
                    elif t.dim() == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                        t = t.permute(2, 0, 1)
                    if images.dim() == 4:
                        return [to_pil_image(t[i].clamp(0, 1)) for i in range(t.shape[0])]
                    else:
                        return to_pil_image(t.clamp(0, 1))
                except Exception:
                    return images
            elif isinstance(images, np.ndarray):
                try:
                    from diffusers.utils import numpy_to_pil
                    return numpy_to_pil(images)
                except Exception:
                    return images
            elif isinstance(images, (list, tuple)):
                # Common case: model returns list of tensors / arrays
                out: list[Any] = []
                for img in images:
                    if PIL_AVAILABLE and isinstance(img, _PILImageModule.Image):
                        out.append(img)
                        continue
                    if isinstance(img, torch.Tensor):
                        # If differentiable mode is enabled, we cannot convert to PIL
                        if self.differentiable:
                            raise ValueError(
                                f"Cannot convert tensors to PIL images when differentiable=True. "
                                f"Reward {self.name} requires PIL but cannot be used in differentiable mode."
                            )
                        try:
                            from torchvision.transforms.functional import to_pil_image
                            t = img.detach().cpu()
                            # Handle channel-last tensors from diffusers (B,H,W,C) or (H,W,C)
                            if t.dim() == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
                                t = t.permute(0, 3, 1, 2)
                            elif t.dim() == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                                t = t.permute(2, 0, 1)
                            # Handle possible batch tensors inside list
                            if t.dim() == 4:
                                out.extend([to_pil_image(t[i].clamp(0, 1)) for i in range(t.shape[0])])
                            else:
                                out.append(to_pil_image(t.clamp(0, 1)))
                        except Exception:
                            out.append(img)
                        continue
                    if isinstance(img, np.ndarray):
                        try:
                            from diffusers.utils import numpy_to_pil
                            pil_list = numpy_to_pil(img)
                            if isinstance(pil_list, list):
                                out.extend(pil_list)
                            else:
                                out.append(pil_list)
                        except Exception:
                            out.append(img)
                        continue
                    out.append(img)
                return out
        else:
            if PIL_AVAILABLE and isinstance(images, _PILImageModule.Image):
                from torchvision.transforms import ToTensor
                return ToTensor()(images).unsqueeze(0)
            elif (
                PIL_AVAILABLE
                and isinstance(images, list)
                and all(isinstance(img, _PILImageModule.Image) for img in images)
            ):
                from torchvision.transforms import ToTensor
                return torch.stack([ToTensor()(img) for img in images])
            elif isinstance(images, (list, tuple)) and len(images) > 0 and all(isinstance(x, torch.Tensor) for x in images):
                # List of tensors -> stack into a batch
                try:
                    return torch.stack([x.detach() for x in images], dim=0)
                except Exception:
                    return images
        return images


class TextPromptRewardFunction(ImageRewardFunction):
    def __init__(self, name: str, device: Optional[str] = None, expects_pil: bool = False, differentiable: bool = False):
        super().__init__(name, device, expects_pil, differentiable)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if context is None or "prompt" not in context:
            raise ValueError(f"{self.name} requires a 'prompt' in context")
        prompt = context["prompt"]
        images = self._convert_to_expected_format(candidates)
        return self._evaluate_with_prompt(images, prompt)

    @abstractmethod
    def _evaluate_with_prompt(self, images: Any, prompt: str) -> torch.Tensor:
        raise NotImplementedError


class CompositeReward(RewardFunction):
    def __init__(self, rewards: list[RewardFunction], weights: Optional[list[float]] = None, device: Optional[str] = None):
        super().__init__("composite", device)
        if not rewards:
            raise ValueError("CompositeReward requires at least one reward function")
        self.rewards = rewards
        if weights is None:
            weights = [1.0] * len(rewards)
        if len(weights) != len(rewards):
            raise ValueError("weights must match rewards length")
        self.weights = [float(w) for w in weights]

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        total = None
        for w, rf in zip(self.weights, self.rewards):
            scores = rf.evaluate(candidates, context)
            scores = scores.to(self.device).to(torch.float32)
            total = scores * w if total is None else total + scores * w
        return total if total is not None else torch.tensor([0.0], device=self.device)

    def get_input_format(self) -> str:
        return "auto"

    def get_output_range(self) -> tuple[float, float]:
        return (-1e9, 1e9)


class MoleculeRewardFunction(RewardFunction):
    def __init__(self, name: str, device: Optional[str] = None):
        super().__init__(name, device)

    def get_input_format(self) -> str:
        return "smiles"

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(candidates, torch.Tensor):
            batch_size = candidates.shape[0]
            return torch.zeros(batch_size, device=self.device)
        elif isinstance(candidates, str):
            return torch.tensor([self._evaluate_molecule(candidates)], device=self.device)
        elif isinstance(candidates, list):
            rewards = [self._evaluate_molecule(mol) for mol in candidates]
            return torch.tensor(rewards, device=self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(candidates)}")

    @abstractmethod
    def _evaluate_molecule(self, molecule: Any) -> float:
        raise NotImplementedError


@dataclass
class RewardConfig:
    model_name: Optional[str] = None
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    image_size: int = 224
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None
    temperature: float = 1.0
    scale_factor: float = 1.0
    differentiable: bool = False  # If True, preserve gradients through reward computation
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For combined rewards (e.g. clip + aesthetics): weights per component, e.g. [0.7, 0.3]
    weights: Optional[List[float]] = None


class RewardFunctionRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, reward_class: type) -> None:
        cls._registry[name] = reward_class

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise ValueError(f"Unknown reward function: {name}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> RewardFunction:
        reward_class = cls.get(name)
        return reward_class(**kwargs)



def get_reward_function(name: str, **kwargs) -> RewardFunction:
    from dataclasses import fields as _dc_fields
    allowed = {f.name for f in _dc_fields(RewardConfig)}
    cfg_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k in allowed}
    if "dtype" in cfg_kwargs and isinstance(cfg_kwargs["dtype"], str):
        dt_str = str(cfg_kwargs["dtype"]).lower()
        if dt_str in ("float16", "fp16", "half"):
            cfg_kwargs["dtype"] = torch.float16
        elif dt_str in ("bfloat16", "bf16"):
            cfg_kwargs["dtype"] = torch.bfloat16
        elif dt_str in ("float32", "fp32", "float"):
            cfg_kwargs["dtype"] = torch.float32
    # Ensure device is valid - fall back to CPU if CUDA is requested but not available
    if "device" in cfg_kwargs:
        requested_device = cfg_kwargs["device"]
        if requested_device == "cuda" and not torch.cuda.is_available():
            cfg_kwargs["device"] = "cpu"
    # Convert weights to list (e.g. from OmegaConf) so RewardConfig accepts it
    if "weights" in cfg_kwargs and cfg_kwargs["weights"] is not None:
        w = cfg_kwargs["weights"]
        cfg_kwargs["weights"] = list(w) if hasattr(w, "__iter__") and not isinstance(w, str) else w
    config = RewardConfig(**cfg_kwargs)
    return RewardFunctionRegistry.create(name, config=config)


def list_reward_functions() -> List[str]:
    return RewardFunctionRegistry.list_available()


