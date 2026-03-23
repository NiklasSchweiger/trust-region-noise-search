from __future__ import annotations

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None
    OmegaConf = None


class ExperimentLogger(ABC):
    """Abstract experiment logger interface."""

    @abstractmethod
    def log(self, metrics: Dict[str, Any]) -> None:
        raise NotImplementedError

    def format_context(self, context: Optional[Dict[str, Any]]) -> str:
        return ""

    def log_image(self, key: str, image: Any, caption: Optional[str] = None, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        return

    def log_images(self, key: str, images: List[Any], captions: Optional[List[str]] = None, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        return

    def close(self) -> None:
        return


def _to_plain_dict(obj: Any) -> Any:
    """Convert nested DictConfig/omegaconf structures into plain Python containers.
    
    Filters out non-serializable objects (functions, callables, etc.) to prevent
    wandb serialization errors.
    """
    if DictConfig is not None and isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    
    # Handle callables (functions, methods, etc.) - replace with string representation
    if callable(obj) and not isinstance(obj, type):
        # Skip built-in types (int, str, etc.) which are callable but serializable
        if type(obj).__module__ not in ('builtins', '__builtin__'):
            return f"<callable: {type(obj).__name__}>"
    
    # Handle dataclasses - convert to dict but filter non-serializable values
    try:
        from dataclasses import is_dataclass, fields
        if is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for field in fields(obj):
                try:
                    value = getattr(obj, field.name)
                    result[field.name] = _to_plain_dict(value)
                except Exception:
                    result[field.name] = f"<non-serializable: {type(value).__name__}>"
            return result
    except (ImportError, AttributeError):
        pass
    
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(v) for v in obj)
    
    # Basic serializable types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # For other types, try to return as-is (wandb will handle serialization)
    # but catch common non-serializable types
    if hasattr(obj, '__module__') and obj.__module__ not in ('builtins', '__builtin__', 'typing'):
        # Check if it's a type that wandb might struggle with
        if hasattr(obj, '__dict__'):
            try:
                # Try to convert object with __dict__ to dict
                return {k: _to_plain_dict(v) for k, v in obj.__dict__.items()}
            except Exception:
                return f"<non-serializable: {type(obj).__name__}>"
    
    return obj


def _as_plain_dict(cfg: Any) -> Any:
    """Convert nested DictConfig/omegaconf structures into plain Python containers.

    Lighter-weight variant of _to_plain_dict, used by loggers to normalise
    logging-config dicts before processing.
    """
    if DictConfig is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return {k: _as_plain_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_as_plain_dict(v) for v in cfg]
    return cfg


class WandbLogger(ExperimentLogger):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable: bool = True,
        wandb_dir: Optional[str] = None,
    ):
        self.enable = enable
        self.config = config
        self.project = project
        self.name = name
        self.wandb = None
        self.run = None
        if self.enable:
            import wandb
            self.wandb = wandb
            # Convert DictConfig to plain dict for wandb serialization
            config_plain = _to_plain_dict(config) if config is not None else None
            # Additional sanitization: filter out any remaining non-serializable values
            config_plain = self._sanitize_config(config_plain) if config_plain is not None else None
            init_kwargs = dict(project=project, name=name, config=config_plain or {}, mode="online")
            if wandb_dir:
                init_kwargs["dir"] = wandb_dir
            try:
                self.run = wandb.init(**init_kwargs)
            except Exception as e:
                # If wandb init fails due to serialization, try with an empty config
                if "serialize" in str(e).lower() or "asdict" in str(e).lower() or "json" in str(e).lower():
                    print(f"[WARNING] WandB config serialization failed, using empty config. Error: {e}")
                    init_kwargs["config"] = {}
                    self.run = wandb.init(**init_kwargs)
                else:
                    raise
            self._define_base_metrics()
    
    def _sanitize_config(self, config: Any) -> Any:
        """Recursively sanitize config to remove non-serializable objects."""
        if config is None:
            return None
        # Known serializable types - return as-is
        if isinstance(config, (str, int, float, bool)):
            return config
        if isinstance(config, dict):
            sanitized = {}
            for k, v in config.items():
                try:
                    sanitized[k] = self._sanitize_config(v)
                except Exception:
                    sanitized[k] = f"<non-serializable: {type(v).__name__}>"
            return sanitized
        if isinstance(config, (list, tuple)):
            sanitized = []
            for item in config:
                try:
                    sanitized.append(self._sanitize_config(item))
                except Exception:
                    sanitized.append(f"<non-serializable: {type(item).__name__}>")
            return sanitized if isinstance(config, list) else tuple(sanitized)
        # Check if it's a callable (function, method, etc.)
        if callable(config) and not isinstance(config, type):
            if type(config).__module__ not in ('builtins', '__builtin__'):
                return f"<callable: {type(config).__name__}>"
        # For other types, try lightweight serialization check
        # Only check if it's not a basic type
        if not isinstance(config, (str, int, float, bool, type(None))):
            try:
                import json
                json.dumps(config)
                return config
            except (TypeError, ValueError):
                # Replace with string representation
                return f"<non-serializable: {type(config).__name__}>"
        return config

    def _define_base_metrics(self) -> None:
        if not self.enable:
            return
        self.wandb.define_metric("iteration_step")
        self.wandb.define_metric("task_step")

    def define_metrics(self, metrics: Dict[str, Optional[str]]) -> None:
        if not self.enable:
            return
        for metric, step_metric in metrics.items():
            if step_metric:
                self.wandb.define_metric(metric, step_metric=step_metric)
            else:
                self.wandb.define_metric(metric)

    def log(self, metrics: Dict[str, Any]) -> None:
        if not self.enable:
            return
        self.wandb.log(metrics)

    def log_image(self, key: str, image: Any, caption: Optional[str] = None, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        if not self.enable:
            return
        img = self.wandb.Image(image, caption=caption) if caption is not None else self.wandb.Image(image)
        payload: Dict[str, Any] = {key: img}
        if step_key is not None and step is not None:
            payload[step_key] = step
        self.wandb.log(payload)

    def log_images(self, key: str, images: List[Any], captions: Optional[List[str]] = None, step_key: Optional[str] = None, step: Optional[int] = None) -> None:
        if not self.enable:
            return
        if captions is not None:
            imgs = [self.wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
        else:
            imgs = [self.wandb.Image(img) for img in images]
        payload: Dict[str, Any] = {key: imgs}
        if step_key is not None and step is not None:
            payload[step_key] = step
        self.wandb.log(payload)

    def close(self) -> None:
        if self.enable and self.run is not None:
            self.run.finish()


