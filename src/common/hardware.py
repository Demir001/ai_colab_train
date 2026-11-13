"""Runtime hardware detection for TPU v5e-1 (PyTorch-XLA) and CUDA GPUs (A100/L4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareSelection:
    """Resolved hardware profile and runtime metadata."""

    profile_name: str
    backend: str
    device_name: str
    config: Dict[str, Any]
    base_model: Dict[str, Any]
    experts: Dict[str, Any]

    @property
    def resolved_model_id(self) -> str:
        """Return profile-specific model_id (e.g. 7B on TPU v5e-1) or base_model fallback."""
        return str(self.config.get("model_id") or self.base_model.get("model_id", "Qwen/Qwen2.5-Coder-7B-Instruct"))


def load_hardware_yaml(config_path: str | Path = "configs/hardware_profiles.yaml") -> Dict[str, Any]:
    """Load hardware profiles YAML safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Hardware config not found at: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError(f"Invalid hardware_profiles.yaml structure in {path}")
    return data


def _detect_xla_available() -> bool:
    """Check if PyTorch-XLA TPU runtime is active (supports both 'xla' and PrivateUse1 'tpu' devices)."""
    try:
        import torch_xla  # type: ignore
        import torch_xla.core.xla_model as xm  # type: ignore

        device = xm.xla_device() if hasattr(xm, "xla_device") else torch_xla.device()
        dev_str = str(device).lower()
        return ("xla" in dev_str) or ("tpu" in dev_str)
    except Exception:
        return False


def _detect_cuda_gpu() -> Optional[Dict[str, Any]]:
    """Return CUDA device name and VRAM in GB if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
        return {"name": props.name, "vram_gb": vram_gb, "count": torch.cuda.device_count()}
    except Exception as exc:
        logger.warning("CUDA detection failed: %s", exc)
        return None


def detect_hardware_profile(
    config_path: str | Path = "configs/hardware_profiles.yaml",
    force_profile: Optional[str] = None,
) -> HardwareSelection:
    """Auto-detect TPU v5e-1, CUDA A100, CUDA L4, or CPU fallback."""
    raw = load_hardware_yaml(config_path)
    profiles = raw["profiles"]
    base_model = raw.get("base_model", {})
    experts = raw.get("experts", {})

    if force_profile:
        if force_profile not in profiles:
            raise ValueError(f"Unknown forced profile {force_profile!r}. Available: {list(profiles.keys())}")
        selected = profiles[force_profile]
        return HardwareSelection(
            profile_name=force_profile,
            backend=selected["backend"],
            device_name=selected["device_name"],
            config=selected,
            base_model=base_model,
            experts=experts,
        )

    if _detect_xla_available():
        selected = profiles["tpu_v5e_1"]
        logger.info("Detected PyTorch-XLA TPU environment -> selecting 'tpu_v5e_1'")
        return HardwareSelection(
            profile_name="tpu_v5e_1",
            backend="xla",
            device_name="TPU v5e-1",
            config=selected,
            base_model=base_model,
            experts=experts,
        )

    cuda_info = _detect_cuda_gpu()
    if cuda_info is not None:
        gpu_name = cuda_info["name"]
        vram_gb = cuda_info["vram_gb"]
        if "A100" in gpu_name.upper() or vram_gb >= 38.0:
            profile_name = "cuda_a100"
        else:
            profile_name = "cuda_l4"
        selected = profiles[profile_name]
        logger.info("Detected CUDA GPU (%s, %.2f GB) -> selecting '%s'", gpu_name, vram_gb, profile_name)
        return HardwareSelection(
            profile_name=profile_name,
            backend="cuda",
            device_name=f"{gpu_name} ({vram_gb} GB)",
            config=selected,
            base_model=base_model,
            experts=experts,
        )

    selected = profiles["cpu_fallback"]
    logger.warning("Neither TPU-XLA nor CUDA GPU detected -> selecting 'cpu_fallback'")
    return HardwareSelection(
        profile_name="cpu_fallback",
        backend="cpu",
        device_name="CPU",
        config=selected,
        base_model=base_model,
        experts=experts,
    )
