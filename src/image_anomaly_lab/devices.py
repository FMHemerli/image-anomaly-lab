"""Accelerator discovery -- *what is torch actually running on?*

THE IDEA
    Every heavy step in this lab (running images through the backbone, training
    the autoencoder, computing nearest-neighbour distances) wants the GPU. But
    on an AMD RDNA4 card the PyTorch **ROCm** build is sneaky: it reports itself
    through the very same CUDA API surface (``torch.cuda``) that an NVIDIA build
    would use. So ``torch.cuda.is_available()`` returning True does *not* mean
    you are on CUDA -- it may be ROCm pretending. The tell-tale is
    ``torch.version.hip``: it is set on a ROCm build and ``None`` on real CUDA.

WHY A WHOLE MODULE FOR THIS
    Reproducibility. Every result we log (AUROC, timings) is meaningless without
    knowing the hardware it came from, so ``results.py`` stamps ``describe_torch``
    into the run metadata. Keeping the detection here, in one place, means the
    scripts, the tests and the Streamlit app all agree on which device to use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """A human-readable snapshot of the torch runtime, for logging in results."""

    framework: str
    framework_version: str
    accelerator: str  # e.g. "ROCm (HIP 6.4.x)", "CUDA", "CPU"
    device_name: str  # e.g. "AMD Radeon RX 9060 XT", "cpu"

    def as_dict(self) -> dict[str, str]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "accelerator": self.accelerator,
            "device_name": self.device_name,
        }


def describe_torch() -> DeviceInfo:
    """Report what torch will run on, distinguishing ROCm from real CUDA.

    The ``hip`` attribute is the discriminator: present on a ROCm wheel, ``None``
    on a CUDA wheel. We surface it so a logged run is never ambiguous about
    whether that "cuda" device was actually an AMD card.
    """
    import torch

    version = torch.__version__
    hip = getattr(torch.version, "hip", None)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        accelerator = f"ROCm (HIP {hip})" if hip else "CUDA"
        return DeviceInfo("torch", version, accelerator, name)

    # No accelerator visible. On RDNA4 this often means ROCm did not enumerate
    # the card yet -- try `export HSA_OVERRIDE_GFX_VERSION=12.0.0` before rerunning
    # (see requirements-torch-rocm.txt). We fall back to CPU rather than crash so
    # tests and small experiments still work.
    return DeviceInfo("torch", version, "CPU", "cpu")


def resolve_torch_device():
    """Return the best available ``torch.device`` -- the GPU if present, else CPU.

    Use this everywhere instead of hardcoding ``"cuda"``: it keeps the code
    runnable on a CPU-only machine (CI, a laptop without the card) while
    transparently using the AMD GPU when it is there.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
