"""Configuration objects -- every knob in the lab, in one readable place.

THE IDEA
    Anomaly detection has a lot of small choices (image size, which backbone
    layers to tap, how big the memory bank is) and *each one is a lesson*. If
    they hide as magic numbers scattered across the code, you can't experiment.
    Gathering them into frozen dataclasses does three things:

      1. Documents the design -- read this file and you know the whole pipeline.
      2. Makes the STUDY_GUIDE experiments trivial: a script just overrides one
         field (e.g. ``BackboneConfig(layers=("layer3",))``) and reruns.
      3. Gets stamped into results.py so every logged number is reproducible.

    Defaults are the values that give a strong result on MVTec metal_nut. Change
    them on purpose, from an experiment, not by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and how each image is normalised before the backbone.

    ``image_size`` is a square resize. ``imagenet_mean/std`` matter more than they
    look: the backbone was pretrained on ImageNet, so its features only mean
    something if we feed it images normalised **the same way** ImageNet was. Skip
    this and the features -- and every distance built on them -- are garbage.
    """

    root: str = "data/mvtec"
    category: str = "metal_nut"
    image_size: int = 256
    # ImageNet statistics -- the normalisation the pretrained backbone expects.
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneConfig:
    """The frozen pretrained CNN we borrow features from -- no training here.

    ``layers`` picks which intermediate feature maps to tap. The choice is the
    single most instructive knob in the whole project:
      - early layers (``layer1``) = fine texture, high resolution, little meaning;
      - deep layers (``layer4``) = semantic, low resolution, coarse localisation.
    ``("layer2", "layer3")`` is the PatchCore sweet spot: mid-level features that
    still carry enough spatial detail to localise a scratch. Try ``("layer3",)``
    alone and watch the heatmap get blurrier (that is a STUDY_GUIDE experiment).
    """

    name: str = "wide_resnet50_2"  # torchvision pretrained; "resnet18" is lighter
    layers: tuple[str, ...] = ("layer2", "layer3")


@dataclass(frozen=True)
class PatchCoreConfig:
    """The embedding memory-bank detector (PatchCore-lite).

    ``coreset_ratio`` is how much of the nominal patch embeddings we keep in the
    bank. Keeping 100% is accurate but slow and memory-hungry; PatchCore's insight
    is that a well-spread subsample loses almost no accuracy. We start with a
    simple random subsample -- ``coreset_ratio=0.1`` keeps a tenth. Pushing it
    down until AUROC drops is a STUDY_GUIDE experiment.

    ``n_neighbors`` is the k in the kNN distance that scores each patch. ``blur_sigma``
    smooths the upsampled anomaly map so a single noisy patch doesn't scream.
    """

    coreset_ratio: float = 0.1
    n_neighbors: int = 1
    blur_sigma: float = 4.0
    seed: int = 42


@dataclass(frozen=True)
class AEConfig:
    """The reconstruction autoencoder baseline (the *weak* method, on purpose).

    Trained only on good images to minimise reconstruction error; at test time,
    high per-pixel error is supposed to mean "anomaly". The lesson is watching it
    underperform -- ``latent_dim`` too large and it reconstructs defects too, which
    is exactly why the memory-bank method exists.
    """

    latent_dim: int = 128
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Bundles the four configs so a script passes one object around."""

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    autoencoder: AEConfig = field(default_factory=AEConfig)
