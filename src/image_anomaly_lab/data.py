"""Loading MVTec AD -- turning a folder tree into tensors, without leaking labels.

THE IDEA
    The MVTec layout encodes the entire experimental protocol (see
    ``scripts/download_data.py``). This module reads it faithfully:

      - TRAINING uses ``train/good`` only. No labels, no defects -- the model must
        learn "normal" from good parts alone. Feeding it anything else would be
        cheating and defeats the point of the whole approach.
      - TESTING uses ``test/*``: ``test/good`` (label 0) and ``test/<defect>``
        (label 1), each defective image paired with its ``ground_truth`` mask so
        we can score *localisation*, not just detection.

DESIGN -- ONE TESTABLE SEAM
    ``list_samples`` is pure ``pathlib``: it walks the tree and returns
    (image_path, label, mask_path) triples with no torch involved, so its label
    logic is unit-tested against a tiny synthetic folder. ``MVTecDataset`` wraps
    that list with the image transform (which needs torch/torchvision). Splitting
    them keeps the label bookkeeping -- the part that is easy to get subtly wrong
    -- honest and cheap to verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DataConfig

GOOD_LABEL = 0
DEFECT_LABEL = 1


@dataclass(frozen=True)
class Sample:
    """One image and everything we know about it *before* loading pixels.

    ``mask_path`` is None for good images (nothing to segment) and for the train
    split; it points to the per-pixel ground-truth PNG for a defective test image.
    """

    image_path: Path
    label: int  # GOOD_LABEL or DEFECT_LABEL
    defect_type: str  # "good" or the defect subfolder name, e.g. "scratch"
    mask_path: Path | None


def list_samples(root: str | Path, category: str, split: str) -> list[Sample]:
    """Enumerate a split of one category. Pure filesystem logic -- no torch.

    ``split`` is "train" or "test". For "train" we *only* look inside ``good``:
    this is where the good-images-only discipline is enforced, in one place. For
    "test" we read every subfolder, label ``good`` as 0 and the rest as 1, and
    match each defective image to its mask in ``ground_truth/<defect>/<stem>_mask.png``
    (the MVTec naming convention).
    """
    root = Path(root)
    split_dir = root / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"missing split '{split}' for category '{category}' under {root}. "
            f"Run: python scripts/download_data.py --category {category}"
        )

    samples: list[Sample] = []
    # Sorting makes the memory bank and every evaluation deterministic across runs.
    for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect_type = sub.name
        is_good = defect_type == "good"
        # Enforce the protocol: the training split contributes good images only.
        if split == "train" and not is_good:
            continue
        label = GOOD_LABEL if is_good else DEFECT_LABEL
        for img in sorted(sub.glob("*.png")):
            mask = None
            if not is_good:
                candidate = root / category / "ground_truth" / defect_type / f"{img.stem}_mask.png"
                mask = candidate if candidate.exists() else None
            samples.append(Sample(img, label, defect_type, mask))
    return samples


def build_transform(cfg: DataConfig):
    """Resize -> tensor -> ImageNet-normalise. The normalisation is not optional.

    The backbone was pretrained on ImageNet; its features are only meaningful on
    inputs standardised with ImageNet's channel statistics. This is the same
    transform for train and test -- *train/serve parity*: any mismatch here would
    move the test embeddings relative to the memory bank and wreck the distances.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),  # -> float CHW in [0, 1]
            transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
        ]
    )


def build_mask_transform(cfg: DataConfig):
    """Masks get resized to match the image but kept as raw 0/1 -- no normalising.

    A ground-truth mask is a label map, not a photo, so it must NOT be run through
    ImageNet normalisation, and it uses nearest-neighbour resizing so no in-between
    grey values are invented at defect boundaries.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),  # -> {0,1} float, shape 1xHxW
        ]
    )


class MVTecDataset:
    """A torch ``Dataset`` over one split, yielding (image, label, mask, path).

    Kept import-light: torch/torchvision are imported lazily so that
    ``list_samples`` (and the tests that use it) run in an environment without the
    deep-learning stack installed.
    """

    def __init__(self, cfg: DataConfig, split: str):
        self.cfg = cfg
        self.split = split
        self.samples = list_samples(cfg.root, cfg.category, split)
        self._transform = build_transform(cfg)
        self._mask_transform = build_mask_transform(cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        s = self.samples[idx]
        image = self._transform(Image.open(s.image_path).convert("RGB"))

        # Good images have no defect region: an all-zeros mask keeps the shape of
        # the batch uniform so pixel-level evaluation can stack every sample.
        if s.mask_path is not None:
            mask = self._mask_transform(Image.open(s.mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.cfg.image_size, self.cfg.image_size)

        return image, s.label, mask, str(s.image_path)


def build_dataloader(cfg: DataConfig, split: str, batch_size: int = 8, shuffle: bool = False):
    """Wrap ``MVTecDataset`` in a torch ``DataLoader``. Lazy torch import."""
    from torch.utils.data import DataLoader

    dataset = MVTecDataset(cfg, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
