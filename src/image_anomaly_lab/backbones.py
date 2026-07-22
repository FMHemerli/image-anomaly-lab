"""The frozen feature backbone -- borrowing an ImageNet CNN's eyes.

THE IDEA
    We never train a network to recognise defects. Instead we take a CNN already
    trained on ImageNet and use its *intermediate* feature maps as a rich,
    general-purpose description of local image content. A patch of clean brushed
    metal and a patch with a scratch produce different feature vectors -- and that
    difference is all PatchCore needs.

WHY FORWARD HOOKS
    We want the output of ``layer2`` and ``layer3`` -- tensors buried *inside* the
    network, not the final classification. Rather than surgically rebuilding the
    model, we attach ``forward hooks``: little callbacks torch fires as data passes
    through a chosen layer, letting us grab its output as a side effect. This is
    the mechanism worth understanding, so it's a fill-in-the-blank in the notebook.

THE ALIGNMENT TRAP
    Deeper layers have coarser spatial grids (``layer3`` is half the height/width
    of ``layer2``). To stack them into one per-patch vector we must first upsample
    the deeper map to the shallower map's grid. Get this wrong and channels from
    different spatial locations get glued together -- silently, with no error, just
    worse results.
"""

from __future__ import annotations

from .config import BackboneConfig


class PatchFeatureExtractor:
    """Runs a frozen pretrained CNN and returns aligned, concatenated feature maps.

    Output of ``__call__`` is a tensor of shape ``(B, C, H, W)`` -- one C-dim
    embedding per spatial location ("patch"), where H and W are the grid of the
    shallowest tapped layer. Everything downstream (the memory bank) consumes this.
    """

    def __init__(self, cfg: BackboneConfig, device=None):
        import torch
        import torchvision.models as models

        self.cfg = cfg
        self.device = device or torch.device("cpu")

        # DEFAULT = the best available ImageNet weights for this architecture.
        model_fn = getattr(models, cfg.name)
        self.model = model_fn(weights="DEFAULT").to(self.device).eval()

        # Freeze: no gradients, ever. This is a feature *extractor*, not a trainee.
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register one forward hook per requested layer. Each hook drops that
        # layer's output into ``self._features`` keyed by name; __call__ reads them
        # after a forward pass and clears them.
        self._features: dict[str, "torch.Tensor"] = {}
        self._handles = []
        for name in cfg.layers:
            module = dict(self.model.named_modules())[name]
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, output):
            # Detach: we only want the values, never a path back for gradients.
            self._features[name] = output.detach()

        return hook

    def __call__(self, images):
        """Embed a batch. ``images`` is a normalised ``(B, 3, H0, W0)`` tensor.

        Steps: forward pass (hooks fire) -> take the shallowest layer's grid as the
        target size -> upsample every deeper map to it -> concatenate along
        channels -> return ``(B, C, H, W)``.
        """
        import torch
        import torch.nn.functional as F

        self._features.clear()
        with torch.no_grad():
            self.model(images.to(self.device))

        # Collect in the configured order so the channel layout is reproducible.
        maps = [self._features[name] for name in self.cfg.layers]

        # Align to the FIRST (shallowest, highest-resolution) map's spatial grid.
        target_hw = maps[0].shape[-2:]
        aligned = [maps[0]]
        for m in maps[1:]:
            aligned.append(F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False))

        return torch.cat(aligned, dim=1)  # concat over channels -> (B, C, H, W)

    @staticmethod
    def to_patch_vectors(feature_map):
        """Flatten ``(B, C, H, W)`` into ``((B*H*W), C)`` patch vectors, plus (H, W).

        The memory bank is a flat set of C-dim vectors -- one per patch, pooled
        across every image -- so this reshape is how a batch of feature maps
        becomes bank entries. Returning (H, W) lets the scorer fold per-patch
        scores back into a 2-D anomaly map.
        """
        b, c, h, w = feature_map.shape
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C). The permute-before-reshape
        # keeps each patch's C channels contiguous; reshaping without it interleaves
        # channels and locations -- the classic silent bug.
        vectors = feature_map.permute(0, 2, 3, 1).reshape(-1, c)
        return vectors, (h, w)

    def close(self):
        """Remove the hooks. Good hygiene if you build several extractors."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
