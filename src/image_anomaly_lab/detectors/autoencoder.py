"""Reconstruction autoencoder -- the baseline that is *meant* to be beaten.

THE IDEA
    Train a convolutional autoencoder to compress a good image to a small latent
    vector and rebuild it, minimising reconstruction error on good parts only. The
    bet: it becomes good at rebuilding *normal* texture and bad at rebuilding
    anything it never saw, so at test time a high per-pixel reconstruction error
    marks a defect.

WHY IT'S THE WEAK METHOD (the actual lesson)
    Convolutional autoencoders generalise *too well*. A scratch is locally just
    "some edges", and an AE that learned to rebuild all the edges of a good part
    often rebuilds the scratch too -- so the error map stays quiet exactly where you
    needed it loud. Watching this fail, and comparing against the memory bank on
    the same part, is the point of keeping it in the lab. Shrinking ``latent_dim``
    forces a tighter bottleneck (worse reconstruction everywhere, but sometimes
    better *contrast* at defects) -- a STUDY_GUIDE knob.

    torch is imported at module top here because an autoencoder *is* a torch model;
    this module is only imported by the training script, never by the numpy core.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import AEConfig, DataConfig


class ConvAutoencoder(nn.Module):
    """A small symmetric conv encoder/decoder for 3x256x256 inputs.

    Four stride-2 downsamples take 256 -> 16 spatially; a 1x1 conv squeezes to
    ``latent_dim`` channels at the bottleneck; the decoder mirrors it back up. Kept
    deliberately modest -- a bigger net would only reconstruct defects more happily.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        def down(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
            )

        def up(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            down(3, 32), down(32, 64), down(64, 128), down(128, 256), nn.Conv2d(256, latent_dim, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 1),
            nn.ReLU(inplace=True),
            up(256, 128),
            up(128, 64),
            up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(data_cfg: DataConfig, ae_cfg: AEConfig, device) -> ConvAutoencoder:
    """Train on the good-only split with MSE loss. Returns the fitted model.

    Only ``train/good`` is used -- the same good-images-only discipline as the
    memory bank. We print the loss so you can see it plateau (and notice it never
    needs to be low to still fail at localisation).
    """
    from ..data import build_dataloader

    torch.manual_seed(ae_cfg.seed)
    loader = build_dataloader(data_cfg, "train", batch_size=ae_cfg.batch_size, shuffle=True)

    model = ConvAutoencoder(ae_cfg.latent_dim).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(ae_cfg.epochs):
        running = 0.0
        for images, *_ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), images)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
        print(f"  epoch {epoch + 1:3d}/{ae_cfg.epochs}  mse={running / len(loader.dataset):.5f}")

    return model.eval()


def score_split(model: ConvAutoencoder, data_cfg: DataConfig, split: str, device) -> dict:
    """Score a split, returning the SAME dict shape as memory_bank.score_split.

    The anomaly map is the per-pixel reconstruction error (mean over colour
    channels); the image score is that map's peak, matching the memory bank so the
    two methods are compared on equal footing.
    """
    from ..data import build_dataloader
    from .memory_bank import image_score_from_map

    loader = build_dataloader(data_cfg, split, batch_size=8, shuffle=False)

    image_scores, labels, maps, masks, paths = [], [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_masks, batch_paths in loader:
            recon = model(images.to(device)).cpu()
            # Per-pixel squared error, averaged across the 3 channels -> (B, H, W).
            err = ((recon - images) ** 2).mean(dim=1).numpy()
            for i in range(len(images)):
                maps.append(err[i])
                image_scores.append(image_score_from_map(err[i]))
                labels.append(int(batch_labels[i]))
                masks.append(batch_masks[i, 0].numpy())
                paths.append(batch_paths[i])

    return {
        "image_scores": np.asarray(image_scores),
        "labels": np.asarray(labels),
        "anomaly_maps": np.asarray(maps),
        "masks": np.asarray(masks),
        "paths": paths,
    }
