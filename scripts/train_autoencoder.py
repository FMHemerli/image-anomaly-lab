"""Train the reconstruction-autoencoder baseline and evaluate it.

Deliberately the weaker method (see detectors/autoencoder.py). Run it on the same
category as the memory bank and compare -- the gap, especially in localisation, is
the lesson. Same metrics, same plot style, same JSON format, so compare.py can put
the two side by side.

Usage:
    python scripts/train_autoencoder.py --category metal_nut
    python scripts/train_autoencoder.py --category metal_nut --epochs 50 --latent-dim 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_anomaly_lab import AEConfig, DataConfig, describe_torch, resolve_torch_device
from image_anomaly_lab.detectors.autoencoder import score_split, train_autoencoder
from image_anomaly_lab.evaluation import (
    image_auroc,
    per_region_overlap,
    pixel_auroc,
    plot_heatmap_panel,
    plot_roc,
    plot_score_histogram,
    threshold_youden,
)
from image_anomaly_lab.results import save_run

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig(root=args.root, category=args.category)
    ae_cfg = AEConfig(epochs=args.epochs, latent_dim=args.latent_dim)

    device = resolve_torch_device()
    print(f"Device: {describe_torch().as_dict()}")
    print(f"Training autoencoder on '{args.category}' (latent_dim={ae_cfg.latent_dim}, epochs={ae_cfg.epochs})")

    model = train_autoencoder(data_cfg, ae_cfg, device)
    res = score_split(model, data_cfg, "test", device)

    metrics = {
        "image_auroc": image_auroc(res["labels"], res["image_scores"]),
        "pixel_auroc": pixel_auroc(res["masks"], res["anomaly_maps"]),
        "pro": per_region_overlap(res["masks"], res["anomaly_maps"]),
    }
    threshold = threshold_youden(res["labels"], res["image_scores"])
    print(
        f"image AUROC={metrics['image_auroc']:.4f}  "
        f"pixel AUROC={metrics['pixel_auroc']:.4f}  PRO={metrics['pro']:.4f}  "
        f"threshold={threshold:.4g}"
    )

    if not args.no_plots:
        plot_roc(res["labels"], res["image_scores"], OUTPUTS / f"{args.category}_ae_roc.png")
        plot_score_histogram(
            res["labels"], res["image_scores"], threshold, OUTPUTS / f"{args.category}_ae_hist.png"
        )
        plot_heatmap_panel(
            res["paths"], res["anomaly_maps"], res["masks"], res["labels"],
            OUTPUTS / f"{args.category}_ae_heatmaps.png", data_cfg.image_size,
        )
        print(f"Plots -> {OUTPUTS}")

    save_run(
        OUTPUTS / f"{args.category}_ae.json",
        method="autoencoder",
        category=args.category,
        metrics={**metrics, "threshold": threshold},
        config={"data": data_cfg, "autoencoder": ae_cfg},
    )


if __name__ == "__main__":
    main()
