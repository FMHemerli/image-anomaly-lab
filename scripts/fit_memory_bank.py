"""Fit the PatchCore-lite memory bank on one category and evaluate it.

The full strong-method pipeline end to end: embed the good images, build the bank,
score the test split, report detection + localisation metrics, and drop the plots
and a reproducible JSON into outputs/. The fitted bank is saved so the Streamlit
app (and reruns) can load it instead of refitting.

Usage:
    python scripts/fit_memory_bank.py --category metal_nut
    python scripts/fit_memory_bank.py --category screw --backbone resnet18
    python scripts/fit_memory_bank.py --category metal_nut --coreset-ratio 0.25 --layers layer3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_anomaly_lab import (
    BackboneConfig,
    DataConfig,
    PatchCoreConfig,
    describe_torch,
    resolve_torch_device,
)
from image_anomaly_lab.backbones import PatchFeatureExtractor
from image_anomaly_lab.data import build_dataloader
from image_anomaly_lab.detectors.memory_bank import fit_memory_bank, score_split
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
BANKS = ROOT / "memory_bank"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig(root=args.root, category=args.category)
    backbone_cfg = BackboneConfig(name=args.backbone, layers=tuple(args.layers))
    pc_cfg = PatchCoreConfig(coreset_ratio=args.coreset_ratio)

    device = resolve_torch_device()
    print(f"Device: {describe_torch().as_dict()}")
    print(f"Fitting PatchCore-lite on '{args.category}' ({backbone_cfg.name}, layers={backbone_cfg.layers})")

    extractor = PatchFeatureExtractor(backbone_cfg, device)

    # Fit: good images only.
    train_loader = build_dataloader(data_cfg, "train", batch_size=8, shuffle=False)
    bank, grid_hw = fit_memory_bank(extractor, train_loader, pc_cfg)
    bank_path = bank.save(BANKS / f"{args.category}.npz")
    print(f"Memory bank: {bank.bank.shape[0]} patches (grid {grid_hw}) -> {bank_path}")

    # Score the test split.
    test_loader = build_dataloader(data_cfg, "test", batch_size=8, shuffle=False)
    res = score_split(extractor, bank, test_loader, data_cfg.image_size)

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
        plot_roc(res["labels"], res["image_scores"], OUTPUTS / f"{args.category}_patchcore_roc.png")
        plot_score_histogram(
            res["labels"], res["image_scores"], threshold, OUTPUTS / f"{args.category}_patchcore_hist.png"
        )
        plot_heatmap_panel(
            res["paths"], res["anomaly_maps"], res["masks"], res["labels"],
            OUTPUTS / f"{args.category}_patchcore_heatmaps.png", data_cfg.image_size,
        )
        print(f"Plots -> {OUTPUTS}")

    save_run(
        OUTPUTS / f"{args.category}_patchcore.json",
        method="patchcore-lite",
        category=args.category,
        metrics={**metrics, "threshold": threshold},
        config={"data": data_cfg, "backbone": backbone_cfg, "patchcore": pc_cfg},
    )
    extractor.close()


if __name__ == "__main__":
    main()
