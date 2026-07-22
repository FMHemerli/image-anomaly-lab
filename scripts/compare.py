"""Put the two methods side by side, reading the JSON each script wrote.

No torch, no refitting -- this just loads outputs/<category>_patchcore.json and
outputs/<category>_ae.json and prints the comparison table. Run the two training
scripts first. The story it tells: PatchCore usually wins everywhere, and the gap
is widest on PRO (localisation), because that's exactly where the autoencoder's
tendency to reconstruct defects hurts most.

Usage:
    python scripts/compare.py --category metal_nut
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"

METRIC_KEYS = [("image_auroc", "image AUROC"), ("pixel_auroc", "pixel AUROC"), ("pro", "PRO")]


def load(category: str, suffix: str) -> dict | None:
    path = OUTPUTS / f"{category}_{suffix}.json"
    if not path.exists():
        print(f"[!] missing {path.name} -- run the corresponding script first.")
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="metal_nut")
    args = parser.parse_args()

    ae = load(args.category, "ae")
    pc = load(args.category, "patchcore")
    if ae is None or pc is None:
        return

    print(f"\nCategory: {args.category}\n")
    header = f"{'metric':<14}{'autoencoder':>14}{'patchcore-lite':>18}{'winner':>10}"
    print(header)
    print("-" * len(header))
    for key, label in METRIC_KEYS:
        a = ae["metrics"].get(key, float("nan"))
        p = pc["metrics"].get(key, float("nan"))
        winner = "patchcore" if p >= a else "autoenc."
        print(f"{label:<14}{a:>14.4f}{p:>18.4f}{winner:>10}")
    print()


if __name__ == "__main__":
    main()
