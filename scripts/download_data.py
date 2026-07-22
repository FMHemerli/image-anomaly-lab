"""Fetch one category of the MVTec AD dataset and verify its layout.

WHAT IS MVTec AD
    The reference benchmark for unsupervised industrial anomaly detection. For
    each object category it gives you a very particular split -- and that split
    *is* the whole paradigm of this project:

        <category>/
          train/good/          <- ONLY defect-free images. This is all you train on.
          test/good/           <- defect-free test images (should score LOW)
          test/<defect_type>/  <- e.g. scratch, bent, color (should score HIGH)
          ground_truth/<defect_type>/  <- per-pixel masks of where the defect is

    You never show the model a defect during training. It learns "what normal
    looks like" and flags departures -- exactly the situation with worn/failed
    automotive parts, where good examples are plentiful and every failure is
    different.

WHY A SCRIPT INSTEAD OF A ONE-LINER
    The MVTec download host changes occasionally and the full archive is ~5 GB.
    This script is defensive: it skips work if the data is already there, tries a
    best-effort direct download, always validates the resulting folder structure,
    and -- if anything is missing -- prints exact manual steps instead of leaving
    you with a half-extracted mess.

Usage:
    python scripts/download_data.py --category metal_nut
    python scripts/download_data.py --category screw --root data/mvtec
    python scripts/download_data.py --category metal_nut --url <direct-tar.xz-url>
    python scripts/download_data.py --category metal_nut --archive ~/Downloads/metal_nut.tar.xz
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# Best-effort direct download URLs for the per-category archives, hosted on the
# public share MVTec uses for the dataset. These CAN change or rate-limit; if a
# download fails, the script falls back to manual instructions rather than
# pretending it worked. You can always override with --url or --archive.
# Primary source: public Hugging Face mirrors that host a category in the exact
# MVTec folder layout (train/good, test/<defect>, ground_truth/<defect>). These are
# far more durable than MVTec's own rotating tar.xz share links, which expire.
HF_REPOS = {
    "metal_nut": "MSherbinii/mvtec-ad-metal-nut",
}

# Legacy fallback: MVTec's per-category tar.xz share. Often 404s (the share id
# rotates); kept only so --url/--archive still have a code path. Prefer HF above.
_MVTEC_SHARE = (
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094"
)
CATEGORY_URLS = {
    "metal_nut": f"{_MVTEC_SHARE}/metal_nut.tar.xz",
    "screw": f"{_MVTEC_SHARE}/screw.tar.xz",
}

MANUAL_HELP = """\
Could not obtain the data automatically. To get it by hand:

  1. Open https://www.mvtec.com/company/research/datasets/mvtec-ad and accept
     the (free, research) license.
  2. Download the archive for category '{category}' (or the full dataset).
  3. Extract it so that this path exists:
         {expected}
  4. Re-run this script -- it will detect the data and just validate it, or
     point it straight at the file you downloaded:
         python scripts/download_data.py --category {category} --archive <path-to.tar.xz>
"""


def category_root(root: Path, category: str) -> Path:
    return root / category


def is_valid_category(cat_dir: Path) -> bool:
    """A category is usable only if the good-training split actually has images.

    We check the one folder the whole method depends on -- train/good -- rather
    than just the directory existing, so a partial/aborted extraction is caught.
    """
    good = cat_dir / "train" / "good"
    return good.is_dir() and any(good.glob("*.png"))


def download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a simple progress line (no extra dependencies)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100 * done / total_size
        print(f"\r  downloading {dest.name}: {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)  # noqa: S310 - trusted dataset host
    print()


def download_from_hf(repo: str, category: str, root: Path) -> None:
    """Download one category from a public HF dataset mirror, file by file.

    Reads the repo's file tree via the public HF API and pulls every file under
    ``<category>/`` to ``root/<path>``, reproducing the MVTec layout exactly. No
    huggingface_hub dependency -- just the JSON tree plus ``resolve/main`` URLs.
    """
    import json

    api = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(api, timeout=30) as resp:  # noqa: S310 - trusted host
        entries = json.load(resp)

    files = [
        e["path"] for e in entries
        if e.get("type") == "file" and e["path"].startswith(f"{category}/")
    ]
    if not files:
        raise RuntimeError(f"HF repo '{repo}' has no files under '{category}/'")

    base = f"https://huggingface.co/datasets/{repo}/resolve/main"
    print(f"Downloading {len(files)} files from Hugging Face mirror '{repo}'")
    for i, rel in enumerate(files, 1):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            urllib.request.urlretrieve(f"{base}/{rel}", dest)  # noqa: S310 - trusted host
        if i % 50 == 0 or i == len(files):
            print(f"\r  {i}/{len(files)} files", end="", flush=True)
    print()


def extract(archive: Path, root: Path) -> None:
    """Extract a .tar.xz into ``root``. Guards against path-traversal entries."""
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(root)  # noqa: S202 - members validated just above


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="metal_nut", help="MVTec category, e.g. metal_nut, screw")
    parser.add_argument("--root", default="data/mvtec", help="where categories are stored")
    parser.add_argument("--hf-repo", default=None, help="HF dataset repo to pull the category from (overrides the registry)")
    parser.add_argument("--url", default=None, help="override direct download URL (.tar.xz)")
    parser.add_argument("--archive", default=None, help="use an already-downloaded .tar.xz instead of downloading")
    args = parser.parse_args()

    root = Path(args.root)
    cat_dir = category_root(root, args.category)
    expected = cat_dir / "train" / "good"

    # 1. Already there? Validate and stop -- downloads are expensive.
    if is_valid_category(cat_dir):
        n = len(list(expected.glob("*.png")))
        print(f"OK: '{args.category}' already present at {cat_dir} ({n} good training images).")
        return

    # 2. Obtain the data. Preference order: an explicit local archive, then the
    #    Hugging Face mirror (primary -- durable), then the legacy tar.xz URL.
    try:
        hf_repo = args.hf_repo or HF_REPOS.get(args.category)
        if args.archive:
            archive_path = Path(args.archive).expanduser()
            if not archive_path.is_file():
                raise FileNotFoundError(f"--archive not found: {archive_path}")
            print(f"Extracting {archive_path.name} ...")
            extract(archive_path, root)
        elif args.url:
            archive_path = root / f"{args.category}.tar.xz"
            if not archive_path.is_file():
                print(f"Downloading '{args.category}' from {args.url}")
                download(args.url, archive_path)
            print(f"Extracting {archive_path.name} ...")
            extract(archive_path, root)
        elif hf_repo:
            download_from_hf(hf_repo, args.category, root)
        elif args.category in CATEGORY_URLS:
            archive_path = root / f"{args.category}.tar.xz"
            if not archive_path.is_file():
                print(f"Downloading '{args.category}' from {CATEGORY_URLS[args.category]}")
                download(CATEGORY_URLS[args.category], archive_path)
            print(f"Extracting {archive_path.name} ...")
            extract(archive_path, root)
        else:
            raise RuntimeError(
                f"no built-in source for '{args.category}'. Pass --hf-repo, --url, or --archive."
            )
    except Exception as exc:  # noqa: BLE001 - we intentionally convert any failure to guidance
        print(f"\n[!] {exc}\n", file=sys.stderr)
        print(MANUAL_HELP.format(category=args.category, expected=expected), file=sys.stderr)
        sys.exit(1)

    # 3. Validate the result no matter where the archive came from.
    if is_valid_category(cat_dir):
        n = len(list(expected.glob("*.png")))
        print(f"OK: extracted '{args.category}' to {cat_dir} ({n} good training images).")
    else:
        print(f"\n[!] Extraction finished but {expected} is missing or empty.\n", file=sys.stderr)
        print(MANUAL_HELP.format(category=args.category, expected=expected), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
