"""Tests for the filesystem/label seam of data.py.

We build a tiny fake MVTec tree so the label logic -- especially "train sees good
only" and "defective test images get their mask" -- is checked without the 5 GB
download or torch.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_anomaly_lab.data import DEFECT_LABEL, GOOD_LABEL, list_samples


def _png(path: Path, color=(128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _build_fake_mvtec(root: Path, category: str = "widget") -> None:
    base = root / category
    # train: good only (plus a stray defect folder that MUST be ignored)
    _png(base / "train" / "good" / "000.png")
    _png(base / "train" / "good" / "001.png")
    _png(base / "train" / "scratch" / "999.png")  # trap: never used for training
    # test: good + two defect types
    _png(base / "test" / "good" / "000.png")
    _png(base / "test" / "scratch" / "010.png")
    _png(base / "test" / "bent" / "020.png")
    # ground-truth masks for the defective test images
    _png(base / "ground_truth" / "scratch" / "010_mask.png", color=(255, 255, 255))
    # note: 'bent' mask intentionally missing to test graceful None


def test_train_split_is_good_only(tmp_path):
    _build_fake_mvtec(tmp_path)
    samples = list_samples(tmp_path, "widget", "train")
    assert len(samples) == 2  # the scratch folder under train/ is ignored
    assert all(s.label == GOOD_LABEL for s in samples)
    assert all(s.defect_type == "good" for s in samples)
    assert all(s.mask_path is None for s in samples)


def test_test_split_labels_and_masks(tmp_path):
    _build_fake_mvtec(tmp_path)
    samples = list_samples(tmp_path, "widget", "test")
    by_type = {s.defect_type: s for s in samples}

    assert by_type["good"].label == GOOD_LABEL
    assert by_type["good"].mask_path is None

    assert by_type["scratch"].label == DEFECT_LABEL
    assert by_type["scratch"].mask_path is not None
    assert by_type["scratch"].mask_path.exists()

    # a defective image whose mask file is absent must still load, mask_path=None
    assert by_type["bent"].label == DEFECT_LABEL
    assert by_type["bent"].mask_path is None


def test_missing_split_raises(tmp_path):
    _build_fake_mvtec(tmp_path)
    try:
        list_samples(tmp_path, "widget", "validation")
    except FileNotFoundError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for a nonexistent split")


def test_samples_are_sorted_for_determinism(tmp_path):
    _build_fake_mvtec(tmp_path)
    samples = list_samples(tmp_path, "widget", "train")
    paths = [str(s.image_path) for s in samples]
    assert paths == sorted(paths)
