"""Tests for the PatchCore-lite core, using synthetic embeddings (no backbone).

The whole point of scoring on numpy/sklearn is that we can prove the algorithm
here: normal patches sit near the bank (low score), outliers sit far (high score).
"""

from __future__ import annotations

import numpy as np

from image_anomaly_lab.config import PatchCoreConfig
from image_anomaly_lab.detectors.memory_bank import (
    MemoryBank,
    image_score_from_map,
    patch_scores_to_map,
)


def _normal_cluster(n, dim, rng, center=0.0, spread=0.05):
    return (rng.standard_normal((n, dim)) * spread + center).astype(np.float32)


def test_outliers_score_higher_than_inliers():
    rng = np.random.default_rng(0)
    dim = 16
    bank_vectors = _normal_cluster(2000, dim, rng)  # tight normal cluster at origin
    bank = MemoryBank(PatchCoreConfig(coreset_ratio=0.5, n_neighbors=1)).fit(bank_vectors)

    inliers = _normal_cluster(50, dim, rng)
    outliers = _normal_cluster(50, dim, rng, center=5.0)  # far away = anomalous

    assert bank.patch_scores(inliers).mean() < bank.patch_scores(outliers).mean()
    # separation should be dramatic, not marginal
    assert bank.patch_scores(outliers).min() > bank.patch_scores(inliers).max()


def test_coreset_ratio_controls_bank_size():
    rng = np.random.default_rng(1)
    vectors = _normal_cluster(1000, 8, rng)
    bank = MemoryBank(PatchCoreConfig(coreset_ratio=0.1)).fit(vectors)
    assert bank.bank.shape[0] == 100  # 10% of 1000


def test_fit_is_reproducible_via_seed():
    rng = np.random.default_rng(2)
    vectors = _normal_cluster(500, 8, rng)
    b1 = MemoryBank(PatchCoreConfig(coreset_ratio=0.2, seed=7)).fit(vectors)
    b2 = MemoryBank(PatchCoreConfig(coreset_ratio=0.2, seed=7)).fit(vectors)
    assert np.array_equal(b1.bank, b2.bank)


def test_scoring_before_fit_raises():
    bank = MemoryBank(PatchCoreConfig())
    try:
        bank.patch_scores(np.zeros((3, 8), dtype=np.float32))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when scoring an unfitted bank")


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    vectors = _normal_cluster(400, 8, rng)
    bank = MemoryBank(PatchCoreConfig(coreset_ratio=0.25, n_neighbors=1)).fit(vectors)
    query = _normal_cluster(20, 8, rng, center=3.0)
    before = bank.patch_scores(query)

    path = bank.save(tmp_path / "bank.npz")
    reloaded = MemoryBank.load(path)
    after = reloaded.patch_scores(query)

    assert np.allclose(before, after)


def test_patch_scores_to_map_shape_and_peak():
    # a 4x4 grid, one hot patch -> upsampled/blurred map peaks near that corner
    grid = np.zeros(16, dtype=np.float32)
    grid[0] = 10.0
    amap = patch_scores_to_map(grid, (4, 4), image_size=32, blur_sigma=1.0)
    assert amap.shape == (32, 32)
    assert image_score_from_map(amap) > 0
    # the peak should be in the top-left quadrant where the hot patch was
    peak_r, peak_c = np.unravel_index(np.argmax(amap), amap.shape)
    assert peak_r < 16 and peak_c < 16
