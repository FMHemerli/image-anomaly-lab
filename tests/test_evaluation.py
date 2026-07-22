"""Tests for the metric math -- built on cases with a known right answer."""

from __future__ import annotations

import numpy as np

from image_anomaly_lab.evaluation import (
    image_auroc,
    per_region_overlap,
    pixel_auroc,
    threshold_good_percentile,
    threshold_youden,
)


def test_image_auroc_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # defects strictly higher
    assert image_auroc(labels, scores) == 1.0


def test_image_auroc_chance():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])  # no information
    assert image_auroc(labels, scores) == 0.5


def test_pixel_auroc_rewards_maps_that_land_on_the_defect():
    # two images: mask marks a 4x4 defect block; the map is hot exactly there
    masks = np.zeros((2, 16, 16), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 8:12, 8:12] = 1.0
    maps = np.random.default_rng(0).random((2, 16, 16)).astype(np.float32) * 0.1
    maps[0, 2:6, 2:6] += 5.0
    maps[1, 8:12, 8:12] += 5.0
    assert pixel_auroc(masks, maps) > 0.99


def test_per_region_overlap_perfect_and_worthless():
    masks = np.zeros((1, 20, 20), dtype=np.float32)
    masks[0, 5:10, 5:10] = 1.0

    # a map that is high exactly on the region and low elsewhere -> PRO near 1
    good_map = np.zeros((1, 20, 20), dtype=np.float32)
    good_map[0, 5:10, 5:10] = 1.0
    assert per_region_overlap(masks, good_map) > 0.9

    # a flat map carries no localisation -> PRO clearly worse
    flat_map = np.full((1, 20, 20), 0.5, dtype=np.float32)
    assert per_region_overlap(masks, flat_map) < per_region_overlap(masks, good_map)


def test_threshold_youden_lands_between_classes():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = threshold_youden(labels, scores)
    assert 0.3 < t <= 0.7


def test_threshold_good_percentile_uses_only_good():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
    t = threshold_good_percentile(scores, labels, percentile=100.0)
    assert t == 4.0  # max of the good scores, defects ignored
