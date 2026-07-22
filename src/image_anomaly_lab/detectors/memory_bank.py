"""PatchCore-lite -- the strong detector: a memory of normal, scored by distance.

THE IDEA
    Fit: push every *good* image through the frozen backbone, collect one embedding
    per patch, and pile them all into a single "memory bank" of what normal looks
    like. That's the entire training -- no gradient steps, just remembering.

    Score: for a test image, embed its patches the same way and ask each one *how
    far is my nearest normal neighbour?* A clean patch sits right on top of a bank
    entry (tiny distance); a scratched patch has no close match (large distance).
    Those per-patch distances, folded back onto the image grid, ARE the anomaly
    heatmap; the image's overall score is the largest patch distance.

    This is metric learning's machinery -- embeddings plus nearest-neighbour search
    -- pointed at defects. If you've done triplet loss, you already know the core.

WHY IT'S numpy + sklearn HERE
    The scoring is a k-nearest-neighbour query, and ``sklearn.NearestNeighbors``
    does it on CPU with no torch dependency. That keeps this module -- the actual
    algorithm -- unit-testable with tiny synthetic embeddings, no GPU or pretrained
    weights required. Only the backbone (backbones.py) needs the GPU. On a huge
    bank you'd swap in ``torch.cdist`` on the GPU or faiss; that's a noted upgrade.

THE CORESET
    Keeping every patch of every good image is wasteful -- neighbouring patches are
    nearly identical. We keep a random ``coreset_ratio`` fraction. PatchCore's paper
    uses a greedy *farthest-point* coreset that spreads coverage better; random is
    the honest baseline and the STUDY_GUIDE asks you to push the ratio down and
    watch where accuracy finally breaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PatchCoreConfig


class MemoryBank:
    """A bank of normal patch embeddings with a nearest-neighbour anomaly score."""

    def __init__(self, cfg: PatchCoreConfig):
        self.cfg = cfg
        self.bank: np.ndarray | None = None  # (M, C) subsampled normal embeddings
        self._nn = None  # fitted sklearn NearestNeighbors index

    def fit(self, patch_vectors: np.ndarray) -> "MemoryBank":
        """Build the bank from all good-image patch embeddings ``(N, C)``.

        We subsample to a coreset and fit a kNN index over it. Subsampling uses a
        seeded RNG so the bank -- and therefore every downstream number -- is
        reproducible.
        """
        from sklearn.neighbors import NearestNeighbors

        patch_vectors = np.ascontiguousarray(patch_vectors, dtype=np.float32)
        n = len(patch_vectors)
        keep = max(1, int(round(n * self.cfg.coreset_ratio)))
        rng = np.random.default_rng(self.cfg.seed)
        # replace=False: a coreset is a subset, never the same patch twice.
        idx = rng.choice(n, size=keep, replace=False) if keep < n else np.arange(n)
        self.bank = patch_vectors[idx]

        # Ask for n_neighbors so scoring can average the k nearest (k=1 = pure
        # nearest neighbour). The index itself does the heavy lifting at query time.
        self._nn = NearestNeighbors(n_neighbors=self.cfg.n_neighbors, algorithm="auto")
        self._nn.fit(self.bank)
        return self

    def patch_scores(self, patch_vectors: np.ndarray) -> np.ndarray:
        """Distance of each query patch ``(M, C)`` to its nearest normal neighbour.

        Returns ``(M,)`` scores. With ``n_neighbors > 1`` we average the k nearest
        distances -- a mild smoothing that makes the score less jumpy on a single
        lucky match.
        """
        if self._nn is None:
            raise RuntimeError("MemoryBank is not fitted -- call fit() first.")
        patch_vectors = np.ascontiguousarray(patch_vectors, dtype=np.float32)
        distances, _ = self._nn.kneighbors(patch_vectors)  # (M, k)
        return distances.mean(axis=1)

    def save(self, path: str | Path) -> Path:
        """Persist the bank so the Streamlit app and later runs can reuse it.

        We save only the coreset array plus the config -- the sklearn index is
        rebuilt on load, which is cheap and avoids pickling a fitted estimator.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            bank=self.bank,
            coreset_ratio=self.cfg.coreset_ratio,
            n_neighbors=self.cfg.n_neighbors,
            blur_sigma=self.cfg.blur_sigma,
            seed=self.cfg.seed,
        )
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        from sklearn.neighbors import NearestNeighbors

        data = np.load(Path(path), allow_pickle=False)
        cfg = PatchCoreConfig(
            coreset_ratio=float(data["coreset_ratio"]),
            n_neighbors=int(data["n_neighbors"]),
            blur_sigma=float(data["blur_sigma"]),
            seed=int(data["seed"]),
        )
        obj = cls(cfg)
        obj.bank = data["bank"].astype(np.float32)
        obj._nn = NearestNeighbors(n_neighbors=cfg.n_neighbors, algorithm="auto").fit(obj.bank)
        return obj


def patch_scores_to_map(
    scores: np.ndarray, grid_hw: tuple[int, int], image_size: int, blur_sigma: float
) -> np.ndarray:
    """Fold ``(H*W,)`` patch scores back to an image-sized, smoothed anomaly map.

    Two steps, each with a reason:
      - *upsample* the H×W score grid to the input resolution so the heatmap lines
        up with the actual pixels (order=1 = bilinear, no blocky artefacts);
      - *Gaussian blur* it, because a lone high patch is usually noise; a real
        defect lights up a neighbourhood. ``blur_sigma`` sets how much we trust a
        single patch versus its surroundings.
    """
    from scipy.ndimage import gaussian_filter, zoom

    h, w = grid_hw
    grid = scores.reshape(h, w)
    zoom_factors = (image_size / h, image_size / w)
    upsampled = zoom(grid, zoom_factors, order=1)
    return gaussian_filter(upsampled, sigma=blur_sigma)


def image_score_from_map(anomaly_map: np.ndarray) -> float:
    """One number per image: the peak of its anomaly map.

    Max, not mean: a part with a single small scratch is still defective, and a
    mean would drown that peak in the sea of normal pixels around it.
    """
    return float(anomaly_map.max())


# --- Orchestration: backbone + bank over a dataloader --------------------------
# These glue the pieces above to real images. torch is imported lazily so the
# algorithmic core (fit/patch_scores) stays importable and testable without it.


def collect_patch_vectors(extractor, dataloader):
    """Run every batch through the backbone and stack all patch embeddings.

    Returns ``(vectors, grid_hw)`` where ``vectors`` is ``(N_total_patches, C)`` as
    a numpy array and ``grid_hw`` is the (H, W) of the feature grid -- needed later
    to rebuild anomaly maps. Used on the good-only training split to build the bank.
    """
    import torch

    chunks = []
    grid_hw = None
    with torch.no_grad():
        for images, *_ in dataloader:
            feats = extractor(images)  # (B, C, H, W)
            vectors, grid_hw = extractor.to_patch_vectors(feats)
            chunks.append(vectors.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0), grid_hw


def fit_memory_bank(extractor, dataloader, cfg: PatchCoreConfig):
    """Convenience: collect good-image patches and fit a ``MemoryBank`` on them."""
    vectors, grid_hw = collect_patch_vectors(extractor, dataloader)
    bank = MemoryBank(cfg).fit(vectors)
    return bank, grid_hw


def score_split(extractor, bank: "MemoryBank", dataloader, image_size: int):
    """Score a whole split, one image at a time.

    Returns a dict of parallel numpy arrays/lists ready for evaluation:
    ``image_scores`` (N,), ``labels`` (N,), ``anomaly_maps`` (N, image_size,
    image_size), ``masks`` (N, image_size, image_size), and ``paths``. Everything
    the ROC curves, pixel metrics and heatmap panels need.
    """
    import torch

    image_scores, labels, maps, masks, paths = [], [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_masks, batch_paths in dataloader:
            feats = extractor(images)
            vectors, grid_hw = extractor.to_patch_vectors(feats)
            scores = bank.patch_scores(vectors.cpu().numpy())  # (B*H*W,)

            # Split the flat per-patch scores back into per-image grids.
            per_image = scores.reshape(len(images), grid_hw[0] * grid_hw[1])
            for i in range(len(images)):
                amap = patch_scores_to_map(
                    per_image[i], grid_hw, image_size, bank.cfg.blur_sigma
                )
                maps.append(amap)
                image_scores.append(image_score_from_map(amap))
                labels.append(int(batch_labels[i]))
                masks.append(batch_masks[i, 0].cpu().numpy())
                paths.append(batch_paths[i])

    return {
        "image_scores": np.asarray(image_scores),
        "labels": np.asarray(labels),
        "anomaly_maps": np.asarray(maps),
        "masks": np.asarray(masks),
        "paths": paths,
    }
