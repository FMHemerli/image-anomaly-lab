"""Scoring the scorers -- detection, localisation, thresholds, and pictures.

Everything here takes plain numpy arrays (scores, labels, anomaly maps, masks), so
it works identically for the memory bank and the autoencoder, and runs without a
GPU. Three questions get answered:

  1. Does the image-level score separate good from defective?  -> image AUROC.
  2. Does the heatmap land ON the defect?                      -> pixel AUROC, PRO.
  3. Where do we draw the pass/fail line?                      -> threshold selection.

Plus the plots that make all of that legible: ROC, score histogram, heatmap panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --- Detection: image-level ----------------------------------------------------


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC for good (0) vs defective (1) at the image level.

    AUROC answers "across every possible threshold, how well does the score rank a
    random defect above a random good part?" 1.0 is perfect, 0.5 is a coin flip.
    It's threshold-free, which is why it's the headline metric before we ever pick
    a cutoff.
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


# --- Localisation: pixel-level -------------------------------------------------


def pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """Pixel-level AUROC: treat every pixel as a sample (defect vs not).

    This rewards a heatmap that is high exactly on the defect and low elsewhere.
    Note its blind spot -- defects are a tiny fraction of pixels, so a lazy map can
    score high just by being calm on the vast normal background. That's why we also
    report PRO, which weights each defect *region* equally regardless of its size.
    """
    from sklearn.metrics import roc_auc_score

    y_true = masks.reshape(-1).astype(np.int32)
    y_score = anomaly_maps.reshape(-1)
    if y_true.max() == 0:
        raise ValueError("no positive pixels in masks -- cannot compute pixel AUROC")
    return float(roc_auc_score(y_true, y_score))


def per_region_overlap(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    fpr_limit: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """PRO: normalised area under the (per-region overlap vs. FPR) curve up to a FPR.

    THE MOTIVATION
        Pixel AUROC lets a few large defects dominate. PRO fixes that by scoring
        each *connected defect region* on its own and averaging regions equally --
        a tiny scratch counts as much as a big dent. For a range of thresholds we
        measure, per region, the fraction of it the heatmap covers (overlap), and
        the false-positive rate on normal pixels. We integrate overlap over FPR in
        ``[0, fpr_limit]`` (MVTec uses 0.3) and normalise, so 1.0 = perfect.
    """
    from scipy.ndimage import label

    # Pre-label every ground-truth defect into connected components once.
    regions: list[tuple[int, np.ndarray]] = []  # (image_index, boolean region mask)
    for i, m in enumerate(masks):
        labelled, n = label(m > 0.5)
        for r in range(1, n + 1):
            regions.append((i, labelled == r))

    if not regions:
        raise ValueError("no defect regions found in masks -- cannot compute PRO")

    normal_pixels = masks <= 0.5  # where a positive prediction is a false positive

    lo, hi = float(anomaly_maps.min()), float(anomaly_maps.max())
    thresholds = np.linspace(hi, lo, num_thresholds)  # high->low: FPR grows

    pros, fprs = [], []
    for t in thresholds:
        pred = anomaly_maps >= t
        # Mean per-region overlap: each region weighted equally.
        overlaps = [pred[i][region].mean() for i, region in regions]
        pros.append(float(np.mean(overlaps)))
        fp = np.logical_and(pred, normal_pixels).sum()
        fprs.append(float(fp / normal_pixels.sum()))

    # Anchor the curve at (fpr=0, pro=0): an infinitely high threshold predicts
    # nothing -- zero false positives and zero coverage. Without this anchor a
    # detector that only ever operates at high FPR would be scored as if it covered
    # regions for free inside the budget.
    fprs = np.concatenate([[0.0], np.asarray(fprs)])
    pros = np.concatenate([[0.0], np.asarray(pros)])

    # For each distinct FPR keep the BEST coverage achievable at that budget, then
    # interpolate onto a dense grid over [0, fpr_limit] and take the normalised area.
    uniq_fpr = np.unique(fprs)
    best_pro = np.array([pros[fprs == f].max() for f in uniq_fpr])

    grid = np.linspace(0.0, fpr_limit, 256)
    pro_on_grid = np.interp(grid, uniq_fpr, best_pro)
    # np.trapz was renamed np.trapezoid in numpy 2.0; support both.
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapezoid(pro_on_grid, grid) / fpr_limit)


# --- Thresholding --------------------------------------------------------------


def threshold_youden(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pick the pass/fail cutoff that maximises (TPR - FPR) -- Youden's J.

    This is the point on the ROC furthest from the chance diagonal: the best
    single trade-off between catching defects and not crying wolf. In a real plant
    you'd bias this by the cost of a miss vs a false alarm; Youden's J is the
    neutral default and a clean starting point.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_good_percentile(
    scores: np.ndarray, labels: np.ndarray, percentile: float = 99.0
) -> float:
    """Alternative cutoff: a high percentile of the GOOD parts' scores.

    "Flag anything above what 99% of known-good parts produce." Uses only good
    data, which matches deployment (you rarely have defect examples up front). The
    STUDY_GUIDE contrasts this with Youden's J on false-positive rate.
    """
    good = scores[labels == 0]
    return float(np.percentile(good, percentile))


# --- Plots (Agg backend, saved to outputs/) ------------------------------------


def _new_axes(figsize=(6, 5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=figsize)


def plot_roc(labels: np.ndarray, scores: np.ndarray, output_path: str | Path, title: str = "") -> Path:
    """ROC curve with the AUROC in the legend -- the detection summary in one image."""
    from sklearn.metrics import roc_auc_score, roc_curve

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt, (fig, ax) = _new_axes()
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title or "Image-level ROC -- good vs defect")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_score_histogram(
    labels: np.ndarray, scores: np.ndarray, threshold: float, output_path: str | Path
) -> Path:
    """Overlaid score histograms for good vs defect, with the chosen threshold line.

    The single most honest picture of a detector: you see the two distributions,
    their overlap (the errors live there), and exactly where the cutoff falls.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt, (fig, ax) = _new_axes(figsize=(7, 5))
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, label="good", color="#2F7A57")
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, label="defect", color="#B4351B")
    ax.axvline(threshold, color="black", ls="--", lw=1.5, label=f"threshold = {threshold:.3g}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.set_title("Where good and defect scores overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_heatmap_panel(
    paths: list[str],
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    image_size: int,
    n_examples: int = 4,
) -> Path:
    """A grid: original part | anomaly heatmap overlay | ground-truth mask.

    Picks a few defective examples so a reader can eyeball whether the heat lands
    on the actual defect -- the qualitative counterpart to the pixel metrics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    defect_idx = np.where(labels == 1)[0][:n_examples]
    if len(defect_idx) == 0:
        raise ValueError("no defective images to display in the heatmap panel")

    fig, axes = plt.subplots(len(defect_idx), 3, figsize=(9, 3 * len(defect_idx)))
    if len(defect_idx) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(defect_idx):
        img = Image.open(paths[idx]).convert("RGB").resize((image_size, image_size))
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("part")
        axes[row, 1].imshow(img)
        axes[row, 1].imshow(anomaly_maps[idx], cmap="jet", alpha=0.5)
        axes[row, 1].set_title("anomaly heatmap")
        axes[row, 2].imshow(masks[idx], cmap="gray")
        axes[row, 2].set_title("ground truth")
        for c in range(3):
            axes[row, c].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
