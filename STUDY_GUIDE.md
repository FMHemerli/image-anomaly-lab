# Study Guide — learn by predicting, running, and checking

This repo is built to be poked at. The loop for every experiment below is the
same, and the **prediction step is the one that actually teaches** — don't skip it:

1. **Read** the concept (each experiment names the module docstring to read).
2. **Predict** what the change does to the metrics or the heatmap — out loud or on paper.
3. **Run** the command.
4. **Check** against your prediction. A wrong prediction is the best outcome:
   that's exactly where your mental model needed fixing.
5. **Peek** at the spoiler only after you've formed your own explanation.

Setup, if you haven't already:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision
python scripts/download_data.py --category metal_nut
```

The numbers in spoilers are indicative for `metal_nut` on the default config; your
exact values will vary a little with the random coreset seed and torchvision
weights, but the *directions* and *gaps* are the lesson and should hold.

---

## Part 1 — The paradigm: training on good parts only

*Read first: `scripts/download_data.py` (the MVTec split) and the module docstring
of `src/image_anomaly_lab/detectors/memory_bank.py`.*

### 1.1 The baseline run

```bash
python scripts/fit_memory_bank.py --category metal_nut
```

Just observe. Note there is **no training loop** — the "fit" is a forward pass over
good images plus building a nearest-neighbour index. Open
`outputs/metal_nut_patchcore_heatmaps.png` and check: does the heat land on the
actual defects?

<details><summary>Spoiler — what you should see</summary>

Image AUROC around **0.97–0.99**, pixel AUROC around **0.97+**, and heatmaps that
light up on the scratches/bends. All of that from a *frozen* ImageNet backbone that
was never shown a single metal_nut, let alone a defect. The only "learning" was
memorising what normal patches look like. That's the headline: for this problem,
*remembering normal* beats *training a classifier* — and you couldn't train a defect
classifier anyway, because you have no labelled defects.
</details>

---

## Part 2 — Which backbone layers carry the signal?

*Read first: `BackboneConfig` in `src/image_anomaly_lab/config.py` and the alignment
note in `src/image_anomaly_lab/backbones.py`.*

### 2.1 Drop to a single deep layer

```bash
python scripts/fit_memory_bank.py --category metal_nut --layers layer3
```

**Predict first:** `layer3` is deeper and coarser than `layer2`. What happens to
the *heatmap resolution*? What happens to pixel AUROC vs image AUROC?

<details><summary>Spoiler</summary>

Image AUROC usually barely moves — deep features are plenty to say *whether* a part
is odd. But the heatmap gets **blockier** and pixel AUROC/PRO typically drop,
because `layer3`'s grid is half the resolution of `layer2`'s, so each patch covers
a bigger chunk of the image and localisation coarsens. The lesson: **detection and
localisation are different jobs.** Mid-level features (`layer2`) buy you the crisp
*where*; deep features are enough for the *whether*. Combining them — the default —
gets both.
</details>

### 2.2 Try the lighter backbone

```bash
python scripts/fit_memory_bank.py --category metal_nut --backbone resnet18
```

**Predict:** faster, smaller bank vectors. Does accuracy collapse?

<details><summary>Spoiler</summary>

It drops a little but stays strong — resnet18 is a smaller feature space, so
distances are slightly less discriminative, but the method is robust to the
backbone choice. Good to know when you care about speed on a real line.
</details>

---

## Part 3 — How small can the memory bank get?

*Read first: the coreset note in `src/image_anomaly_lab/detectors/memory_bank.py`.*

### 3.1 Shrink the coreset

```bash
python scripts/fit_memory_bank.py --category metal_nut --coreset-ratio 0.25
python scripts/fit_memory_bank.py --category metal_nut --coreset-ratio 0.01
```

**Predict first:** the default keeps 10% of patches. At 25% you keep more; at 1%,
far fewer. Where do you expect AUROC to finally start dropping — and why later than
you'd think?

<details><summary>Spoiler</summary>

Accuracy holds remarkably flat from 100% down to a few percent, then falls off a
cliff. Neighbouring patches in good images are nearly duplicates, so the bank is
hugely redundant — throwing most of it away costs almost nothing until you're so
sparse that a normal test patch no longer has a close match and starts looking
anomalous (false positives rise). This redundancy is exactly why PatchCore's greedy
*farthest-point* coreset (a noted extension) does even better than the random
subsample: it spends the budget on spread-out, non-redundant patches.
</details>

---

## Part 4 — Why the autoencoder struggles

*Read first: the "why it's the weak method" note in
`src/image_anomaly_lab/detectors/autoencoder.py`.*

### 4.1 Train the baseline and compare

```bash
python scripts/train_autoencoder.py --category metal_nut
python scripts/compare.py --category metal_nut
```

**Predict first:** where will the gap between AE and PatchCore be *largest* —
image AUROC (detection) or PRO (localisation)? Why?

<details><summary>Spoiler</summary>

The gap is usually widest on **PRO/pixel AUROC**. A conv autoencoder that learned to
rebuild all the edges of a good part happily rebuilds a scratch too (it's just more
edges), so the reconstruction error stays low *exactly on the defect* — the one
place you needed it high. It can still sometimes flag that *something* is off at the
image level, but it points at the wrong pixels. PatchCore has no such loophole: a
scratch patch simply has no near neighbour in the bank. This is the whole argument
for the memory-bank approach in one comparison.
</details>

### 4.2 Tighten the bottleneck

```bash
python scripts/train_autoencoder.py --category metal_nut --latent-dim 16
```

**Predict:** a much smaller latent forces worse reconstruction everywhere. Better or
worse anomaly detection?

<details><summary>Spoiler</summary>

Often a wash, sometimes slightly better *contrast*: a tighter bottleneck can't
memorise fine detail, so it reconstructs everything a bit blurrily — including
defects — which can raise error at the defect relative to the background. But it
also raises error on legitimate fine texture, hurting good parts. Autoencoders live
on this knife-edge; there's no bottleneck size that makes the loophole go away. That
tension is the point.
</details>

---

## Part 5 — Drawing the pass/fail line

*Read first: `threshold_youden` and `threshold_good_percentile` in
`src/image_anomaly_lab/evaluation.py`.*

### 5.1 Two ways to choose a threshold

The fit script prints the Youden's-J threshold and the score histogram
(`outputs/metal_nut_patchcore_hist.png`) shows where it lands. Youden's J uses the
*labels* (it needs defect examples); the percentile rule uses only good scores.

**Predict:** in a real factory where you have thousands of good parts but few known
defects, which rule is actually deployable? What does each optimise?

<details><summary>Spoiler</summary>

Youden's J maximises TPR − FPR: the best neutral trade-off, but it **needs labelled
defects** to compute — which you often don't have up front. The good-percentile rule
("flag anything above the 99th percentile of known-good parts") needs **no defects
at all**, matching deployment, at the cost of not directly optimising recall. Real
lines usually start from the percentile rule and tune it by the business cost of a
miss versus a false alarm. Look at the histogram overlap: every error lives in that
overlap region, and moving the line just trades false alarms for missed defects.
</details>

---

## Where to go next

- Run a second category: `python scripts/fit_memory_bank.py --category screw`.
- Implement the greedy coreset and beat the random subsample at low ratios.
- Add PaDiM (per-position Gaussian + Mahalanobis) as a third detector and fold it
  into `compare.py`.
