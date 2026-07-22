"""Detectors: the two anomaly scorers we contrast.

- ``autoencoder`` -- the naive reconstruction baseline (the weak method).
- ``memory_bank`` -- PatchCore-lite, embeddings + nearest-neighbour (the strong one).

Both are imported lazily by the scripts to avoid importing torch just to read the
config, so nothing is re-exported here yet.
"""
