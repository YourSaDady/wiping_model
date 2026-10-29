"""Named table types and the train / held-out sampling distributions.

`make_config("flat_hard")` gives a ready EpisodeConfig. `sample_config(rng, split)`
draws a random table + region + speed from the training (or held-out) ranges.
Edit the ranges here; everything else reads them.
"""

from __future__ import annotations

import math

import numpy as np

from .config import EpisodeConfig, ExpertConfig, RegionConfig, TableConfig

DEG = math.pi / 180.0

# k_e values for "foam" .. "hard". Point-contact stiffness (N/m).
KE_SOFT, KE_MEDIUM, KE_HARD, KE_VERY_HARD = 2.0e3, 1.0e4, 5.0e4, 1.0e5

PRESETS = {
    # --- week 1: flat tables ------------------------------------------------
    "flat_soft": dict(tilt=0.0, k_e=KE_SOFT),
    "flat_medium": dict(tilt=0.0, k_e=KE_MEDIUM),
    "flat_hard": dict(tilt=0.0, k_e=KE_HARD),
    # --- training tilts -------------------------------------------------------
    "tilt10_medium": dict(tilt=10 * DEG, tilt_dir=0.0, k_e=KE_MEDIUM),
    "tilt10_hard": dict(tilt=10 * DEG, tilt_dir=math.pi / 2, k_e=KE_HARD),
    "tilt15_soft": dict(tilt=15 * DEG, tilt_dir=math.pi, k_e=KE_SOFT),
    # --- held out (test only) ---------------------------------------------
    "heldout_tilt25_medium": dict(tilt=25 * DEG, tilt_dir=math.pi / 4, k_e=KE_MEDIUM),
    "heldout_flat_veryhard": dict(tilt=0.0, k_e=KE_VERY_HARD),
}

TRAIN_PRESETS = ["flat_soft", "flat_medium", "flat_hard", "tilt10_medium", "tilt10_hard", "tilt15_soft"]
HELDOUT_PRESETS = ["heldout_tilt25_medium", "heldout_flat_veryhard"]


def make_config(preset: str = "flat_medium", backend: str = "drake", seed: int = 0, **over) -> EpisodeConfig:
    """EpisodeConfig for a named table type. Extra nested overrides via
    `table__mu=0.3`, `expert__speed=0.05`, ..."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = EpisodeConfig(table=TableConfig(**PRESETS[preset]), seed=seed, name=preset)
    cfg = cfg.replace(sim__backend=backend)
    return cfg.replace(**over) if over else cfg


# Training ranges. Held-out = steeper tilt and stiffer/softer k_e than any of these.
TRAIN_RANGES = dict(
    tilt=(0.0, 15 * DEG),
    log10_k_e=(math.log10(KE_SOFT), math.log10(KE_HARD)),
    mu=(0.3, 0.7),
    height=(0.08, 0.12),
    center_jitter=0.03,  # region center, +- in u and v (m)
    L=(0.10, 0.14),
    speed=(0.04, 0.08),
)
HELDOUT_RANGES = dict(TRAIN_RANGES, tilt=(18 * DEG, 25 * DEG), log10_k_e=(math.log10(KE_HARD), math.log10(KE_VERY_HARD)))


def sample_config(rng: np.random.Generator, split: str = "train", backend: str = "drake", flat_only: bool = False,
                  seed: int = 0) -> EpisodeConfig:
    R = TRAIN_RANGES if split == "train" else HELDOUT_RANGES
    tilt = 0.0 if flat_only else rng.uniform(*R["tilt"])
    table = TableConfig(
        height=rng.uniform(*R["height"]),
        tilt=tilt,
        tilt_dir=rng.uniform(-math.pi, math.pi),
        k_e=10 ** rng.uniform(*R["log10_k_e"]),
        mu=rng.uniform(*R["mu"]),
    )
    j = R["center_jitter"]
    region = RegionConfig(center_uv=(rng.uniform(-j, j), rng.uniform(-j, j)), L=rng.uniform(*R["L"]))
    expert = ExpertConfig(speed=rng.uniform(*R["speed"]))
    cfg = EpisodeConfig(table=table, region=region, expert=expert, seed=seed, name=f"{split}_{seed}")
    return cfg.replace(sim__backend=backend)
