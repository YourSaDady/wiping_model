"""Scripted wiping program + data collection for the 6.4212 action-label project.

Importing the package does not import pydrake; only the "drake" backend does.
"""

from .config import (ControllerConfig, EpisodeConfig, ExpertConfig, RegionConfig, SimConfig, SuccessConfig,
                     TableConfig)
from .dataset import DemoWriter, load_demos, make_labels
from .expert import PHASE_NAMES, ScriptedWiper
from .presets import HELDOUT_PRESETS, PRESETS, TRAIN_PRESETS, make_config, sample_config
from .rollout import episode_metrics, policy_obs, run_episode

__all__ = [
    "ControllerConfig", "EpisodeConfig", "ExpertConfig", "RegionConfig", "SimConfig", "SuccessConfig", "TableConfig",
    "DemoWriter", "load_demos", "make_labels", "PHASE_NAMES", "ScriptedWiper", "PRESETS", "TRAIN_PRESETS",
    "HELDOUT_PRESETS", "make_config", "sample_config", "episode_metrics", "policy_obs", "run_episode",
]
