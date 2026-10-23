"""Episode runners: scripted expert, open-loop target replay, hybrid baseline.

Every runner returns an episode dict with 20 Hz arrays
  t (T+1), x, xdot, f, q (T+1, .), u (T+1) uncovered fraction so far,
  x_d (T, 3) commanded targets (impedance runners only), phase (T) expert phase,
plus 'metrics' (see WipeEnv.metrics) and 'scene' (dict).
"""
from __future__ import annotations

import numpy as np

from .config import DT_POLICY, MAIN_GAINS, T_MAX, Gains
from .expert import PHASE_ID, ScriptedExpert
from .scene import Scene
from .sim import WipeEnv


def _pack(env: WipeEnv, obs_list, x_d=None, phase=None, extra=None):
    ep = {k: np.array([o[k] for o in obs_list]) for k in ("t", "x", "xdot", "f", "q", "u")}
    if x_d is not None:
        ep["x_d"] = np.array(x_d)
    if phase is not None:
        ep["phase"] = np.array(phase, dtype=np.int8)
    ep["metrics"] = env.metrics()
    ep["scene"] = env.scene.to_dict()
    ep["samples"] = np.array(env.samples, dtype=np.float32)   # 100 Hz (t, u1, u2, gap, fn)
    if env.record_q:
        ep["q_hi"] = np.array(env.q_log, dtype=np.float32)
    if extra:
        ep.update(extra)
    return ep


def run_expert(scene: Scene, gains: Gains = MAIN_GAINS, record_q=False, max_steps=None, **expert_kw):
    env = WipeEnv(scene, gains, record_q=record_q)
    ex = ScriptedExpert(scene, gains, **expert_kw)
    obs = env.observe()
    obs_list, xds, phases = [obs], [], []
    max_steps = max_steps or int(round(T_MAX / DT_POLICY))
    while not ex.done and env.k < max_steps:
        xd = ex.act(obs)
        if ex.done:          # expert finished: no command for this step
            break
        phases.append(PHASE_ID[ex.phase])
        xds.append(xd)
        obs = env.step(xd)
        obs_list.append(obs)
    return _pack(env, obs_list, xds, phases, extra={"expert_done": ex.done})


def replay_targets(scene: Scene, targets: np.ndarray, gains: Gains = MAIN_GAINS, extra_steps: int = 0,
                   record_q=False):
    """Open-loop replay of a target sequence (one target per 20 Hz step).
    After the sequence ends, the last target is held for extra_steps."""
    env = WipeEnv(scene, gains, record_q=record_q)
    obs_list, xds = [env.observe()], []
    seq = list(targets) + [targets[-1]] * extra_steps
    for xd in seq:
        xds.append(xd)
        obs_list.append(env.step(xd))
    return _pack(env, obs_list, xds)


def run_hybrid(scene: Scene, gains: Gains = MAIN_GAINS, record_q=False, t_max=T_MAX, **hybrid_kw):
    env = WipeEnv(scene, gains, controller="hybrid", record_q=record_q, hybrid_kwargs=hybrid_kw)
    obs_list, modes = [env.observe()], []
    while env.hybrid_mode() != 4 and env.k * DT_POLICY < t_max - 1e-9:
        modes.append(env.hybrid_mode())
        obs_list.append(env.step())
    return _pack(env, obs_list, None, modes)
