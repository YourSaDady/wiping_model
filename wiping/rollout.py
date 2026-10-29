"""Roll out the scripted program (or any policy) and log at the control rate.

Logged at every control step t (20 Hz), as arrays with a leading time axis:
    t, x (tcp pos), xdot (tcp vel), f (force of the table on the tool),
    f_n, x_d (target held over [t, t+1)), u (uncovered fraction after sample t),
    phase, q, v (iiwa joints)
plus the final sample N (so labels at step N-1 can use sample N): the state
arrays have N+1 rows, x_d and phase have N rows.
"""

from __future__ import annotations

import time

import numpy as np

from .config import EpisodeConfig
from .coverage import CoverageGrid
from .expert import DONE, WIPE, ScriptedWiper

STATE_KEYS = ("t", "x", "xdot", "f", "q", "v")


def make_env(cfg: EpisodeConfig, start_tcp, meshcat=None):
    if cfg.sim.backend == "drake":
        from .drake_env import DrakeWipeEnv

        return DrakeWipeEnv(cfg, start_tcp, meshcat=meshcat)
    if cfg.sim.backend == "pointmass":
        from .pointmass_env import PointMassWipeEnv

        return PointMassWipeEnv(cfg, start_tcp)
    raise ValueError(f"unknown backend {cfg.sim.backend!r}")


def policy_obs(cfg: EpisodeConfig, raw: dict) -> dict:
    """Observation a learned policy may use: (x, xdot, f) + region (p0, L) + n."""
    return dict(t=raw["t"], x=raw["x"], xdot=raw["xdot"], f=raw["f"],
                p0=cfg.region.p0(cfg.table), L=cfg.region.L, n=cfg.table.normal())


def run_episode(cfg: EpisodeConfig, policy=None, meshcat=None, record=False, T_max=None, verbose=False,
                stop_when_done=True):
    """Run one episode.

    policy=None   -> the scripted wiping program (logs its phase)
    policy=callable(obs_dict) -> x_d (closed loop; obs from `policy_obs`)
    policy=array (N x 3)       -> open-loop replay of a target sequence
    Returns a dict of numpy arrays (see module doc) + "metrics" + "cfg".
    """
    expert = ScriptedWiper(cfg)
    start = expert.start_tcp()
    env = make_env(cfg, start, meshcat=meshcat)
    if meshcat is not None:
        from .viz import draw_scene_overlays

        draw_scene_overlays(meshcat, cfg, expert.waypoints)
    if record and hasattr(env, "start_recording"):
        env.start_recording()

    grid = CoverageGrid(cfg)
    if T_max is None:
        T_max = cfg.sim.T_max if cfg.sim.T_max is not None else 1.3 * expert.expected_duration() + 2.0
    if policy is not None and not callable(policy):
        T_max = len(policy) / cfg.sim.rate
    n_max = int(round(T_max * cfg.sim.rate))
    n_vec = cfg.table.normal()

    rec = {k: [] for k in STATE_KEYS + ("x_d", "phase", "u")}
    t0 = time.time()
    for k in range(n_max):
        raw = env.observe()
        u = grid.update(raw["x"], raw["f"])
        if policy is None:
            x_d, phase = expert(raw)
        elif callable(policy):
            x_d, phase = np.asarray(policy(policy_obs(cfg, raw)), float), -1
        else:
            x_d, phase = np.asarray(policy[k], float), -1
        for key in STATE_KEYS:
            rec[key].append(np.asarray(raw[key], float))
        rec["u"].append(u)
        rec["x_d"].append(np.asarray(x_d, float))
        rec["phase"].append(phase)
        env.step(x_d)
        if meshcat is not None and k % 10 == 0:
            from .viz import draw_trail

            draw_trail(meshcat, cfg, np.array(rec["x"]), np.array(rec["f"]))
        if stop_when_done and policy is None and phase == DONE:
            break
    raw = env.observe()
    rec["u"].append(grid.update(raw["x"], raw["f"]))
    for key in STATE_KEYS:
        rec[key].append(np.asarray(raw[key], float))
    if record and hasattr(env, "stop_recording"):
        env.stop_recording()

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["phase"] = out["phase"].astype(np.int8)
    out["f_n"] = out["f"] @ n_vec
    out["K"], out["D"] = env.K, env.D
    out["clean_image"] = grid.image().copy()
    out["wall_time"] = time.time() - t0
    out["cfg"] = cfg
    out["metrics"] = episode_metrics(cfg, out, expert_done=(policy is None and expert.phase == DONE))
    if meshcat is not None:
        from .viz import draw_trail

        draw_trail(meshcat, cfg, out["x"], out["f"])
    if verbose:
        m = out["metrics"]
        print(f"[{cfg.name}] success={m['success']}  u_T={m['u_T']:.3f}  max f_n={m['fn_max']:.1f} N  "
              f"wipe f_n={m['fn_wipe_mean']:.2f}+-{m['fn_wipe_std']:.2f} N  "
              f"sim {out['t'][-1]:.1f}s in {out['wall_time']:.1f}s wall")
    return out


def episode_metrics(cfg: EpisodeConfig, ep: dict, expert_done: bool = True) -> dict:
    """Proposal success test: u_T <= eps_u and max_t f_n <= f_max (= 2 f_des)."""
    fn = ep["f_n"]
    u_T = float(ep["u"][-1])
    wipe = ep["phase"] == WIPE
    fn_w = fn[:-1][wipe] if wipe.any() else fn[:-1]
    m = dict(
        u_T=u_T,
        fn_max=float(fn.max()),
        fn_wipe_mean=float(fn_w.mean()),
        fn_wipe_std=float(fn_w.std()),
        fn_wipe_min=float(fn_w.min()),
        frac_wipe_below_fmin=float((fn_w < cfg.f_min()).mean()),
        T=float(ep["t"][-1]),
        n_steps=int(len(ep["x_d"])),
        reached_done=bool(expert_done),
        lifted=bool(fn[-1] < cfg.expert.f_contact),
    )
    m["success"] = bool(u_T <= cfg.success.eps_u and m["fn_max"] <= cfg.f_max())
    if ep["phase"].max() >= 0:  # scripted run: also require it to have finished the plan
        m["success"] = m["success"] and m["reached_done"]
    return m
