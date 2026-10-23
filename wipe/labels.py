"""Action labels built from one demonstration.

Label k is the target held over [t_k, t_{k+1}), so (a), (c), (c') use the
sample the command produced, k+1. f is the force the table applies to the
tool (along +n when pressing). K, D are the 3x3 world-frame impedance
matrices of the controller that recorded the demo.

  (a)  x_{k+1}                                    measured pose
  (b)  x_{d,k}                                    commanded target
  (c)  x_{k+1} - K^{-1} f_{k+1}                   force-compensated (static)
  (c') x_{k+1} + K^{-1} (D xdot_{k+1} - f_{k+1})  force + damping compensated
  (pf) [x_{k+1}, xdot_{k+1}, f_{k+1}]             pose + velocity + force (9 numbers);
       a controller with its own K', D' rebuilds x_d = x + K'^{-1}(D' xdot - f)
       (stretch goal: stiffness-agnostic label)
"""
from __future__ import annotations

import numpy as np

LABELS = ("a", "b", "c", "cp")          # the four main labels; "pf" is the stretch label
LABEL_NAMES = {"a": "(a) measured x", "b": "(b) commanded x_d", "c": "(c) x - K^-1 f",
               "cp": "(c') x + K^-1(D xdot - f)"}


def make_labels(x, xdot, f, x_d, K, D, which: str) -> np.ndarray:
    """x, xdot, f: (T+1, 3) samples at t_0..t_T; x_d: (T, 3). Returns (T, 3)."""
    x, xdot, f, x_d = (np.asarray(a, float) for a in (x, xdot, f, x_d))
    Kinv = np.linalg.inv(K)
    if which == "a":
        return x[1:].copy()
    if which == "b":
        return x_d.copy()
    if which == "c":
        return x[1:] - f[1:] @ Kinv.T
    if which == "cp":
        return x[1:] + (xdot[1:] @ D.T - f[1:]) @ Kinv.T
    if which == "pf":
        return np.concatenate([x[1:], xdot[1:], f[1:]], axis=1)
    raise ValueError(which)


def rebuild_target(pf, K, D) -> np.ndarray:
    """(..., 9) pose/velocity/force prediction -> (..., 3) target for gains K, D."""
    pf = np.asarray(pf, float)
    x, xdot, f = pf[..., :3], pf[..., 3:6], pf[..., 6:9]
    return x + (xdot @ D.T - f) @ np.linalg.inv(K).T


def corrupt_force(f, n, sigma: float, bias: float, rng) -> np.ndarray:
    """Force-sensor model for the label stretch study: iid Gaussian noise (std sigma, N)
    on each axis plus a constant bias (N) along the table normal n."""
    f = np.asarray(f, float)
    return f + rng.normal(0.0, sigma, f.shape) + bias * np.asarray(n, float)
