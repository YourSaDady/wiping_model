"""Action labels built from one demonstration.

Label k is the target held over [t_k, t_{k+1}), so (a), (c), (c') use the
sample the command produced, k+1. f is the force the table applies to the
tool (along +n when pressing). K, D are the 3x3 world-frame impedance
matrices of the controller that recorded the demo.

  (a)  x_{k+1}                                    measured pose
  (b)  x_{d,k}                                    commanded target
  (c)  x_{k+1} - K^{-1} f_{k+1}                   force-compensated (static)
  (c') x_{k+1} + K^{-1} (D xdot_{k+1} - f_{k+1})  force + damping compensated
"""
from __future__ import annotations

import numpy as np

LABELS = ("a", "b", "c", "cp")
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
    raise ValueError(which)
