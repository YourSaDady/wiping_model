"""Meshcat overlays: the dirt region, the planned lanes, and the wiped trail."""

from __future__ import annotations

import numpy as np


def _rgba(r, g, b, a=1.0):
    from pydrake.all import Rgba

    return Rgba(r, g, b, a)


def draw_scene_overlays(meshcat, cfg, waypoints_uv):
    from pydrake.all import Box, RigidTransform, RotationMatrix

    tab, reg = cfg.table, cfg.region
    n = tab.normal()
    # dirt patch (thin box 1 mm above the surface)
    center = tab.to_world([reg.center_uv[0], reg.center_uv[1], 0.0]) + 0.0006 * n
    meshcat.SetObject("/wiping/dirt", Box(reg.L, reg.L, 0.001), _rgba(0.45, 0.30, 0.15, 0.85))
    meshcat.SetTransform("/wiping/dirt", RigidTransform(RotationMatrix(tab.frame()), center))
    # planned lanes, 3 mm above the surface
    w = np.asarray(waypoints_uv)
    pts = tab.to_world(np.column_stack([w, np.full(len(w), 0.003)]))
    meshcat.SetLine("/wiping/plan", pts.T.copy(), 2.0, _rgba(0.1, 0.4, 0.9, 1.0))
    meshcat.Delete("/wiping/trail")


def draw_trail(meshcat, cfg, x, f):
    """Contact points with f_n >= f_min, drawn as a white line on the surface."""
    tab = cfg.table
    n = tab.normal()
    good = (np.asarray(f) @ n) >= cfg.f_min()
    if good.sum() < 2:
        return
    uvw = tab.to_table(np.asarray(x)[good])
    uvw[:, 2] = 0.002
    meshcat.SetLine("/wiping/trail", tab.to_world(uvw).T.copy(), 4.0, _rgba(1.0, 1.0, 1.0, 1.0))
