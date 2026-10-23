"""Unit tests: label formulas and time alignment, force signs, the
two-springs-in-series force law, gain matrices, path and coverage."""
import numpy as np
import pytest

from wipe.config import BALL_RADIUS, FRICTION, MAIN_GAINS, TASK, Gains
from wipe.coverage import Coverage
from wipe.labels import make_labels
from wipe.paths import LanePath
from wipe.scene import make_scene
from wipe.sim import WipeEnv


def test_label_formulas_and_alignment():
    rng = np.random.default_rng(0)
    T = 7
    x, xdot, f = rng.normal(size=(T + 1, 3)), rng.normal(size=(T + 1, 3)), rng.normal(size=(T + 1, 3))
    x_d = rng.normal(size=(T, 3))
    n = np.array([0.0, -np.sin(0.3), np.cos(0.3)])
    g = Gains()
    K, D = g.K(n), g.D(n)
    a = make_labels(x, xdot, f, x_d, K, D, "a")
    b = make_labels(x, xdot, f, x_d, K, D, "b")
    c = make_labels(x, xdot, f, x_d, K, D, "c")
    cp = make_labels(x, xdot, f, x_d, K, D, "cp")
    for k in range(T):
        assert np.allclose(a[k], x[k + 1])
        assert np.allclose(b[k], x_d[k])
        assert np.allclose(c[k], x[k + 1] - np.linalg.solve(K, f[k + 1]))
        assert np.allclose(cp[k], x[k + 1] + np.linalg.solve(K, D @ xdot[k + 1] - f[k + 1]))


def test_gain_matrices_table_frame():
    n = np.array([0.0, -np.sin(0.4), np.cos(0.4)])
    g = Gains(K_t=1000, K_n=500, D_t=110, D_n=250)
    assert np.allclose(g.K(n) @ n, 500 * n)
    t = np.array([1.0, 0.0, 0.0])
    assert np.allclose(g.K(n) @ t, 1000 * t)
    assert np.allclose(g.D(n) @ n, 250 * n)
    assert np.allclose(g.K_inv(n), np.linalg.inv(g.K(n)))


def test_lane_path_covers_region():
    sc = make_scene("tilt00_ke2e4", "dev", 0)
    path = LanePath(sc, (0.0, 0.0))
    cov = Coverage(sc.L)
    for s in np.linspace(0, path.length, 4000):
        cov.add(*path.at(s), fn=TASK.f_des)
    assert cov.uncovered == 0.0
    assert np.allclose(path.at(path.length), (0.0, sc.L))      # ends at the far corner of the last lane
    assert path.length == pytest.approx(TASK.n_lanes * sc.L + sc.L, rel=1e-9)


def test_coverage_requires_force():
    cov = Coverage(0.15)
    cov.add(0.07, 0.07, fn=TASK.f_min - 0.01)
    assert cov.uncovered == 1.0
    cov.add(0.07, 0.07, fn=TASK.f_min)
    frac = np.pi * TASK.r_wipe ** 2 / 0.15 ** 2
    assert 1.0 - cov.uncovered == pytest.approx(frac, rel=0.05)


@pytest.mark.parametrize("K_n,k_e", [(500, 5e2), (500, 2e4), (1000, 2e3), (250, 1e5)])
def test_force_sign_and_springs_in_series(K_n, k_e):
    """Hold the target delta below the surface on a flat table: the table
    pushes the tool along +n with f_n = K_eff * delta, K_eff = K k_e / (K + k_e)."""
    from dataclasses import replace
    from wipe.config import TABLES, TableSpec
    import wipe.config as cfg
    name = f"_test_flat_{int(k_e)}"
    cfg.TABLES[name] = TableSpec(name, 0.0, float(k_e))
    try:
        sc = make_scene(name, "dev", 0)
        g = replace(MAIN_GAINS, K_n=float(K_n))
        env = WipeEnv(sc, g)
        delta = 0.01
        x_d = sc.from_plane(0.02, 0.02, BALL_RADIUS - delta)
        # first move above the point, then descend slowly, then hold
        for k in range(40):
            env.step(sc.from_plane(0.02, 0.02, BALL_RADIUS + 0.01))
        for k in range(60):
            a = min(1.0, (k + 1) / 50)
            env.step(sc.from_plane(0.02, 0.02, BALL_RADIUS + 0.01 - a * (0.01 + delta)))
        for k in range(80):
            obs = env.step(x_d)
        f = obs["f"]
        n = sc.arr("n")
        K_eff = K_n * k_e / (K_n + k_e)
        assert f @ n > 0, "table must push the tool along +n"
        assert np.linalg.norm(f - (f @ n) * n) <= FRICTION * (f @ n) + 1e-6   # static friction only
        assert f @ n == pytest.approx(K_eff * delta, rel=0.01, abs=0.02)
    finally:
        del cfg.TABLES[name]


def test_pose_force_label_rebuilds_cp():
    from wipe.labels import rebuild_target
    rng = np.random.default_rng(1)
    T = 5
    x, xdot, f = rng.normal(size=(T + 1, 3)), rng.normal(size=(T + 1, 3)), rng.normal(size=(T + 1, 3))
    x_d = rng.normal(size=(T, 3))
    n = np.array([0.0, -np.sin(0.2), np.cos(0.2)])
    g = Gains()
    pf = make_labels(x, xdot, f, x_d, g.K(n), g.D(n), "pf")
    cp = make_labels(x, xdot, f, x_d, g.K(n), g.D(n), "cp")
    assert pf.shape == (T, 9)
    assert np.allclose(rebuild_target(pf, g.K(n), g.D(n)), cp)


def test_relative_windows_only_shift_position():
    from wipe.dp.dataset import make_windows
    T = 6
    obs = np.zeros((T + 1, 16)); x = np.arange((T + 1) * 3, dtype=float).reshape(T + 1, 3)
    lab = np.ones((T, 9))
    O, A = make_windows(obs, lab, x, 2, 4, relative=True)
    assert np.allclose(A[2, 0, :3], 1 - x[2]) and np.allclose(A[2, 0, 3:], 1.0)
