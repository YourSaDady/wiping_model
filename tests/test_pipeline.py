"""Fast checks of everything except the Drake physics (point-mass backend).

    python tests/test_pipeline.py          # or: pytest tests/
    python tests/test_pipeline.py --drake  # also one short Drake episode
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wiping import (DemoWriter, EpisodeConfig, PRESETS, load_demos, make_config, run_episode,  # noqa: E402
                    sample_config)
from wiping.config import TableConfig  # noqa: E402
from wiping.coverage import CoverageGrid  # noqa: E402
from wiping.dataset import OBS_DIM, episode_slices  # noqa: E402
from wiping.expert import WIPE  # noqa: E402
from wiping.lanes import plan_lanes  # noqa: E402


def test_table_frame():
    for tilt in [0.0, 0.2, 0.4]:
        for d in [0.0, 1.0, -2.5]:
            tab = TableConfig(tilt=tilt, tilt_dir=d)
            R = tab.frame()
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0)
            assert np.allclose(R[:, 2], tab.normal())
            p = np.array([0.01, -0.02, 0.03])
            assert np.allclose(tab.to_table(tab.to_world(p)), p)


def test_lanes_cover_region():
    cfg = make_config("flat_medium", backend="pointmass")
    wp = plan_lanes(cfg.region, cfg.expert)
    g = CoverageGrid(cfg)
    # walk the plan densely with full force -> everything must be clean
    f = 2 * cfg.expert.f_des * cfg.table.normal()
    for a, b in zip(wp[:-1], wp[1:]):
        for s in np.linspace(0, 1, 30):
            uv = a + s * (b - a)
            g.update(cfg.table.to_world([uv[0], uv[1], 0.0]), f)
    assert g.uncovered() == 0.0, g.uncovered()


def test_config_roundtrip():
    cfg = sample_config(np.random.default_rng(3), "train", backend="pointmass", seed=3)
    cfg2 = EpisodeConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    cfg3 = cfg.replace(table__tilt=0.1, expert__speed=0.05)
    assert cfg3.table.tilt == 0.1 and cfg3.expert.speed == 0.05


def test_presets_succeed_pointmass():
    for p in PRESETS:
        ep = run_episode(make_config(p, backend="pointmass"))
        m = ep["metrics"]
        assert m["success"], (p, m)
        assert abs(m["fn_wipe_mean"] - 10.0) < 0.5


def test_labels_and_writer():
    w = DemoWriter()
    eps = []
    for p in ["flat_hard", "tilt10_medium"]:
        ep = run_episode(make_config(p, backend="pointmass"))
        eps.append(ep)
        w.add(ep)
    with tempfile.TemporaryDirectory() as d:
        path = w.save(Path(d) / "demo.npz")
        z = load_demos(path)
    N = sum(len(e["x_d"]) for e in eps)
    assert z["obs"].shape == (N, OBS_DIM)
    assert z["label_d_pose_force"].shape == (N, 6)
    assert len(z["meta"]) == 2 and z["K"].shape == (2, 3, 3)
    for i, sl in enumerate(episode_slices(z)):
        ep = eps[i]
        assert np.allclose(z["label_a_measured"][sl], ep["x"][1:])
        assert np.allclose(z["label_b_commanded"][sl], ep["x_d"])
        # quasi-static steady wiping: (c) tracks (b) along n to within a mm
        wipe = z["phase"][sl] == WIPE
        n = z["n"][i]
        gap = (z["label_c_force_comp"][sl][wipe] - z["label_b_commanded"][sl][wipe]) @ n
        assert np.median(np.abs(gap)) < 1e-3, np.median(np.abs(gap))
        # and (a) sits ~ f_des / K_n above (b) along n: the spring stretch label (a) throws away
        gap_a = np.median((z["label_a_measured"][sl][wipe] - z["label_b_commanded"][sl][wipe]) @ n)
        K_n = n @ z["K"][i] @ n
        assert abs(gap_a - 10.0 / K_n) < 2e-3, gap_a


def test_drake_smoke():
    import importlib.util

    if importlib.util.find_spec("pydrake") is None:
        print("skip test_drake_smoke: pydrake not installed")
        return
    # week-1 milestone: the scripted program wipes each flat table, with margin on both force bounds
    for p in ["flat_soft", "flat_medium", "flat_hard"]:
        cfg = make_config(p, backend="drake")
        ep = run_episode(cfg, verbose=True)
        m = ep["metrics"]
        assert m["success"], (p, m)
        assert m["u_T"] <= 0.01, (p, m)
        assert m["fn_wipe_min"] >= 1.2 * cfg.f_min(), (p, m)
        assert m["fn_max"] <= 0.8 * cfg.f_max(), (p, m)


if __name__ == "__main__":
    tests = [test_table_frame, test_lanes_cover_region, test_config_roundtrip, test_presets_succeed_pointmass,
             test_labels_and_writer]
    if "--drake" in sys.argv:
        tests.append(test_drake_smoke)
    for t in tests:
        t()
        print("ok ", t.__name__)
