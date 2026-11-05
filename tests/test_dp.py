"""Diffusion Policy data path: observation order, windows, frames, save/load round trip."""
import json

import numpy as np
import torch

from diffusion_policy.dataset import (OBS_DIM, Normalizer, actions_from_frame, actions_to_frame, featurize,
                                      make_windows, table_rotation)
from diffusion_policy.model import ConditionalUnet1D
from diffusion_policy.policy import DiffusionPolicy
from wipe.data import obs_vector
from wipe.scene import make_scene


def _episode(T=30, seed=0):
    sc = make_scene("tilt20_ke2e4", "dev", 0)
    rng = np.random.default_rng(seed)
    x = sc.arr("p0") + rng.normal(0, 0.05, (T, 3))
    obs = obs_vector(x, rng.normal(size=(T, 3)), rng.normal(size=(T, 3)), sc).astype(np.float32)
    lab = (x + rng.normal(0, 0.01, (T, 3))).astype(np.float32)
    return sc, obs, lab


def test_obs_order_matches_env():
    sc, obs, _ = _episode()
    assert obs.shape[1] == OBS_DIM
    np.testing.assert_allclose(obs[0, 9:12], sc.arr("p0"), atol=1e-6)
    np.testing.assert_allclose(obs[0, 12], sc.L, atol=1e-6)
    np.testing.assert_allclose(obs[0, 13:16], sc.arr("n"), atol=1e-6)


def test_windows_alignment():
    _, obs, lab = _episode()
    O, A = make_windows(obs, lab, To=2, H=16)
    T = len(lab)
    assert O.shape == (T, 2, OBS_DIM) and A.shape == (T, 16, 3)
    np.testing.assert_array_equal(O[0, 0], obs[0])            # history clamped at the start
    np.testing.assert_array_equal(O[5, 0], obs[4])
    np.testing.assert_array_equal(O[5, 1], obs[5])
    np.testing.assert_allclose(A[5, 3], lab[8] - obs[5, :3], atol=1e-6)   # relative to x_t
    np.testing.assert_allclose(A[T - 1, 15], lab[T - 1] - obs[T - 1, :3], atol=1e-6)   # last label repeats


def test_table_frame_roundtrip_and_features():
    sc, obs, _ = _episode()
    R = table_rotation(sc.arr("n"))
    np.testing.assert_allclose(R[:, 0], sc.arr("t1"), atol=1e-9)
    np.testing.assert_allclose(R[:, 1], sc.arr("t2"), atol=1e-9)
    F = featurize(obs, "table")
    np.testing.assert_allclose(F[:, 0:3], sc.to_plane(obs[:, 0:3]), atol=1e-5)
    A = np.random.default_rng(1).normal(size=(4, 16, 3)).astype(np.float32)
    n = np.tile(sc.arr("n"), (4, 1))
    np.testing.assert_allclose(actions_from_frame(actions_to_frame(A, n, "table"), n, "table"), A, atol=1e-5)


def test_policy_save_load_roundtrip(tmp_path):
    _, obs, lab = _episode()
    O, A = make_windows(obs, lab)
    on, an = Normalizer.fit(O), Normalizer.fit(A)
    torch.manual_seed(0)
    model = ConditionalUnet1D(3, 2 * OBS_DIM, down_dims=(16, 32))
    torch.save({"ema": model.state_dict(), "step": 1}, tmp_path / "ema_best.pt")
    cfg = dict(To=2, H=16, frame="world", obs_dim=OBS_DIM, down_dims="16,32", diffusion_steps=100,
               ddim_steps=4, K=500, obs_norm=on.state_dict(), act_norm=an.state_dict())
    json.dump(cfg, open(tmp_path / "config.json", "w"))
    pol = DiffusionPolicy(str(tmp_path), "best", device="cpu")
    X = obs[:3, :3]
    T1 = pol.act(O[:3], X, seeds=[1, 2, 3], calls=[0, 0, 0])
    T2 = pol.act(O[1:2], X[1:2], seeds=[2], calls=[0])
    assert T1.shape == (3, 16, 3)
    np.testing.assert_allclose(T1[1], T2[0], atol=1e-4)      # noise fixed per (seed, call), not by batch
    lo, hi = an.lo + X[:, None], an.hi + X[:, None]
    assert np.all(T1 >= lo - 1e-4) and np.all(T1 <= hi + 1e-4)  # clipped samples stay in the label range
