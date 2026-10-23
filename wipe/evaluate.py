"""Closed-loop evaluation of chunked policies with lockstep worker processes.

The main process holds the policy (GPU) and batches inference over all active
environments; each worker process owns one WipeEnv at a time. Every round, each
active worker executes n_exec targets of its chunk (receding horizon) and
returns its latest To observations. Episodes run for t_max seconds (no early
termination), then the worker reports metrics and takes the next scene.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback

import numpy as np

from .config import DT_POLICY, GAINS_BY_KN, T_MAX


def _worker(conn, To, obs_version):
    from .data import obs_vector
    from .scene import Scene
    from .sim import WipeEnv
    env = obs_hist = None
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "close":
                break
            if cmd == "reset":
                _, scene_d, K_n, record = msg
                scene = Scene.from_dict(scene_d)
                env = WipeEnv(scene, GAINS_BY_KN[K_n], record_q=record)
                o = env.observe()
                obs_hist = [obs_vector(o["x"], o["xdot"], o["f"], scene, o["u"], obs_version)]
                trace = {"x": [o["x"]], "f": [o["f"]], "xd": []}
                conn.send(("obs", _hist(obs_hist, To), o["x"]))
            elif cmd == "step":
                _, targets, t_max = msg
                done = False
                for xd in targets:
                    o = env.step(xd)
                    obs_hist.append(obs_vector(o["x"], o["xdot"], o["f"], env.scene, o["u"], obs_version))
                    trace["x"].append(o["x"]); trace["f"].append(o["f"]); trace["xd"].append(np.asarray(xd))
                    if env.k * DT_POLICY >= t_max - 1e-9:
                        done = True
                        break
                if done:
                    out = {"metrics": env.metrics(),
                           "trace": {k: np.asarray(v, np.float32) for k, v in trace.items()},
                           "samples": np.asarray(env.samples, np.float32)}
                    if env.record_q:
                        out["q_hi"] = np.asarray(env.q_log, np.float32)
                    conn.send(("done", out))
                else:
                    conn.send(("obs", _hist(obs_hist, To), o["x"]))
    except Exception:
        conn.send(("error", traceback.format_exc()))


def _hist(obs_hist, To):
    idx = [max(0, len(obs_hist) - To + i) for i in range(To)]
    return np.stack([obs_hist[i] for i in idx])


def evaluate_policy(policy, jobs, n_workers=24, n_exec=8, t_max=T_MAX, on_result=None, log_every=60):
    """jobs: list of dicts with keys scene (Scene), K_n, seed, record (bool), and any
    metadata. on_result(job, out) is called as each episode finishes."""
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for _ in range(min(n_workers, len(jobs))):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, policy.To, getattr(policy, "obs_version", 1)), daemon=True)
        p.start()
        pipes.append(a)
        procs.append(p)
    queue = list(jobs)
    active = {}          # worker id -> dict(job, O, x, calls)
    results = []
    t0, last = time.time(), time.time()

    def start(w):
        if not queue:
            return
        job = queue.pop(0)
        pipes[w].send(("reset", job["scene"].to_dict(), job["K_n"], job.get("record", False)))
        kind, O, x = _recv(pipes[w])
        active[w] = dict(job=job, O=O, x=x, calls=0)

    for w in range(len(pipes)):
        start(w)
    while active:
        ws = sorted(active)
        O = np.stack([active[w]["O"] for w in ws])
        X = np.stack([active[w]["x"] for w in ws])
        seeds = [active[w]["job"]["seed"] for w in ws]
        calls = [active[w]["calls"] for w in ws]
        A = policy.act(O, X, seeds, calls)
        for i, w in enumerate(ws):
            pipes[w].send(("step", A[i, :n_exec], t_max))
            active[w]["calls"] += 1
        for w in ws:
            msg = _recv(pipes[w])
            if msg[0] == "obs":
                active[w]["O"], active[w]["x"] = msg[1], msg[2]
            else:
                job = active.pop(w)["job"]
                out = msg[1]
                results.append((job, out))
                if on_result:
                    on_result(job, out)
                start(w)
        if time.time() - last > log_every:
            last = time.time()
            print(f"  eval: {len(results)}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)
    for p in pipes:
        p.send(("close",))
    for p in procs:
        p.join(timeout=10)
    return results


def _recv(pipe):
    msg = pipe.recv()
    if msg[0] == "error":
        raise RuntimeError("worker failed:\n" + msg[1])
    return msg
