"""Train one state-based Diffusion Policy on one action label.

Usage:
  python scripts/train_dp.py --label b --K 500 --seed 0 --out runs/K500_b_s0 [--steps 60000]
Resumes automatically from <out>/ckpt_last.pt. Every --eval-every steps it
logs the EMA model's validation denoising loss and the DDIM first-action error
on held-out demo episodes, and saves an EMA snapshot <out>/ema_<step>.pt.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from wipe.config import TRAIN_TABLES
from wipe.data import split_paths
from wipe.dp.dataset import Normalizer, build_windows
from wipe.dp.diffusion import DDPM, EMA
from wipe.dp.model import ConditionalUnet1D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["a", "b", "c", "cp", "pf"])
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--To", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--abs", action="store_true", help="absolute targets (default: relative to x_t)")
    ap.add_argument("--down-dims", default="256,512,1024")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--tables", default="")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ddim-steps", type=int, default=16)
    ap.add_argument("--obs-version", type=int, default=1)
    ap.add_argument("--label-f-noise", type=float, default=0.0, help="std [N] of noise on the force inside the label")
    ap.add_argument("--label-f-bias", type=float, default=0.0, help="bias [N] along n on the force inside the label")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tables = [t.name for t in TRAIN_TABLES if not args.tables or t.name in args.tables.split(",")]
    tr_paths, va_paths = split_paths(args.K, tables, args.n_train, args.n_val)
    rel = not args.abs
    nz = dict(f_noise_sigma=args.label_f_noise, f_noise_bias=args.label_f_bias)
    O, A, _ = build_windows(tr_paths, args.label, args.K, args.To, args.H, rel, args.obs_version,
                            noise_seed=1000 + args.seed, **nz)
    Ov, Av, _ = build_windows(va_paths, args.label, args.K, args.To, args.H, rel, args.obs_version,
                              noise_seed=2000 + args.seed, **nz)
    on, an = Normalizer.fit(O), Normalizer.fit(A)
    cfg = dict(vars(args), relative=rel, obs_norm=on.state_dict(), act_norm=an.state_dict(),
               n_train_windows=len(O), n_val_windows=len(Ov), train_paths=tr_paths, val_paths=va_paths)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    Ot = torch.tensor(on.norm(O).reshape(len(O), -1), device=dev)
    At = torch.tensor(an.norm(A), device=dev)
    Ovt = torch.tensor(on.norm(Ov).reshape(len(Ov), -1), device=dev)
    Avt = torch.tensor(an.norm(Av), device=dev)
    dims = tuple(int(d) for d in args.down_dims.split(","))
    adim = At.shape[-1]
    model = ConditionalUnet1D(adim, Ot.shape[1], down_dims=dims).to(dev)
    ema = EMA(model)
    ddpm = DDPM(100, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup)))))
    step = 0
    ck = os.path.join(args.out, "ckpt_last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.model.load_state_dict(s["ema"]); ema.step = s["ema_step"]
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]
        torch.set_rng_state(s["rng_cpu"]); torch.cuda.set_rng_state(s["rng_cuda"])
        print(f"resumed at step {step}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M train windows {len(O)} val {len(Ov)}",
          flush=True)

    # fixed validation subset / noise for comparable numbers across checkpoints
    g = torch.Generator(device=dev).manual_seed(1234)
    vi = torch.randperm(len(Ovt), generator=g, device=dev)[:4096]
    v_noise = torch.randn(Avt[vi].shape, generator=g, device=dev)
    v_t = torch.randint(0, 100, (len(vi),), generator=g, device=dev)
    si = vi[:1024]
    s_noise = torch.randn((len(si), args.H, adim), generator=g, device=dev)

    def evaluate():
        m = ema.model
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(m(ddpm.add_noise(Avt[vi], v_noise, v_t), v_t, Ovt[vi]), v_noise).item()
            pred = ddpm.ddim_sample(m, Ovt[si], (len(si), args.H, adim), args.ddim_steps, noise=s_noise)
        pa = an.unnorm(pred.cpu().numpy())[..., :3]       # position part
        ta = an.unnorm(Avt[si].cpu().numpy())[..., :3]
        err = np.linalg.norm(pa[:, :8] - ta[:, :8], axis=-1)       # executed part of the chunk
        return vl, float(1e3 * err[:, 0].mean()), float(1e3 * err.mean())

    log = open(os.path.join(args.out, "log.csv"), "a")
    if step == 0:
        log.write("step,train_loss,val_loss,val_a0_err_mm,val_a0_7_err_mm,lr,time\n")
    t0, run_loss = time.time(), []
    model.train()
    while step < args.steps:
        idx = torch.randint(0, len(Ot), (args.batch,), device=dev)
        x0, cond = At[idx], Ot[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, 100, (args.batch,), device=dev)
        loss = torch.nn.functional.mse_loss(model(ddpm.add_noise(x0, noise, t), t, cond), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step(); ema.update(model)
        run_loss.append(loss.item()) if step % 50 == 0 else None
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            vl, e0, e07 = evaluate()
            log.write(f"{step},{np.mean(run_loss):.5f},{vl:.5f},{e0:.3f},{e07:.3f},{sched.get_last_lr()[0]:.2e},"
                      f"{time.time()-t0:.0f}\n"); log.flush()
            print(f"step {step} loss {np.mean(run_loss):.4f} val {vl:.4f} a0err {e0:.2f}mm a0-7 {e07:.2f}mm "
                  f"{time.time()-t0:.0f}s", flush=True)
            run_loss = []
            torch.save({"ema": ema.model.state_dict(), "step": step}, os.path.join(args.out, f"ema_{step}.pt"))
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    open(os.path.join(args.out, "DONE"), "w").write(str(step))


if __name__ == "__main__":
    main()
