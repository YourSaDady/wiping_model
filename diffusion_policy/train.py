"""Train one state-based Diffusion Policy on one action label.

Usage (from the repo root):
  python -m diffusion_policy.train --data data/hf/wipe-labels --K 500 --label c --out runs/K500_c_s0
  python -m diffusion_policy.train --data <hf-user>/wipe-labels --label c --out /content/drive/.../K500_c_s0
  smoke test: --steps 300 --eval-every 100 --max-episodes 24 --down-dims 64,128,256

--data is a local dataset folder or a Hugging Face dataset repo id. The run resumes from
<out>/ckpt_last.pt. Every --eval-every steps the EMA model is scored on the validation demos
(held-out episodes of the training tables): denoising loss, and the error of DDIM-sampled
chunks against the label (first action, executed part, and its normal component). Each
evaluation writes a line to <out>/log.csv and an EMA snapshot <out>/ema_<step>.pt; the snapshot
with the lowest executed-chunk error is also copied to <out>/ema_best.pt. --patience N stops
when that error has not improved for N evaluations (0 = train the full --steps).
Imports only numpy / pandas / torch (no pydrake).
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch

from diffusion_policy.dataset import (LABELS, FRAMES, Normalizer, build_windows, load_steps, obs_dim,
                                      resolve_data)
from diffusion_policy.diffusion import DDPM, EMA
from diffusion_policy.model import ConditionalUnet1D


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder or HF dataset repo id")
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--frame", default="world", choices=FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--To", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--n-exec", type=int, default=8, help="executed part of the chunk (for the val error)")
    ap.add_argument("--down-dims", default="256,512,1024")
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--ddim-steps", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0, help="use only the first N train/val episodes (smoke)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + grad scaling (CUDA only; ~2x on a T4)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = resolve_data(args.data, args.K)
    maxe = args.max_episodes or None
    O, A, E = build_windows(load_steps(root, args.K, "train"), args.label, args.To, args.H, args.frame, maxe)
    Ov, Av, _ = build_windows(load_steps(root, args.K, "val"), args.label, args.To, args.H, args.frame, maxe)
    on, an = Normalizer.fit(O), Normalizer.fit(A)
    cfg = dict(vars(args), data_root=root, device=dev, obs_dim=obs_dim(args.frame), relative=True,
               obs_norm=on.state_dict(), act_norm=an.state_dict(), n_train_windows=len(O),
               n_train_episodes=int(E.max() + 1), n_val_windows=len(Ov))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    Ot = torch.tensor(on.norm(O).reshape(len(O), -1), device=dev)
    At = torch.tensor(an.norm(A), device=dev)
    Ovt = torch.tensor(on.norm(Ov).reshape(len(Ov), -1), device=dev)
    Avt = torch.tensor(an.norm(Av), device=dev)
    dims = tuple(int(d) for d in args.down_dims.split(","))
    model = ConditionalUnet1D(3, Ot.shape[1], down_dims=dims).to(dev)
    ema = EMA(model)
    TD = args.diffusion_steps
    ddpm = DDPM(TD, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup)))))
    use_amp = args.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best, since_best = 0, float("inf"), 0
    ck = os.path.join(args.out, "ckpt_last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.model.load_state_dict(s["ema"]); ema.step = s["ema_step"]
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]
        if s.get("scaler"):
            scaler.load_state_dict(s["scaler"])
        best, since_best = s.get("best", best), s.get("since_best", 0)
        torch.set_rng_state(s["rng_cpu"].cpu())
        if dev == "cuda" and s.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        print(f"resumed at step {step}", flush=True)
    print(f"device {dev}  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"train windows {len(O)} ({cfg['n_train_episodes']} episodes)  val windows {len(Ov)}  "
          f"epoch = {len(O) / args.batch:.0f} steps", flush=True)

    # fixed validation subset and noise, so numbers are comparable across checkpoints
    g = torch.Generator().manual_seed(1234)
    vi = torch.randperm(len(Ovt), generator=g)[:4096].to(dev)
    v_noise = torch.randn(tuple(Avt[vi].shape), generator=g).to(dev)
    v_t = torch.randint(0, TD, (len(vi),), generator=g).to(dev)
    si = vi[:1024]
    s_noise = torch.randn((len(si), args.H, 3), generator=g).to(dev)
    n_val = Ov[si.cpu().numpy(), -1, 13:16] if args.frame == "world" else np.tile([0.0, 0.0, 1.0], (len(si), 1))

    def evaluate():
        m = ema.model
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(m(ddpm.add_noise(Avt[vi], v_noise, v_t), v_t, Ovt[vi]), v_noise).item()
            pred = ddpm.ddim_sample(m, Ovt[si], (len(si), args.H, 3), args.ddim_steps, noise=s_noise)
        d = an.unnorm(pred.cpu().numpy()) - an.unnorm(Avt[si].cpu().numpy())
        ex = d[:, :args.n_exec]
        err = np.linalg.norm(ex, axis=-1)
        err_n = np.abs(np.einsum("bhi,bi->bh", ex, n_val))
        return vl, float(1e3 * err[:, 0].mean()), float(1e3 * err.mean()), float(1e3 * err_n.mean())

    log_path = os.path.join(args.out, "log.csv")
    if step == 0 or not os.path.exists(log_path):
        open(log_path, "w").write("step,epoch,train_loss,val_loss,val_a0_err_mm,val_exec_err_mm,"
                                  "val_exec_err_n_mm,lr,time_s\n")
    log = open(log_path, "a")
    t0, run_loss = time.time(), []
    model.train()
    while step < args.steps:
        idx = torch.randint(0, len(Ot), (args.batch,), device=dev)
        x0, cond = At[idx], Ot[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, TD, (args.batch,), device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(ddpm.add_noise(x0, noise, t), t, cond)
        loss = torch.nn.functional.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
        if step % 50 == 0:
            run_loss.append(loss.item())
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            vl, e0, ee, en = evaluate()
            ep = step * args.batch / len(O)
            log.write(f"{step},{ep:.2f},{np.mean(run_loss):.5f},{vl:.5f},{e0:.3f},{ee:.3f},{en:.3f},"
                      f"{sched.get_last_lr()[0]:.2e},{time.time() - t0:.0f}\n"); log.flush()
            print(f"step {step} (epoch {ep:.1f}) loss {np.mean(run_loss):.4f} val {vl:.4f} "
                  f"a0 {e0:.2f}mm exec {ee:.2f}mm exec_n {en:.2f}mm {time.time() - t0:.0f}s", flush=True)
            run_loss = []
            snap = os.path.join(args.out, f"ema_{step}.pt")
            torch.save({"ema": ema.model.state_dict(), "step": step}, snap)
            if ee < best:
                best, since_best = ee, 0
                shutil.copyfile(snap, os.path.join(args.out, "ema_best.pt"))
                json.dump(dict(step=step, val_exec_err_mm=ee), open(os.path.join(args.out, "best.json"), "w"))
            else:
                since_best += 1
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "best": best,
                        "since_best": since_best, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            if args.patience and since_best >= args.patience:
                print(f"early stop: no improvement for {since_best} evaluations", flush=True)
                break
    open(os.path.join(args.out, "DONE"), "w").write(str(step))


if __name__ == "__main__":
    main()
