"""Upload trained Diffusion Policy runs (weights + config) to a Hugging Face model repo.

Only what inference and the record need is uploaded, not the 20 snapshots of a run:
config.json (arguments + normalizers), log.csv, best.json, ema_best.pt, plus the snapshots
named in --steps (e.g. the one picked on the validation scenes). Each run goes to <run name>/
in the repo, so `hf download <repo> --local-dir runs` restores runs/<run name>/.

Needs a write token: `hf auth login` once (or HF_TOKEN). Works from Colab too.
Usage: python scripts/upload_runs_hf.py --repo <user>/wipe-dp --runs runs/K500_c_s0 [runs/K500_b_s0 ...] [--steps 40000]
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="model repo id, e.g. Lo-Fi-gahara/wipe-dp")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders written by diffusion_policy/train.py")
    ap.add_argument("--steps", default="", help="comma list of extra ema_<step>.pt snapshots to include")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    patterns = ["config.json", "log.csv", "best.json", "ema_best.pt", "selected.json", "val_eval.csv"]
    patterns += [f"ema_{s}.pt" for s in args.steps.split(",") if s]
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        assert os.path.exists(os.path.join(run, "config.json")), f"{run}: not a run folder"
        api.upload_folder(folder_path=run, repo_id=args.repo, repo_type="model", path_in_repo=name,
                          allow_patterns=patterns, commit_message=f"Upload {name}")
        print(f"uploaded {name}: https://huggingface.co/{args.repo}/tree/main/{name}")


if __name__ == "__main__":
    main()
