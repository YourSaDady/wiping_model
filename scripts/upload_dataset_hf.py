"""Upload the exported CSV dataset (scripts/export_dataset.py) to a Hugging Face dataset repo.

Needs a token with write access: run `hf auth login` once (or set HF_TOKEN).

Usage: python scripts/upload_dataset_hf.py --repo <user>/wipe-labels [--dir data/hf/wipe-labels] [--private]
Then train from it anywhere with `--data <user>/wipe-labels` (diffusion_policy/train.py).
"""
import argparse
import os

from huggingface_hub import HfApi

from wipe.data import DATA_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="dataset repo id, e.g. YourSaDady/wipe-labels")
    ap.add_argument("--dir", default=os.path.join(DATA_ROOT, "hf", "wipe-labels"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="Upload wiping demonstrations (CSV, four action labels)")
    args = ap.parse_args()
    assert os.path.exists(os.path.join(args.dir, "README.md")), f"{args.dir}: run scripts/export_dataset.py first"
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="dataset", commit_message=args.message)
    print(f"done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
