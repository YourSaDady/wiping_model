"""Run the scripted wiping program once and look at it.

Examples (from the repo root):
    python scripts/run_wipe.py                                # flat_medium, Drake, plot
    python scripts/run_wipe.py --preset flat_hard --meshcat   # watch in the browser
    python scripts/run_wipe.py --preset tilt10_hard --set table__mu=0.3 expert__speed=0.04
    python scripts/run_wipe.py --backend pointmass            # no Drake needed
    python scripts/run_wipe.py --save data/one_episode.npz    # write it in the demo format
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wiping import PRESETS, DemoWriter, make_config, run_episode  # noqa: E402


def parse_sets(items):
    over = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            over[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            over[k] = v
    return over


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="flat_medium", choices=sorted(PRESETS))
    ap.add_argument("--backend", default="drake", choices=["drake", "pointmass"])
    ap.add_argument("--set", nargs="*", metavar="KEY=VAL", help="nested overrides, e.g. table__k_e=3e4")
    ap.add_argument("--config", help="JSON file with a full EpisodeConfig dict (overrides --preset)")
    ap.add_argument("--meshcat", action="store_true", help="visualize in meshcat and keep a replayable recording")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-path", default=None, help="default: figures/<preset>.png")
    ap.add_argument("--save", default=None, help="write the episode (demo .npz format)")
    args = ap.parse_args()

    if args.config:
        from wiping import EpisodeConfig

        cfg = EpisodeConfig.from_dict(json.loads(Path(args.config).read_text()))
        cfg = cfg.replace(sim__backend=args.backend, **parse_sets(args.set))
    else:
        cfg = make_config(args.preset, backend=args.backend, **parse_sets(args.set))

    meshcat = None
    if args.meshcat:
        if args.backend != "drake":
            sys.exit("--meshcat needs the drake backend")
        from pydrake.all import StartMeshcat

        meshcat = StartMeshcat()
        print("meshcat:", meshcat.web_url())
        time.sleep(1.0)

    ep = run_episode(cfg, meshcat=meshcat, record=meshcat is not None, verbose=True)
    print(json.dumps(ep["metrics"], indent=1))

    if args.save:
        w = DemoWriter()
        w.add(ep)
        print("saved", w.save(args.save))
    if not args.no_plot:
        import matplotlib

        if args.plot_path is not None or meshcat is None:
            matplotlib.use("Agg")
        from wiping.plotting import plot_episode

        path = Path(args.plot_path or ROOT / "figures" / f"{cfg.name}_{args.backend}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_episode(ep, path=path)
        print("figure:", path)
    if meshcat is not None:
        input("meshcat recording published (use the Animation controls). Press Enter to quit.")


if __name__ == "__main__":
    main()
