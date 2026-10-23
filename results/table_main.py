"""Main results table: success (Wilson 95 % CI), coverage, mean and peak normal force,
per method, in-distribution (training tables, new scenes) vs OOD tables, on the test split.
Also per-table success for every method, and label-vs-label tests:
Fisher's exact (pooled counts) and McNemar's exact (paired by scene, seed 0), with Holm
correction over the 6 label pairs.

Usage: python results/table_main.py [--K 500]
Outputs: results/tables/main_{K}.csv, results/tables/per_table_{K}.csv, results/tables/label_tests_{K}.csv,
         results/tables/main_{K}.md
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common import LABEL_ORDER, LABEL_SHORT, load, success_table
from wipe.stats import fisher, holm, mcnemar_exact


def fmt_rate(r):
    return f"{100*r.rate:.0f}% [{100*r.lo:.0f}, {100*r.hi:.0f}] ({int(r.k)}/{int(r.n)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    df = load(args.split)
    df = df[df.K_n == args.K]
    df["method_group"] = np.where(df.family == "dp", "DP " + df.label.map(LABEL_SHORT).fillna(""), df.family)
    df["dist"] = np.where(df.ood, "OOD", "ID")
    main = success_table(df, ["method_group", "dist"])
    main["coverage"] = 1 - main.uncovered
    main.to_csv(f"results/tables/main_{args.K}.csv", index=False)
    per = success_table(df, ["method_group", "table"])
    per.to_csv(f"results/tables/per_table_{args.K}.csv", index=False)
    # per-seed success for the DP rows
    seeds = df[df.family == "dp"].groupby(["label", "seed", "dist"]).success.mean().unstack("dist")

    tests = []
    dp = df[df.family == "dp"]
    for dist in ("ID", "OOD"):
        d = dp[dp.dist == dist]
        rows = []
        for la, lb in itertools.combinations(LABEL_ORDER, 2):
            A, B = d[d.label == la], d[d.label == lb]
            if A.empty or B.empty:
                continue
            p_f = fisher(int(A.success.sum()), len(A), int(B.success.sum()), len(B))
            a0 = A[A.seed == 0].set_index(["table", "index"]).success
            b0 = B[B.seed == 0].set_index(["table", "index"]).success
            common = a0.index.intersection(b0.index)
            p_m, n10, n01 = mcnemar_exact(a0[common].values, b0[common].values)
            rows.append(dict(dist=dist, A=la, B=lb, succ_A=A.success.mean(), succ_B=B.success.mean(),
                             n_A=len(A), n_B=len(B), fisher_p=p_f, mcnemar_p_seed0=p_m,
                             only_A=n10, only_B=n01))
        if rows:
            r = pd.DataFrame(rows)
            r["fisher_p_holm"] = holm(r.fisher_p)
            r["mcnemar_p_holm"] = holm(r.mcnemar_p_seed0)
            tests.append(r)
    tests = pd.concat(tests) if tests else pd.DataFrame()
    tests.to_csv(f"results/tables/label_tests_{args.K}.csv", index=False)

    order = ["scripted", "hybrid"] + ["DP " + LABEL_SHORT[l] for l in LABEL_ORDER]
    lines = [f"# Main results, K_n = {args.K}, split = {args.split}", "",
             "| method | ID success | ID coverage | ID mean f_n | ID peak (mean/max) | OOD success | OOD coverage | OOD mean f_n | OOD peak (mean/max) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        cells = [m]
        for dist in ("ID", "OOD"):
            r = main[(main.method_group == m) & (main.dist == dist)]
            if r.empty:
                cells += ["–"] * 4
                continue
            r = r.iloc[0]
            cells += [fmt_rate(r), f"{100*r.coverage:.1f}%", f"{r.fn:.2f} ± {r.fn_sd:.2f} N",
                      f"{r.peak:.1f} / {r.peak_max:.1f} N"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "DP rows pool 3 seeds × 50 scenes × tables. Wilson 95 % intervals. Mean f_n ± mean within-episode std "
              "over wiping samples.", "", "## Per-seed DP success", "", seeds.round(3).to_markdown(), "",
              "## Label tests", "", tests.round(4).to_markdown(index=False) if len(tests) else "(none)"]
    open(f"results/tables/main_{args.K}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
