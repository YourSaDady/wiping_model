"""Time-limit sensitivity (secondary analysis, pre-registered 2026-10-23).

Seed-0 K_n = 500 policies on the 600 training-table test scenes with T_MAX = 50 s (primary) and
100 s. Failure breakdown: coverage only (u > ε_u, peak ok), peak only (peak > f_max, coverage ok),
both.

Usage: python results/table_time_limit.py
Output: results/tables/time_limit.csv, results/tables/time_limit.md
"""
import json

import pandas as pd

from common import LABEL_ORDER, LABEL_SHORT
from wipe.config import TASK
from wipe.stats import mcnemar_exact, wilson

rows = []
for lab in LABEL_ORDER:
    d = {}
    for T, name in ((50, f"K500_{lab}_s0"), (100, f"K500_{lab}_s0_T100")):
        x = pd.DataFrame([json.loads(l) for l in open(f"results/raw/{name}_test.jsonl")])
        x = x[~x.ood].set_index(["table", "index"]).sort_index()
        d[T] = x
        cov = x.uncovered > TASK.eps_u
        pk = x.peak_fn > TASK.f_max
        k, n = int(x.success.sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append(dict(label=lab, T_max=T, k=k, n=n, rate=k / n, lo=lo, hi=hi,
                         fail_coverage_only=int((cov & ~pk).sum()), fail_peak_only=int((pk & ~cov).sum()),
                         fail_both=int((cov & pk).sum()), mean_uncovered=x.uncovered.mean(),
                         median_first_contact=x.first_contact_t.median()))
    common = d[50].index.intersection(d[100].index)
    p, n10, n01 = mcnemar_exact(d[100].success[common].values, d[50].success[common].values)
    rows[-1].update(mcnemar_p_vs_50=p, only_100=n10, only_50=n01)
t = pd.DataFrame(rows)
t.to_csv("results/tables/time_limit.csv", index=False)
lines = ["# Time-limit sensitivity (seed 0, K_n = 500, 600 training-table test scenes)", "",
         "| label | T_max | success [95 % CI] | fails: coverage only / peak only / both | McNemar p (100 vs 50 s) |",
         "|---|---|---|---|---|"]
for _, r in t.iterrows():
    p = r.get("mcnemar_p_vs_50")
    lines.append(f"| {LABEL_SHORT[r.label]} | {r.T_max} s | {100*r.rate:.1f} % [{100*r.lo:.1f}, {100*r.hi:.1f}] "
                 f"({r.k}/{r.n}) | {r.fail_coverage_only} / {r.fail_peak_only} / {r.fail_both} | "
                 f"{'' if pd.isna(p) else f'{p:.1e}'} |")
open("results/tables/time_limit.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
