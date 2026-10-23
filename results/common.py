"""Shared helpers for the result scripts: load per-episode JSONL results,
method naming and colours, K_eff, Wilson intervals."""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wipe.stats import wilson  # noqa: E402

RAW = "results/raw"
LABEL_ORDER = ["a", "b", "c", "cp"]
LABEL_TEX = {"a": "(a) x", "b": "(b) x_d", "c": "(c) x - K^-1 f", "cp": "(c') x + K^-1(D xdot - f)"}
LABEL_SHORT = {"a": "(a)", "b": "(b)", "c": "(c)", "cp": "(c')"}
COLORS = {"a": "#d1495b", "b": "#2e86ab", "c": "#edae49", "cp": "#66a182", "scripted": "#222222",
          "hybrid": "#8e6c8a"}


def k_eff(K, k_e):
    return K * k_e / (K + k_e)


def load(split="test", pattern="*"):
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, f"{pattern}_{split}.jsonl"))):
        for line in open(f):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    m = df.method.str.extract(r"^K(?P<K>\d+)_(?P<label>a|b|cp|c)_s(?P<seed>\d+)$")
    df["label"] = m.label
    df["seed"] = pd.to_numeric(m.seed)
    base = df.method.str.extract(r"^(?P<base>scripted|hybrid)_K\d+$").base
    df["family"] = np.where(df.label.notna(), "dp", base)
    df["K_eff"] = k_eff(df.K_n, df.k_e)
    df["ratio_pred_a"] = df.K_n / (df.K_n + df.k_e)
    return df


def success_table(df, by):
    g = df.groupby(by)
    out = g.agg(k=("success", "sum"), n=("success", "size"), uncovered=("uncovered", "mean"),
                fn=("mean_fn_wipe", "mean"), fn_sd=("std_fn_wipe", "mean"), peak=("peak_fn", "mean"),
                peak_max=("peak_fn", "max")).reset_index()
    ci = [wilson(int(k), int(n)) for k, n in zip(out.k, out.n)]
    out["rate"] = out.k / out.n
    out["lo"] = np.minimum([c[0] for c in ci], out.rate)     # guard float round-off at 0 and 1
    out["hi"] = np.maximum([c[1] for c in ci], out.rate)
    return out
