"""Is one model really better than another? Paired comparisons on the test split.

    python post_analysis/m2_compare_models.py \\
        --group full "smri_pw0.5_s*" "smri_pw0.5_2026*" \\
        --group fixed "smri_pw0.5_fixed*" --group off "smri_pw0.5_off*" \\
        --baselines hippocampus pca_smri

--group NAME GLOB [GLOB ...]   one model = its seed runs (globs over checkpoints/, quoted);
                               the first group is the reference
--baselines NAME [NAME ...]    baselines from results/baselines/baseline_index.csv
                               (run i2_baselines.py first)
--out NAME                     output name (default: the reference group's name)

Compared pairs: every pair of groups, and the reference group against each baseline.

Metrics (test split, both index versions), written so that higher = better:
    auc            AD vs CN, all test visits
    mmse_ad        -Spearman(index, MMSE) within AD (sign flipped: lower MMSE = worse)
    ptau_alzpath   Spearman(index, pTau-217 AlzPath) within AD
    ptau_janssen   Spearman(index, pTau-217 Janssen) within AD
A model's value = its metric averaged over its seeds (as reported by m1_collect_seeds).

Uncertainty: two-level bootstrap, N_BOOTSTRAP rounds. Each round
    1. redraws the subjects of that metric's group (the visits that have the measurement),
       same number of subjects, with replacement, all visits of a subject together;
       the same subjects are used for both models of a pair (paired);
    2. redraws each model's seeds (k from its k, with replacement); baselines have one;
    3. computes both models' seed-averaged metric and their difference.
95% CI = 2.5th-97.5th percentile of the differences; p = two-sided bootstrap p.
Holm correction over the compared pairs, separately for each metric and index version
(each metric is its own question; within it, several pairs are tested).

Output: results/compare/<out>.csv (one row per index version x metric x pair).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from i1_index_core import AGECORR_INDEX, PRIMARY_SPLIT, RAW_INDEX
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
BASELINE_INDEX = PROJECT_ROOT / "results" / "baselines" / "baseline_index.csv"
OUT_DIR = PROJECT_ROOT / "results" / "compare"

N_BOOTSTRAP = 2000
SEED = 42
INDEXES = [AGECORR_INDEX, RAW_INDEX]

# metric -> (target column or None for AUC, sign so that higher = better)
METRICS = {
    "auc": (None, 1.0),
    "mmse_ad": ("mmse", -1.0),
    "ptau_alzpath": ("ptau217_alzpath", 1.0),
    "ptau_janssen": ("ptau217_janssen", 1.0),
}
META = ["visit_id", "adni_id", "diagnosis", "mmse", "ptau217_alzpath", "ptau217_janssen"]


# ---------- loading ----------

def load_group(name, patterns):
    runs = sorted({r for p in patterns for r in CHECKPOINTS.glob(p)})
    tables = []
    for run in runs:
        csv = run / "s2_index" / "disease_index.csv"
        if not csv.is_file():
            print(f"skip {run.name}: no s2_index/disease_index.csv")
            continue
        df = pd.read_csv(csv, dtype={"visit_id": str, "adni_id": str})
        tables.append(df[df["split"] == PRIMARY_SPLIT].set_index("visit_id"))
    if not tables:
        sys.exit(f"group {name}: no finished runs match {patterns}")
    print(f"{name}: {len(tables)} runs")
    return tables


def load_baselines(names):
    if not BASELINE_INDEX.is_file():
        sys.exit(f"no {BASELINE_INDEX}: run post_analysis/i2_baselines.py first")
    df = pd.read_csv(BASELINE_INDEX, dtype={"visit_id": str, "adni_id": str})
    out = {}
    for name in names:
        part = df[(df["baseline"] == name) & (df["split"] == PRIMARY_SPLIT)]
        if part.empty:
            sys.exit(f"baseline {name} not in {BASELINE_INDEX}")
        out[name] = [part.set_index("visit_id")]
    return out


def common_visits(models):
    """Test visits present in every run of every model (they should all be the same)."""
    sets = [set(t.index) for tables in models.values() for t in tables]
    common = sorted(set.intersection(*sets))
    if any(len(s) != len(common) for s in sets):
        print(f"WARNING: runs differ in test visits; using the {len(common)} visits all share")
    return common


# ---------- statistics ----------

def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(scores, is_ad):
    n_ad, n_cn = is_ad.sum(), (~is_ad).sum()
    if n_ad == 0 or n_cn == 0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[is_ad].sum() - n_ad * (n_ad + 1) / 2) / (n_ad * n_cn))


def metric_value(metric, scores, meta_rows):
    target, sign = METRICS[metric]
    if target is None:
        return auc(scores, meta_rows["is_ad"])
    return sign * spearman(scores, meta_rows[target])


def metric_rows(meta, metric):
    """Row positions (into the common visit order) that enter this metric."""
    target, _ = METRICS[metric]
    if target is None:
        return np.arange(len(meta))
    keep = (meta["diagnosis"] == "ad") & meta[target].notna()
    return np.flatnonzero(keep.to_numpy())


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


# ---------- main ----------

def compare(models, pairs, meta, index_name, rng):
    scores = {m: np.vstack([t.loc[meta.index, index_name].to_numpy(dtype=float) for t in tables])
              for m, tables in models.items()}                     # model -> (n_seeds, n_visits)
    columns = {"is_ad": (meta["diagnosis"] == "ad").to_numpy()}
    for target in ("mmse", "ptau217_alzpath", "ptau217_janssen"):
        columns[target] = meta[target].to_numpy(dtype=float)

    rows = []
    for metric in METRICS:
        positions = metric_rows(meta, metric)
        subjects = meta["adni_id"].to_numpy()[positions]
        groups = pd.Series(np.arange(len(positions))).groupby(subjects).indices
        ids = np.array(list(groups))
        sub = {k: v[positions] for k, v in columns.items()}

        def value(model, rows_, seed_draw):
            per_seed = [metric_value(metric, scores[model][s, positions][rows_],
                                     {k: v[rows_] for k, v in sub.items()})
                        for s in seed_draw]
            return float(np.nanmean(per_seed))

        all_rows = np.arange(len(positions))
        observed = {m: value(m, all_rows, range(len(scores[m]))) for m in models}

        diffs = {pair: [] for pair in pairs}
        for _ in range(N_BOOTSTRAP):
            rows_ = np.concatenate([groups[s] for s in rng.choice(ids, len(ids))])
            draw = {m: rng.integers(0, len(scores[m]), len(scores[m])) for m in models}
            cache = {m: value(m, rows_, draw[m]) for m in models}
            for a, b in pairs:
                diffs[(a, b)].append(cache[a] - cache[b])

        for a, b in pairs:
            d = np.asarray(diffs[(a, b)])
            d = d[np.isfinite(d)]
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            rows.append({
                "index_name": index_name, "metric": metric, "model_a": a, "model_b": b,
                "n_visits": len(positions), "n_subjects": len(ids),
                "seeds_a": len(scores[a]), "seeds_b": len(scores[b]),
                "value_a": observed[a], "value_b": observed[b],
                "difference": observed[a] - observed[b],
                "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
                "p": max(p, 1.0 / len(d)),          # bootstrap p cannot go below 1/rounds
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("metric")["p"].transform(holm)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", action="append", required=True,
                        metavar=("NAME", "GLOB"))
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    for g in args.group:
        if len(g) < 2:
            parser.error(f"--group {g[0]} needs at least one glob")
    return args


if __name__ == "__main__":
    args = parse_args()
    groups = {g[0]: load_group(g[0], g[1:]) for g in args.group}
    baselines = load_baselines(args.baselines)
    models = {**groups, **baselines}

    names = list(groups)
    reference = names[0]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs += [(reference, b) for b in baselines]

    visits = common_visits(models)
    first = next(iter(models.values()))[0]
    meta = first.loc[visits, [c for c in META if c in first.columns and c != "visit_id"]]

    rng = np.random.default_rng(SEED)
    result = pd.concat([compare(models, pairs, meta, index_name, rng) for index_name in INDEXES],
                       ignore_index=True)

    pd.set_option("display.width", 220)
    show = result.copy()
    show["CI"] = show.apply(lambda r: f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1)
    for index_name in INDEXES:
        part = show[show["index_name"] == index_name]
        print(f"\n{index_name} (test; higher = better; mmse_ad is -r)")
        print(part[["metric", "model_a", "model_b", "n_subjects", "value_a", "value_b",
                    "difference", "CI", "p", "p_holm"]]
              .round({"value_a": 3, "value_b": 3, "difference": 3, "p": 4, "p_holm": 4})
              .to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.out or reference}.csv"
    result.to_csv(path, index=False)
    print(f"\nSaved: {path}")