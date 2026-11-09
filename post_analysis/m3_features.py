"""Which features does each model's prediction loss use, how stable are they, and do they
relate to pTau? Reads what training already logged; no model is run.

    python post_analysis/m3_features.py \\
        --group full "smri_pw0.5_s*" "smri_pw0.5_2026*" --group fixed "smri_pw0.5_fixed*"

--group NAME GLOB [GLOB ...]   one model = its seed runs (globs over checkpoints/, quoted);
                               the first group is the reference
--out NAME                     output name (default: the reference group's name)

Per run:
    best epoch E   read from checkpoints/<run>/best.ckpt (the checkpoint every result uses)
    active top-k   the list the model was trained toward when it was saved: the last entry of
                   logs/<run>/topk_history.csv selected before epoch E (a list chosen at the end
                   of epoch e is used to train epoch e+1). Before the warm-up ends, all features
                   are predicted; with cimlr_mode off the logged list is never used in training.
    epoch share    per feature, the share of the epochs up to E in which it was in the top-k

Per group: in how many seeds each feature is active, its mean epoch share, and how similar the
seeds' active lists are (mean pairwise Jaccard). Between groups: the same counts side by side.
Link to pTau: for every feature active in any run, its own Spearman (FreeSurfer / DTI value)
with pTau-217 (both assays) and MMSE within AD, test split, pTau outliers excluded as in s2.

Outputs in results/features/:
    <out>_runs.csv       one row per run: group, seed, best epoch, which list was active
    <out>_features.csv   one row per group x feature: seeds active, epoch share
    <out>_links.csv      one row per feature: counts per group + correlations within AD
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from i1_index_core import PRIMARY_SPLIT, TARGETS, drop_excluded, safe_spearman

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
LOGS = PROJECT_ROOT / "logs"
OUT_DIR = PROJECT_ROOT / "results" / "features"


# ---------- per run ----------

def best_epoch(run):
    ckpt = torch.load(CHECKPOINTS / run / "best.ckpt", map_location="cpu", weights_only=False)
    return int(ckpt["epoch"])


def read_run(run):
    config = yaml.safe_load((LOGS / run / "config.yml").read_text())
    history = pd.read_csv(LOGS / run / "topk_history.csv")
    epoch = best_epoch(run)
    before = history[history["epoch"] < epoch].sort_values("epoch")

    if before.empty:                     # saved before the first top-k: all features predicted
        active, chosen_at = [], None
    else:
        last = before.iloc[-1]
        active, chosen_at = str(last["feature_names"]).split("|"), int(last["epoch"])

    counts = {}
    for names in before["feature_names"].astype(str):
        for name in names.split("|"):
            counts[name] = counts.get(name, 0) + 1
    share = {name: c / len(before) for name, c in counts.items()} if len(before) else {}

    mode = config["training"].get("cimlr_mode", "full")
    return {
        "run": run, "seed": config["training"]["seed"], "modality": config["data"]["modality"],
        "cimlr_mode": mode, "prepared_dir": config["data"]["prepared_dir"],
        "best_epoch": epoch, "active_chosen_at_epoch": chosen_at,
        "n_active": len(active) if active else "all",
        "used_in_training": mode != "off",
        "active": active, "share": share,
    }


def load_group(name, patterns):
    runs = sorted({r.name for p in patterns for r in CHECKPOINTS.glob(p)})
    out = []
    for run in runs:
        missing = [str(p) for p in (CHECKPOINTS / run / "best.ckpt", LOGS / run / "topk_history.csv")
                   if not p.is_file()]
        if missing:
            print(f"skip {run}: missing {missing}")
            continue
        out.append(read_run(run))
    if not out:
        sys.exit(f"group {name}: no usable runs match {patterns}")
    return out


# ---------- summaries ----------

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def feature_table(name, runs):
    names = sorted({f for r in runs for f in r["active"]} | {f for r in runs for f in r["share"]})
    rows = []
    for f in names:
        rows.append({"group": name, "feature": f,
                     "seeds_active": sum(f in r["active"] for r in runs), "n_seeds": len(runs),
                     "epoch_share": float(np.mean([r["share"].get(f, 0.0) for r in runs]))})
    return pd.DataFrame(rows).sort_values(["seeds_active", "epoch_share"], ascending=False)


def stability(runs):
    lists = [r["active"] for r in runs if r["active"]]
    if len(lists) < 2:
        return np.nan
    return float(np.mean([jaccard(a, b) for a, b in itertools.combinations(lists, 2)]))


def correlations(features, prepared_dir):
    """Spearman of each feature with the targets within AD on the test split (outliers out)."""
    prepared = Path(prepared_dir)
    cohort = pd.read_csv(prepared / "cohort.csv", dtype={"visit_id": str, "adni_id": str})
    split = pd.read_csv(prepared / "split.csv", dtype={"visit_id": str})
    tables = []
    for m in ("smri", "dti"):
        path = prepared / f"features_{m}.csv"
        if path.is_file():
            df = pd.read_csv(path, dtype={"visit_id": str}).set_index("visit_id")
            tables.append(df.drop(columns=[c for c in df.columns if c.endswith("_image_id")]))
    values = pd.concat(tables, axis=1, join="outer")

    keep = ["visit_id", "adni_id", "diagnosis"] + [t for t in TARGETS if t in cohort.columns]
    data = split[["visit_id", "split"]].merge(cohort[keep], on="visit_id", how="left")
    data = drop_excluded(data)
    data = data[(data["split"] == PRIMARY_SPLIT) & (data["diagnosis"] == "ad")].set_index("visit_id")

    rows = []
    for f in features:
        row = {"feature": f}
        if f not in values.columns:
            rows.append(row)
            continue
        x = values[f].reindex(data.index)
        for target in TARGETS:
            if target not in data.columns:
                continue
            ok = x.notna() & data[target].notna()
            r, p = safe_spearman(x[ok].to_numpy(), data.loc[ok, target].to_numpy())
            row.update({f"n_{target}": int(ok.sum()), f"r_{target}": r, f"p_{target}": p})
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", action="append", required=True,
                        metavar=("NAME", "GLOB"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    for g in args.group:
        if len(g) < 2:
            parser.error(f"--group {g[0]} needs at least one glob")
    return args


if __name__ == "__main__":
    args = parse_args()
    groups = {g[0]: load_group(g[0], g[1:]) for g in args.group}
    reference = next(iter(groups))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)

    runs_table = pd.DataFrame([{"group": g, **{k: v for k, v in r.items() if k not in ("active", "share")}}
                               for g, runs in groups.items() for r in runs])
    print("\nruns\n" + runs_table.drop(columns=["prepared_dir"]).to_string(index=False))
    for g, runs in groups.items():
        if not all(r["used_in_training"] for r in runs):
            print(f"note: {g} has cimlr_mode off; its top-k lists were logged but not used in training")

    features = pd.concat([feature_table(g, runs) for g, runs in groups.items()], ignore_index=True)
    for g, runs in groups.items():
        part = features[features["group"] == g]
        print(f"\n{g}: active top-k at the best epoch, {len(runs)} seeds | "
              f"mean pairwise Jaccard {stability(runs):.2f} | "
              f"{int((part['seeds_active'] > 0).sum())} distinct features active in any seed")
        print(part[part["seeds_active"] > 0][["feature", "seeds_active", "epoch_share"]]
              .round(2).head(40).to_string(index=False))

    wide = features.pivot_table(index="feature", columns="group", values="seeds_active",
                                fill_value=0).astype(int)
    wide = wide[[g for g in groups if g in wide.columns]]
    wide = wide[(wide > 0).any(axis=1)]
    links = correlations(list(wide.index), groups[reference][0]["prepared_dir"])
    links = wide.add_prefix("seeds_").reset_index().merge(links, on="feature", how="left")
    order = [f"seeds_{g}" for g in groups]
    if len(groups) > 1:
        links["seeds_diff"] = links[f"seeds_{reference}"] - links[order[1:]].max(axis=1)
        order = ["seeds_diff"] + order
    links = links.sort_values(order, ascending=False)

    show = ["feature"] + [c for c in links.columns if c.startswith("seeds_")] + \
        [c for c in links.columns if c.startswith("r_")] + \
        [c for c in links.columns if c.startswith("p_ptau")]
    print(f"\nfeatures active in any run, {reference} first; own Spearman within AD (test)\n"
          + links[show].round(3).head(60).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.out or reference
    runs_table.to_csv(OUT_DIR / f"{name}_runs.csv", index=False)
    features.to_csv(OUT_DIR / f"{name}_features.csv", index=False)
    links.to_csv(OUT_DIR / f"{name}_links.csv", index=False)
    for suffix in ("runs", "features", "links"):
        print(f"Saved: {OUT_DIR / f'{name}_{suffix}.csv'}")