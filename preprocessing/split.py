"""One train / val / test split, shared by every modality run.

Rules:
    - train = split.train_visits visits of whole subjects without pTau, stratified by diagnosis;
              no train subject appears in val or test
    - test  = every visit with a split.test_anchor value (pTau),
              plus some non-pTau visits of those same subjects, chosen to balance CN/AD,
              until test and val are the same size
    - val   = the remaining subjects, plus the rest of the pTau subjects' non-pTau visits
So a pTau subject can have visits in both val and test; train stays fully separate.
A subject whose diagnosis changes across visits is stratified by its last visit.

Outputs (paths.out_dir):
    split.csv           visit_id, adni_id, split
    split_report.txt    counts per split and overlap checks
"""

import numpy as np
import pandas as pd
from common import load_config, out_dir


def pick_train(subjects, target, rng):
    """Whole subjects per diagnosis until each stratum reaches its share of the target visits."""
    total = subjects["n_visits"].sum()
    train = []
    for _, group in subjects.groupby("last_dx"):
        share = target * group["n_visits"].sum() / total
        order = group.index.tolist()
        rng.shuffle(order)
        got = 0
        for s in order:
            n = group.loc[s, "n_visits"]
            if abs(got + n - share) > abs(got - share):
                continue
            train.append(s)
            got += n

    # whole subjects rarely land exactly on target: fix it with single-visit subjects
    singles_in = [s for s in train if subjects.loc[s, "n_visits"] == 1]
    singles_out = [s for s in subjects.index
                   if s not in set(train) and subjects.loc[s, "n_visits"] == 1]
    rng.shuffle(singles_in)
    rng.shuffle(singles_out)
    total = int(subjects.loc[train, "n_visits"].sum())
    while total > target and singles_in:
        train.remove(singles_in.pop())
        total -= 1
    while total < target and singles_out:
        train.append(singles_out.pop())
        total += 1
    return train


def pick_balancing(candidates, test_dx, n, rng):
    """Choose n visits from candidates, each time taking the diagnosis test has fewer of."""
    pool = {dx: list(rng.permutation(g.index)) for dx, g in candidates.groupby("diagnosis")}
    counts = test_dx.value_counts().to_dict()
    chosen = []
    for _ in range(n):
        available = [dx for dx in pool if pool[dx]]
        if not available:
            break
        dx = min(available, key=lambda d: counts.get(d, 0))
        chosen.append(pool[dx].pop())
        counts[dx] = counts.get(dx, 0) + 1
    return chosen


def build_split(cohort, cfg):
    rng = np.random.default_rng(cfg.get("seed", 42))
    anchor = cfg.get("test_anchor", "ptau217_alzpath")

    cohort = cohort.sort_values(["adni_id", "record_date"]).reset_index(drop=True)
    has_anchor = cohort[anchor].notna()
    anchor_subjects = set(cohort.loc[has_anchor, "adni_id"])

    subjects = cohort.groupby("adni_id").agg(n_visits=("visit_id", "size"),
                                             last_dx=("diagnosis", "last"))
    free = subjects[~subjects.index.isin(anchor_subjects)]

    # train: whole subjects without pTau
    train_subjects = set(pick_train(free, cfg.get("train_visits", 768), rng))
    split = pd.Series("val", index=cohort.index)
    split[cohort["adni_id"].isin(train_subjects)] = "train"

    # test: every pTau visit
    split[has_anchor] = "test"

    # the pTau subjects' other visits: some to test (balancing CN/AD), the rest to val
    extra = cohort[cohort["adni_id"].isin(anchor_subjects) & ~has_anchor]
    n_val_now = int(((split == "val") & ~cohort.index.isin(extra.index)).sum())
    n_test_now = int(has_anchor.sum())
    n_to_test = int(np.clip(round((n_val_now + len(extra) - n_test_now) / 2), 0, len(extra)))
    to_test = pick_balancing(extra, cohort.loc[has_anchor, "diagnosis"], n_to_test, rng)
    split[to_test] = "test"

    out = cohort[["visit_id", "adni_id"]].assign(split=split)
    merged = cohort.assign(split=split)

    report = [(f"anchor: {anchor} | pTau visits {n_test_now} on {len(anchor_subjects)} subjects | "
              f"their other visits {len(extra)}: {len(to_test)} to test, {len(extra) - len(to_test)} to val")]
    for name in ("train", "val", "test"):
        part = merged[merged["split"] == name]
        report.append(f"{name:5s} visits {len(part):4d} | subjects {part['adni_id'].nunique():3d} | "
                      f"{part['diagnosis'].value_counts().to_dict()} | "
                      f"with {anchor}: {int(part[anchor].notna().sum())}")

    splits_per_subject = merged.groupby("adni_id")["split"].agg(set)
    train_shared = splits_per_subject.apply(lambda s: "train" in s and len(s) > 1).sum()
    val_test_shared = splits_per_subject.apply(lambda s: s == {"val", "test"}).sum()
    report.append(f"train subjects also in val/test: {train_shared}  (must be 0)")
    report.append(f"subjects with visits in both val and test: {val_test_shared}  (accepted)")
    report.append(f"visits without a split: {int(out['split'].isna().sum())}")
    return out, report


if __name__ == "__main__":
    config = load_config()
    out = out_dir(config)
    cohort = pd.read_csv(out / "cohort.csv", dtype={"visit_id": str, "adni_id": str})

    split, report = build_split(cohort, config.get("split", {}))
    split.to_csv(out / "split.csv", index=False)
    (out / "split_report.txt").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nSaved: {out / 'split.csv'}")