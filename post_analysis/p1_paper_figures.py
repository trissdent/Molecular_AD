import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from i1_index_core import (
    AGECORR_INDEX,
    PRIMARY_SPLIT,
    PTAU_TARGETS,
    RAW_INDEX,
    drop_excluded,
    safe_spearman,
)
from m3_features import best_epoch
from matplotlib import font_manager
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
    NullFormatter,
)
from sklearn.decomposition import PCA

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "paper"

BLACK, AD_COLOR = "#000000", "#AA66FF"
GROUPS = [("cn", "CN", BLACK, "o"), ("ad", "AD", AD_COLOR, "o")]
MODELS = {"t1": "T1", "dti": "DTI"}
INDEX_LABELS = {RAW_INDEX: "Disease index (uncorrected)", AGECORR_INDEX: "Disease index (age-corrected)"}
TARGET_LABELS = {"mmse": "MMSE", "ptau217_alzpath": "pTau-217 AlzPath (pg/mL)",
                 "ptau217_janssen": "pTau-217 Janssen (pg/mL)", "ptau217": "pTau-217 ELISA (pg/mL)"}
TITLE_SIZE = 13


def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Calibri", "Carlito") if f in names), "DejaVu Sans")
    if family != "Calibri":
        print(f"note: Calibri not found, using {family}")
    plt.rcParams.update({
        "font.family": family, "font.weight": "bold", "font.size": 11,
        "axes.labelweight": "bold", "axes.labelsize": 11, "axes.titleweight": "bold",
        "axes.titlesize": TITLE_SIZE, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.minor.width": 1.0, "ytick.minor.width": 1.0, "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "custom", "mathtext.rm": f"{family}:bold", "mathtext.bf": f"{family}:bold",
        "mathtext.it": f"{family}:italic:bold",
        "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def reported_run(patterns, seed):
    runs = sorted({p for pattern in patterns for p in CHECKPOINTS.glob(pattern)})
    runs = [r for r in runs if yaml.safe_load((LOGS / r.name / "config.yml").read_text())["training"]["seed"] == seed]
    if not runs:
        raise SystemExit(f"no seed-{seed} run among {patterns}")
    return runs[-1]


def adni_index(run):
    df = pd.read_csv(run / "s2_index" / "disease_index.csv", dtype={"visit_id": str, "adni_id": str})
    df = drop_excluded(df[df["split"] == PRIMARY_SPLIT])
    cohort = pd.read_csv(Path(yaml.safe_load((LOGS / run.name / "config.yml").read_text())["data"]["prepared_dir"])
                         / "cohort.csv", dtype={"visit_id": str}, usecols=["visit_id", "apoe_genotype"])
    return with_e4(df.merge(cohort, on="visit_id", how="left"))


def panel_letters(fig, axes):
    fig.canvas.draw()
    renderer, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    for ax, letter in zip(axes, "ABCDEFGH"):
        left = ax.get_tightbbox(renderer).transformed(inv).x0
        fig.text(left, ax.get_position().y1 + 0.012, letter, fontsize=TITLE_SIZE, fontweight="bold",
                 ha="left", va="bottom")


def save(fig, name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    data.to_csv(OUT / f"{name}_data.csv", index=False)
    print(f"Saved: results/paper/{name}.pdf / .png")


def hist_panel(ax, df, index_name, bins=None):
    values = df[index_name].dropna().to_numpy(float)
    bins = np.histogram_bin_edges(values, bins=20) if bins is None else bins
    for code, label, color, _ in GROUPS:
        v = df.loc[df["diagnosis"] == code, index_name].dropna().to_numpy(float)
        ax.hist(v, bins=bins, color=color, alpha=0.45, label=f"{label} (n = {len(v)})")
        ax.hist(v, bins=bins, color=color, histtype="step", linewidth=1.5)
    ax.set_xlabel(INDEX_LABELS[index_name])
    ax.set_ylabel("Number of imaging sessions")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return bins


def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def minus(text):
    return text.replace("-", "−")


def scatter_panel(ax, df, index_name, target):
    for code, label, color, marker in GROUPS:
        g = df[df["diagnosis"] == code].dropna(subset=[index_name, target])
        ax.scatter(g[index_name], g[target], s=30, marker=marker, color=color,
                   edgecolors="white", linewidths=0.4, label=label, zorder=3)
    log = target in PTAU_TARGETS or target == "ptau217"
    result, lines = {"target": target}, []
    for name, group, color in [("All", df, BLACK), ("AD", df[df["diagnosis"] == "ad"], AD_COLOR)]:
        g = group.dropna(subset=[index_name, target])
        x, y = g[index_name].to_numpy(float), g[target].to_numpy(float)
        line_x = np.linspace(x.min(), x.max(), 50)
        line_y = np.polyval(np.polyfit(x, np.log(y) if log else y, 1), line_x)
        ax.plot(line_x, np.exp(line_y) if log else line_y, color=color, linewidth=2, zorder=4)
        r, p = safe_spearman(x, y)
        lines.append(minus(f"{name}: $r_s$ = {r:.2f}, " + format_p(p).replace("p", "$p$", 1)))
        result.update({f"rho_{name}": r, f"p_{name}": p, f"n_{name}": len(g)})
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(1.0, 1.03, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom")
    return result


def match_limits(axes, axis="y"):
    lims = [ax.get_ylim() if axis == "y" else ax.get_xlim() for ax in axes]
    lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
    for ax in axes:
        ax.set_ylim(lo, hi) if axis == "y" else ax.set_xlim(lo, hi)


def fit_legend(ax, extra=True):
    handles, labels = ax.get_legend_handles_labels()
    if extra:
        handles += [plt.Line2D([], [], color=BLACK, linewidth=2), plt.Line2D([], [], color=AD_COLOR, linewidth=2)]
        labels += ["Fit, all", "Fit, AD"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.4)


def fig_scatter(models, index_name, targets, name, box_first=False):
    rows = (["box"] if box_first else []) + list(targets)
    fig, axes = plt.subplots(len(rows), len(models), figsize=(3.8 * len(models), 3.0 * len(rows)),
                             squeeze=False)
    stats = []
    for c, (key, df) in enumerate(models.items()):
        axes[0, c].set_title(MODELS[key], pad=42 if not box_first else 12)
        for r, row in enumerate(rows):
            ax = axes[r, c]
            if row == "box":
                hist_panel(ax, df, index_name)
                ax.set_ylabel("Number of participants")
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.03, 1.0), handletextpad=0.4)
                continue
            stats.append({"model": MODELS[key], **scatter_panel(ax, df, index_name, row)})
            ax.set_ylabel(TARGET_LABELS[row])
            ax.set_xlabel(INDEX_LABELS[index_name])
    scatter_axes = axes[1:] if box_first else axes
    for r in range(scatter_axes.shape[0]):
        match_limits(scatter_axes[r])
    match_limits(scatter_axes.flat, "x")
    for ax in scatter_axes.flat:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    if box_first:
        match_limits(axes[0])
    for ax in axes.flat:
        ax.tick_params(labelbottom=True, labelleft=True)
    fit_legend(scatter_axes[0, -1])
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.align_ylabels(axes[:, 0])
    panel_letters(fig, axes.flatten())
    save(fig, name, pd.DataFrame(stats))
    print(pd.DataFrame(stats).round(3).to_string(index=False))


def fig_apoe(models, partials, index_name, targets, name):
    colors = {1.0: ("APOE ε4 carrier", "#40007F"), 0.0: ("Non-carrier", AD_COLOR)}
    shape = (len(targets), len(models))
    fig, axes = plt.subplots(*shape, figsize=(3.8 * shape[1], 3.0 * shape[0]), squeeze=False)
    for c, (key, df) in enumerate(models.items()):
        axes[0, c].set_title(MODELS[key], pad=24)
        part = partials[key]
        for r, target in enumerate(targets):
            ax = axes[r, c]
            ad = df[df["diagnosis"] == "ad"].dropna(subset=[index_name, target, "e4_carrier"])
            for flag, (label, color) in colors.items():
                g = ad[ad["e4_carrier"] == flag]
                ax.scatter(g[index_name], g[target], s=30, color=color, edgecolors="white",
                           linewidths=0.4, label=label, zorder=3)
            log = target in PTAU_TARGETS or target == "ptau217"
            x, y = ad[index_name].to_numpy(float), ad[target].to_numpy(float)
            line_x = np.linspace(x.min(), x.max(), 50)
            line_y = np.polyval(np.polyfit(x, np.log(y) if log else y, 1), line_x)
            ax.plot(line_x, np.exp(line_y) if log else line_y, color=BLACK, linewidth=2, zorder=4)
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            row = part[(part["index_name"] == index_name) & (part["target"] == target)]
            if len(row):
                rr, pp = float(row["adjusted_r"].iloc[0]), float(row["adjusted_p"].iloc[0])
                ax.text(1.0, 1.03, minus(f"Partial $r_s$ = {rr:.2f}, " + format_p(pp).replace("p", "$p$", 1)),
                        transform=ax.transAxes, ha="right", va="bottom")
            ax.set_ylabel(TARGET_LABELS[target])
            ax.set_xlabel(INDEX_LABELS[index_name])
    for r in range(len(targets)):
        match_limits(axes[r])
    match_limits(list(axes.flat), "x")
    for ax in axes.flat:
        ax.tick_params(labelbottom=True, labelleft=True)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.4)
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.align_ylabels(axes[:, 0])
    panel_letters(fig, axes.flatten())
    save(fig, name, pd.concat([p.assign(model=MODELS[k]) for k, p in partials.items()]))



def fig_latent_external(runs, ext_name, name):
    keys = [k for k, r in runs.items() if (r / f"external_{ext_name}" / "latents.npz").is_file()]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    rows = []
    for ax, key in zip(axes[0], keys):
        run = runs[key]
        adni = np.load(run / "latents_best_ckpt.npz", allow_pickle=True)
        pca = PCA(n_components=2, random_state=0).fit(adni["train_mu"].astype(float))
        test = pca.transform(adni["test_mu"].astype(float))
        test_dx = np.char.lower(adni["test_diagnoses"].astype(str))
        ext = np.load(run / f"external_{ext_name}" / "latents.npz", allow_pickle=True)
        table = external_index(run, ext_name).set_index("subject_id")
        ext_dx = table.loc[ext["subject_ids"].astype(str), "diagnosis"].to_numpy()
        ext_pc = pca.transform(ext["mu"].astype(float))
        for code, label, color, marker in GROUPS:
            m = test_dx == code
            ax.scatter(test[m, 0], test[m, 1], s=18, facecolors="none", edgecolors=color, linewidths=0.8,
                       label=f"ADNI test {label}", zorder=2)
        for code, label, color, marker in GROUPS:
            m = ext_dx == code
            ax.scatter(ext_pc[m, 0], ext_pc[m, 1], s=45, color=color, edgecolors="white", linewidths=0.5,
                       label=f"Vietnamese {label}", zorder=3)
        ax.set_xlabel("PC1 (fitted on ADNI training)")
        ax.set_ylabel("PC2")
        ax.set_title(MODELS[key], pad=12)
        rows.append(pd.DataFrame({"model": MODELS[key], "cohort": "Vietnamese", "diagnosis": ext_dx,
                                  "pc1": ext_pc[:, 0], "pc2": ext_pc[:, 1]}))
        rows.append(pd.DataFrame({"model": MODELS[key], "cohort": "ADNI test", "diagnosis": test_dx,
                                  "pc1": test[:, 0], "pc2": test[:, 1]}))
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.4)
    fig.tight_layout(w_pad=2.5)
    panel_letters(fig, axes[0])
    save(fig, name, pd.concat(rows))

def fig_index_vs_age(models, name):
    fig, axes = plt.subplots(len(models), 2, figsize=(7.6, 3.0 * len(models)), squeeze=False)
    stats = []
    for r, (key, df) in enumerate(models.items()):
        for c, index_name in enumerate([RAW_INDEX, AGECORR_INDEX]):
            ax = axes[r, c]
            for code, label, color, marker in GROUPS:
                g = df[df["diagnosis"] == code]
                ax.scatter(g["age"], g[index_name], s=30, marker=marker, color=color, edgecolors="white",
                           linewidths=0.4, label=label, zorder=3)
            rr, pp = safe_spearman(df["age"].to_numpy(float), df[index_name].to_numpy(float))
            ax.text(1.0, 1.03, minus("$r_s$ with age = " + f"{rr:.2f}, " + format_p(pp).replace("p", "$p$", 1)),
                    transform=ax.transAxes, ha="right", va="bottom")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel(f"{MODELS[key]} disease index")
            ax.yaxis.set_major_locator(MultipleLocator(1))
            if r == 0:
                ax.set_title("Uncorrected" if c == 0 else "Age-corrected", pad=24)
            stats.append({"model": MODELS[key], "index": index_name, "rho_age": rr, "p": pp})
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.4)
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.align_ylabels(axes[:, 0])
    panel_letters(fig, axes.flatten())
    save(fig, name, pd.DataFrame(stats))


def fig_decodability(runs, name):
    order = ["Diagnosis", "Head size", "Age", "Sex"]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.2), sharex=True, squeeze=False)
    data = []
    for ax, (key, run) in zip(axes[0], runs.items()):
        d = pd.read_csv(run / "s3_alignment" / "alignment_decodability.csv")
        d = d[d["split"] == "test"]
        first = d.iloc[0]
        rows = [("Diagnosis", first["diagnosis_before"], first["diagnosis_after"],
                 first["diagnosis_features"], first["diagnosis_chance"])]
        rows += [(r["nuisance"].capitalize(), r["nuisance_before"], r["nuisance_after"],
                  r["nuisance_features"], r["nuisance_chance"]) for _, r in d.iterrows()]
        t = pd.DataFrame(rows, columns=["variable", "image", "latent", "features", "chance"])
        t = t.set_index("variable").loc[[v for v in order if v in set(t["variable"])]].reset_index()
        y = np.arange(len(t))[::-1]
        for yy, chance in zip(y, t["chance"]):
            ax.plot([chance, chance], [yy - 0.35, yy + 0.35], color="#808080", linestyle="--", linewidth=1.5)
        for yy, a, b in zip(y, t["image"], t["latent"]):
            ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops={"arrowstyle": "->", "color": AD_COLOR, "lw": 1.5})
        ax.scatter(t["image"], y, s=60, facecolors="white", edgecolors=BLACK, linewidths=1.5,
                   label="Image (before encoding)", zorder=3)
        ax.scatter(t["latent"], y, s=60, color=AD_COLOR, label="Latent (after encoding)", zorder=3)
        ax.scatter(t["features"], y, s=60, marker="s", facecolors="white", edgecolors="#808080",
                   linewidths=1.5, label="Regional features", zorder=3)
        ax.set_yticks(y, t["variable"])
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_xlabel("Balanced accuracy (test)")
        ax.set_title(MODELS[key], pad=12)
        ax.tick_params(labelbottom=True)
        data.append(t.assign(model=MODELS[key]))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#808080", linestyle="--", linewidth=1.5))
    fig.legend(handles, labels + ["Chance"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.5)
    panel_letters(fig, axes[0])
    save(fig, name, pd.concat(data))


def external_index(run, name):
    csv = run / f"external_{name}" / "index.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv, dtype={"subject_id": str})
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.lower()
    return df


def with_e4(df):
    df = df.copy()
    if "e4_carrier" not in df:
        df["e4_carrier"] = df["apoe_genotype"].map(lambda g: float("4" in g) if isinstance(g, str) else np.nan)
    return df


def fig_box(models, index_name, name):
    fig, axes = plt.subplots(1, len(models), figsize=(3.8 * len(models), 3.2), squeeze=False)
    axes = axes[0]
    data = []
    allv = pd.concat([df[index_name] for df in models.values()]).dropna().to_numpy(float)
    bins = np.histogram_bin_edges(allv, bins=24)
    for ax, (key, df) in zip(axes, models.items()):
        hist_panel(ax, df, index_name, bins)
        ax.set_title(MODELS[key], pad=12)
        data.append(df[["visit_id", "diagnosis", index_name]].assign(model=MODELS[key]))
    match_limits(axes)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.4)
    fig.tight_layout(w_pad=2.5)
    panel_letters(fig, axes)
    save(fig, name, pd.concat(data))


TRACTS = {
    "GCC": "Genu of corpus callosum", "BCC": "Body of corpus callosum",
    "SCC": "Splenium of corpus callosum", "CC": "Corpus callosum", "FX": "Fornix",
    "FX/ST": "Fornix (cres) / stria terminalis", "FXST": "Fornix (cres) / stria terminalis",
    "CST": "Corticospinal tract", "ALIC": "Anterior limb of internal capsule",
    "PLIC": "Posterior limb of internal capsule", "RLIC": "Retrolenticular internal capsule",
    "IC": "Internal capsule", "ACR": "Anterior corona radiata", "SCR": "Superior corona radiata",
    "PCR": "Posterior corona radiata", "CR": "Corona radiata", "PTR": "Posterior thalamic radiation",
    "SS": "Sagittal stratum", "EC": "External capsule", "CGC": "Cingulum (cingulate gyrus)",
    "CGH": "Cingulum (hippocampus)", "SLF": "Superior longitudinal fasciculus",
    "SFO": "Superior fronto-occipital fasciculus", "IFO": "Inferior fronto-occipital fasciculus",
    "UNC": "Uncinate fasciculus", "WholeSkeleton": "Whole skeleton",
}
CORTEX = {
    "bankssts": "Banks of superior temporal sulcus", "caudalanteriorcingulate": "Caudal anterior cingulate",
    "caudalmiddlefrontal": "Caudal middle frontal", "cuneus": "Cuneus", "entorhinal": "Entorhinal",
    "fusiform": "Fusiform", "inferiorparietal": "Inferior parietal", "inferiortemporal": "Inferior temporal",
    "isthmuscingulate": "Isthmus cingulate", "lateraloccipital": "Lateral occipital",
    "lateralorbitofrontal": "Lateral orbitofrontal", "lingual": "Lingual",
    "medialorbitofrontal": "Medial orbitofrontal", "middletemporal": "Middle temporal",
    "parahippocampal": "Parahippocampal", "paracentral": "Paracentral",
    "parsopercularis": "Pars opercularis", "parsorbitalis": "Pars orbitalis",
    "parstriangularis": "Pars triangularis", "pericalcarine": "Pericalcarine",
    "postcentral": "Postcentral", "posteriorcingulate": "Posterior cingulate",
    "precentral": "Precentral", "precuneus": "Precuneus",
    "rostralanteriorcingulate": "Rostral anterior cingulate", "rostralmiddlefrontal": "Rostral middle frontal",
    "superiorfrontal": "Superior frontal", "superiorparietal": "Superior parietal",
    "superiortemporal": "Superior temporal", "supramarginal": "Supramarginal",
    "frontalpole": "Frontal pole", "temporalpole": "Temporal pole",
    "transversetemporal": "Transverse temporal", "insula": "Insula",
    "hippocampus": "Hippocampus", "amygdala": "Amygdala", "thalamus": "Thalamus",
    "thalamus-proper": "Thalamus", "putamen": "Putamen", "caudate": "Caudate",
    "pallidum": "Pallidum", "accumbens-area": "Accumbens", "accumbens": "Accumbens",
}
ASEG_NAMES = {
    "hippocampus": "Hippocampus", "amygdala": "Amygdala", "inf-lat-vent": "Inferior lateral ventricle",
    "lateral-ventricle": "Lateral ventricle", "thalamus-proper": "Thalamus", "thalamus": "Thalamus",
    "putamen": "Putamen", "caudate": "Caudate", "pallidum": "Pallidum", "accumbens-area": "Accumbens",
    "ventraldc": "Ventral diencephalon", "choroid-plexus": "Choroid plexus",
    "cerebellum-cortex": "Cerebellar cortex", "cerebellum-white-matter": "Cerebellar white matter",
}
HIPPO_NAMES = {
    "molecular-layer-hp": "Molecular layer", "subiculum": "Subiculum", "presubiculum": "Presubiculum",
    "parasubiculum": "Parasubiculum", "ca1": "CA1", "ca3": "CA3", "ca4": "CA4",
    "gc-ml-dg": "Dentate gyrus", "hippocampal-tail": "Hippocampal tail", "hata": "HATA",
    "fimbria": "Fimbria", "hippocampal-fissure": "Hippocampal fissure",
    "whole-hippocampus": "Whole hippocampus", "whole-hippocampal": "Hippocampus",
}
AMYG_NAMES = {
    "lateral-nucleus": "lateral", "basal-nucleus": "basal", "accessory-basal-nucleus": "accessory basal",
    "anterior-amygdaloid-area-aaa": "anterior area", "central-nucleus": "central",
    "medial-nucleus": "medial", "cortical-nucleus": "cortical",
    "corticoamygdaloid-transitio": "corticoamygdaloid transition",
    "corticoamygdaloid-transition": "corticoamygdaloid transition",
    "paralaminar-nucleus": "paralaminar", "whole-amygdala": "whole",
}
SIDE = {"lh": "L", "rh": "R", "left": "L", "right": "R", "l": "L", "r": "R"}
UNMATCHED = set()


def _key(text):
    return re.sub(r"[\s_]+", "-", text.strip()).lower()


def _with_side(text, side):
    return f"{text} {side}" if side else text


def _t1_name(s):
    if m := re.match(r"^aseg[ .]+(Left|Right)-(.+?)(?: volume)?$", s):
        region = _key(m.group(2))
        return _with_side(ASEG_NAMES.get(region, m.group(2).replace("-", " ")), SIDE[m.group(1).lower()])
    if m := re.match(r"^(lh|rh)\.hipposubfields[ .]+(.+?)(?: volume)?$", s):
        part, sub = _key(m.group(2)), ""
        if hs := re.match(r"^(.+)-(head|body)$", part):
            part, sub = hs.group(1), " " + hs.group(2)
        if part in HIPPO_NAMES:
            return _with_side(HIPPO_NAMES[part] + sub, SIDE[m.group(1)])
    if m := re.match(r"^(lh|rh)\.amygdalar-nuclei[ .]+(.+?)(?: volume)?$", s):
        part = _key(m.group(2))
        if part in AMYG_NAMES:
            return _with_side(f"Amygdala, {AMYG_NAMES[part]}", SIDE[m.group(1)])
    return None


def _gm_region(rest):
    for pattern, side_first in [(r"ctx-(lh|rh)-(\w+)", True), (r"(Left|Right)-(.+)", True),
                                (r"(lh|rh)[\s_.-]+(\w+)", True),
                                (r"(.+?)[\s_-]+(L|R|lh|rh|left|right)", False), (r"(.+)", None)]:
        if m := re.fullmatch(pattern, rest, flags=re.IGNORECASE):
            if side_first is None:
                region, side = m.group(1), None
            else:
                side, region = (m.group(1), m.group(2)) if side_first else (m.group(2), m.group(1))
            name = CORTEX.get(_key(region)) or CORTEX.get(_key(region).replace("-", ""))
            if name:
                return name, SIDE.get(side.lower()) if side else None
    return None, None


def _dti_name(s):
    m = re.match(r"^(FA|MD|AxD|RD)\b[\s,_:]*(.*)$", s)
    if not m:
        return None
    measure, rest = m.group(1), m.group(2)
    gray = bool(re.search(r"gray[\s_-]*matter|\bgm\b", rest, flags=re.IGNORECASE))
    rest = re.sub(r"gray[\s_-]*matter|\bgm\b", "", rest, flags=re.IGNORECASE).strip(" ,_:")
    if "(" in rest:
        rest = re.findall(r"\(([^()]*)\)", rest)[-1]
    tract = re.fullmatch(r"([A-Za-z]+(?:/[A-Za-z]+)?)(?:-([LR]))?", rest)
    if not gray and tract and tract.group(1) in TRACTS:
        return f"{_with_side(TRACTS[tract.group(1)], tract.group(2))}, WM {measure}"
    region, side = _gm_region(rest)
    if region:
        return f"{_with_side(region, side)}, GM {measure}"
    return None


def readable(name):
    s = str(name).strip()
    spaced = s.replace("_", " ")
    out = _dti_name(spaced) or _t1_name(spaced)
    if out is None:
        UNMATCHED.add(s)
        return s.replace("_", " ")
    return out


def cimlr_selection(run):
    epoch = best_epoch(run.name)
    history = pd.read_csv(LOGS / run.name / "topk_history.csv")
    row = history[history["epoch"] == epoch]
    row = row if len(row) else history.head(1)
    order = [int(x) for x in row.iloc[0]["feature_idx"].split("|")]
    ranking = pd.read_csv(run / "s1_latent_space" / "feature_ranking.csv").set_index("feature_idx")
    n_best = set(ranking.nlargest(len(order), "test_r2").index)
    selected = ranking.loc[order, ["feature_name", "test_r2"]].reset_index()
    selected["rank"] = np.arange(1, len(order) + 1)
    selected["also_best_encoded"] = selected["feature_idx"].isin(n_best)
    return selected, float(ranking["test_r2"].mean())

def fig_cimlr(runs, name):
    for letter, (key, run) in zip("ab", runs.items()):
        sel, pool_mean = cimlr_selection(run)
        fig, ax = plt.subplots(figsize=(8, 0.27 * len(sel) + 1.8))
        y = np.arange(len(sel))[::-1]
        colors = ["#40007F" if f else AD_COLOR for f in sel["also_best_encoded"]]
        ax.barh(y, sel["test_r2"], color=colors, height=0.7)
        ax.axvline(pool_mean, color="#808080", linestyle="--", linewidth=1.5)
        ax.set_yticks(y, [readable(f) for f in sel["feature_name"]])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Variance explained by the latent (R², test)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, len(sel) - 0.3)
        ax.set_title(MODELS[key], pad=12)
        handles = [plt.Rectangle((0, 0), 1, 1, color="#40007F"), plt.Rectangle((0, 0), 1, 1, color=AD_COLOR),
                   plt.Line2D([], [], color="#808080", linestyle="--", linewidth=1.5)]
        fig.legend(handles, ["Selected, also among the best encoded", "Selected only", "Mean over all features"],
                   loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, f"{name}{letter}_{key}", sel.assign(model=MODELS[key], pool_mean_r2=pool_mean))
    if UNMATCHED:
        print(f"note: Fig. 5 names not recognised, shown as they are: {sorted(UNMATCHED)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", nargs="+", required=True)
    parser.add_argument("--dti", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external", default=None)
    args = parser.parse_args()

    set_style()
    runs = {"t1": reported_run(args.t1, args.seed), "dti": reported_run(args.dti, args.seed)}
    for key, run in runs.items():
        print(f"{MODELS[key]}: {run.name}")
    adni = {k: adni_index(r) for k, r in runs.items()}

    fig_box(adni, RAW_INDEX, "fig2_index_uncorrected")
    fig_box(adni, AGECORR_INDEX, "fig3_index_agecorr")
    targets = ["mmse", "ptau217_alzpath", "ptau217_janssen"]
    fig_scatter(adni, AGECORR_INDEX, targets, "fig4_mmse_ptau")
    fig_scatter(adni, RAW_INDEX, targets, "supp_fig4_uncorrected")
    fig_cimlr(runs, "fig5")
    fig_decodability(runs, "fig6_decodability")
    fig_index_vs_age(adni, "supp_index_vs_age")
    partials = {k: pd.read_csv(r / "s4_apoe" / "apoe_partial_model.csv") for k, r in runs.items()}
    fig_apoe(adni, partials, AGECORR_INDEX, targets, "supp_apoe_partial")
    if args.external:
        ext = {k: d for k, d in ((k, external_index(r, args.external)) for k, r in runs.items()) if d is not None}
        missing = [MODELS[k] for k in runs if k not in ext]
        if missing:
            print(f"no external_{args.external} output for {', '.join(missing)}: Fig. 7 drawn without it")
        if ext:
            fig_scatter(ext, AGECORR_INDEX, ["mmse", "ptau217"], "fig7_external", box_first=True)
            fig_scatter(ext, RAW_INDEX, ["mmse", "ptau217"], "supp_fig7_uncorrected", box_first=True)
            ext_partials = {k: pd.read_csv(runs[k] / f"external_{args.external}" / "s4_apoe" / "apoe_partial_model.csv")
                            for k in ext}
            fig_apoe({k: with_e4(d) for k, d in ext.items()}, ext_partials, AGECORR_INDEX, ["mmse", "ptau217"],
                     "supp_apoe_partial_external")
            fig_latent_external(runs, args.external, "supp_latent_space_external")