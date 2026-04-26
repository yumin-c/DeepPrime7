"""
Exhaustive evaluation matrix (7 models x 7 datasets) with multiple modes:
- Overall Spearman
- PAM-stratified Spearman (NGG / Non-NGG)
- Edit-length-wise Spearman curves for the four PE7 targets

Outputs are written under eval_grid/.

# Example usage:
python eval/eval_grid.py
--mathis-path data/mathis_2024_deepprime_input.csv
--pridict2-preds data/24k_final_pridict_test_preds_5fold.csv
--batch-size 1024

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import run_deepprime_original as rdo


# NPJ-like minimal style for plots
def set_npj_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3,
            "xtick.major.width": 1.0,
            "ytick.major.size": 3,
            "ytick.major.width": 1.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": True,
        }
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
DEEPPRIME_ROOT = REPO_ROOT.parent / "DeepPrime"
OUT_DIR_DEFAULT = REPO_ROOT / "eval_grid"

DATASETS = {
    "DP_base": DEEPPRIME_ROOT / "data" / "DeepPrime_dataset_final_Feat8.csv",
    "DP_A549": DEEPPRIME_ROOT / "data" / "DP_variant_A549_PE2max_Opti_221114.csv",
    "DP_DLD1": DEEPPRIME_ROOT / "data" / "DP_variant_DLD1_PE2max_Opti_221114.csv",
    # PE7 targets (mapped later)
    "PE7_DN": REPO_ROOT / "data" / "processed_test.csv",
    "PE7_DC": REPO_ROOT / "data" / "processed_test.csv",
    "PE7_DT": REPO_ROOT / "data" / "processed_test.csv",
    "PE7_AN": REPO_ROOT / "data" / "processed_test.csv",
}

PE7_LABEL_MAP = {
    "PE7_DN": "DN_M_EF",
    "PE7_DC": "DC_M_EF",
    "PE7_DT": "DT_M_EF",
    "PE7_AN": "AN_M_EF",
}

MODEL_NAMES = [
    "DP_base",
    "DP_DLD1",
    "DP_A549",
    "DP7_DN",
    "DP7_DC",
    "DP7_DT",
    "DP7_AN",
    "PRIDICT2_HEK",
    "PRIDICT2_K562",
]

MODEL_DISPLAY = {
    "DP_base": "DeepPrime",
    "DP_DLD1": "DeepPrime-FT (DLD1, PE2max)",
    "DP_A549": "DeepPrime-FT (A549, PE2max)",
    "DP7_DN": "DeepPrime7 (DLD1, NGG)",
    "DP7_DC": "DeepPrime7 (DLD1, NRCH)",
    "DP7_DT": "DeepPrime7 (DLD1, NRTH)",
    "DP7_AN": "DeepPrime7 (A549, NGG)",
    "PRIDICT2_HEK": "PRIDICT2 (HEK)",
    "PRIDICT2_K562": "PRIDICT2 (K562)",
}

DATASET_DISPLAY = {
    "DP_base": "Library-ClinVar (HEK293T, PE2)",
    "DP_A549": "Library-Small (A549, PE2max)",
    "DP_DLD1": "Library-Small (DLD1, PE2max)",
    "PE7_DN": "Library-Balanced (DLD1, PE7max)",
    "PE7_DC": "Library-Balanced (DLD1, PE7max-NRCH)",
    "PE7_DT": "Library-Balanced (DLD1, PE7max-NRTH)",
    "PE7_AN": "Library-Balanced (A549, PE7max)",
    "PRIDICT_HEK": "Mathis 2024 (HEK)",
    "PRIDICT_K562": "Mathis 2024 (K562)",
}

LENGTH_GROUPS = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 8),
    (9, 11),
    (12, 14),
    (15, 17),
    (18, 20),
]


def load_dp_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Fold"] == "Test"].reset_index(drop=True)
    return df


def prepare_pe7_deepprime(pe7_path: Path, source_path: Path) -> pd.DataFrame:
    """Merge PE7 test with raw to obtain DeepPrime feature schema."""
    pe7_df, merged = rdo._load_pe7(pe7_path, source_path, verbose=False)
    dp_df = rdo.convert_to_deepprime_schema(merged)
    # carry over measurement columns for evaluation
    for col in ["DN_M_EF", "DC_M_EF", "DT_M_EF", "AN_M_EF"]:
        if col in pe7_df.columns:
            dp_df[col] = pe7_df[col].values
    dp_df["idx"] = pe7_df["idx"].values
    return dp_df


def get_pam_mask(df: pd.DataFrame, seq_col: str) -> Dict[str, np.ndarray]:
    pam = df[seq_col].str.slice(24, 27)
    ngg = pam.isin(["GGG", "AGG", "TGG", "CGG"])
    return {"NGG": ngg.to_numpy(), "Non-NGG": (~ngg).to_numpy()}


def spearman_with_mask(y: np.ndarray, v: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 3:
        return np.nan
    return float(pd.Series(v[mask]).corr(pd.Series(y[mask]), method="spearman"))


def evaluate_dataset(
    name: str,
    df_dp: pd.DataFrame,
    label_col: str,
    device: torch.device,
    base_models: List[Path],
    dld1_models: List[Path],
    a549_models: List[Path],
    dp7_path: Path,
    dp7_scaler: Path,
    batch_size: int | None = 1024,
) -> Dict[str, Dict[str, float]]:
    """Return dict: mode -> model -> spearman."""
    results: Dict[str, Dict[str, float]] = {}

    # Prepare tensors for DeepPrime ensembles
    g_tensor, x_tensor = rdo._tensorize_inputs(df_dp, device)
    pred_dp_base = rdo.ensemble_predict(base_models, g_tensor, x_tensor, device, batch_size=batch_size)
    pred_dp_dld1 = rdo.ensemble_predict(dld1_models, g_tensor, x_tensor, device, batch_size=batch_size)
    pred_dp_a549 = rdo.ensemble_predict(a549_models, g_tensor, x_tensor, device, batch_size=batch_size)

    # DeepPrime7 inputs
    dp7_input = rdo._dp_dataset_to_dp7(df_dp)
    _, dp7_pred_map = rdo.deepprime7_predict(
        dp7_input, device, dp7_path, dp7_scaler, batch_size=batch_size
    )

    preds = {
        "DP_base": pred_dp_base,
        "DP_DLD1": pred_dp_dld1,
        "DP_A549": pred_dp_a549,
        "DP7_DN": dp7_pred_map["DN_M_EF"],
        "DP7_DC": dp7_pred_map["DC_M_EF"],
        "DP7_DT": dp7_pred_map["DT_M_EF"],
        "DP7_AN": dp7_pred_map["AN_M_EF"],
    }

    y = df_dp[label_col].to_numpy()
    mask_valid = ~pd.isna(y)

    # Mode 1: overall
    results["overall"] = {m: spearman_with_mask(y, v, mask_valid) for m, v in preds.items()}

    # Mode 2: PAM stratified
    seq_col = "WT74_On" if "WT74_On" in df_dp.columns else "Target"
    pam_masks = get_pam_mask(df_dp, seq_col)
    results["PAM_NGG"] = {
        m: spearman_with_mask(y, v, mask_valid & pam_masks["NGG"]) for m, v in preds.items()
    }
    results["PAM_nonNGG"] = {
        m: spearman_with_mask(y, v, mask_valid & pam_masks["Non-NGG"]) for m, v in preds.items()
    }

    return {"preds": preds, "metrics": results}


def line_plot_lengths(plot_dir: Path, dataset_name: str, lengths: Dict[str, Dict], allowed_models: List[str], metric: str = "spearman") -> None:
    plt.figure(figsize=(8, 4))
    labels = lengths["labels"]
    x_pos = np.arange(len(labels))
    for model in allowed_models:
        ys = [lengths["metrics"][metric].get((lo, hi), {}).get(model, np.nan) for (lo, hi) in lengths["groups"]]
        disp_name = MODEL_DISPLAY.get(model, model)
        plt.plot(x_pos, ys, marker="o", label=disp_name)
    plt.xticks(x_pos, labels, rotation=30)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.legend(fontsize=8, ncol=2, title="")
    if metric == "spearman":
        plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{dataset_name}_length_curve_{metric}.png", dpi=200)
    plt.close()


def corr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    if mask.sum() < 2:
        return {"spearman": np.nan, "pearson": np.nan, "n": int(mask.sum())}
    yt = pd.Series(y_true[mask])
    yp = pd.Series(y_pred[mask])
    return {
        "spearman": float(yt.corr(yp, method="spearman")),
        "pearson": float(yt.corr(yp, method="pearson")),
        "n": int(mask.sum()),
    }


def scatter_actual_vs_pred(
    dataset_name: str,
    df_dp: pd.DataFrame,
    label_col: str,
    preds: Dict[str, np.ndarray],
    models: List[str],
    plot_dir: Path,
) -> None:
    """Draw NPJ-style scatter plots for selected models on a dataset."""
    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
    y_true = df_dp[label_col].to_numpy()
    vals = [y_true[~pd.isna(y_true)]]
    for m in models:
        vals.append(preds[m][~pd.isna(preds[m])])
    data_min = min(np.min(v) for v in vals if len(v) > 0)
    data_max = max(np.max(v) for v in vals if len(v) > 0)
    pad = 0.02 * (data_max - data_min) if data_max != data_min else 1.0
    # lo, hi = data_min - pad, data_max + pad
    lo, hi = 0.0, 100.0

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    for idx, m in enumerate(models):
        metrics = corr_metrics(y_true, preds[m])
        mask = (~pd.isna(y_true)) & (~pd.isna(preds[m]))
        disp = MODEL_DISPLAY.get(m, m)
        label = f"{disp} (ρ={metrics['spearman']:.2f}, r={metrics['pearson']:.2f})"
        sns.scatterplot(
            x=y_true[mask],
            y=preds[m][mask],
            ax=ax,
            s=5,
            alpha=1.0,
            edgecolor="w",
            linewidth=0.5,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.plot([lo, hi], [lo, hi], ls="--", color="dimgray", lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured efficiency")
    ax.set_ylabel("Predicted efficiency")
    ax.set_title(DATASET_DISPLAY.get(dataset_name, dataset_name))
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    fname = f"scatter_{dataset_name}.png"
    plt.savefig(plot_dir / fname, dpi=300)
    plt.close()


def compute_length_curves(
    df_dp: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    label_col: str,
    mask_type: pd.Series,
) -> Dict[str, Dict]:
    metrics: Dict[str, Dict[tuple, Dict[str, float]]] = {"spearman": {}, "mae": {}}
    y = df_dp[label_col].to_numpy()
    mask_base = (~pd.isna(y)) & mask_type.to_numpy()
    for lo, hi in LENGTH_GROUPS:
        mask = mask_base & (df_dp["Edit_len"] >= lo) & (df_dp["Edit_len"] <= hi)
        if mask.sum() < 3:
            continue
        metrics["spearman"][(lo, hi)] = {}
        metrics["mae"][(lo, hi)] = {}
        for m, v in preds.items():
            metrics["spearman"][(lo, hi)][m] = float(pd.Series(v[mask]).corr(pd.Series(y[mask]), method="spearman"))
            metrics["mae"][(lo, hi)][m] = float(np.mean(np.abs(v[mask] - y[mask])))
    labels = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in LENGTH_GROUPS]
    return {"groups": LENGTH_GROUPS, "metrics": metrics, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="7x7 evaluation grid for DeepPrime and DeepPrime7 models.")
    parser.add_argument("--pe7-path", type=Path, default=DATASETS["PE7_DN"])
    parser.add_argument("--pe7-source", type=Path, default=REPO_ROOT / "data" / "250716_24k_final.csv")
    parser.add_argument("--mathis-path", type=Path, default=None)
    parser.add_argument(
        "--mathis-skipped",
        type=Path,
        default=REPO_ROOT / "data" / "mathis_2024_deepprime_input_skipped.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--dp7-path", type=Path, default=REPO_ROOT / "models" / "deepprime7.pth")
    parser.add_argument("--dp7-scaler", type=Path, default=REPO_ROOT / "models" / "deepprime7_scaler.pth")
    parser.add_argument("--pridict2-preds", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    set_npj_style()
    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    base_models = sorted((DEEPPRIME_ROOT / "models" / "ontarget" / "final").glob("model_*.pt"))
    dld1_models = sorted((DEEPPRIME_ROOT / "models" / "ontarget_variants" / "DP_variant_DLD1_PE2max_Opti_221114").glob("final_model_*.pt"))
    a549_models = sorted((DEEPPRIME_ROOT / "models" / "ontarget_variants" / "DP_variant_A549_PE2max_Opti_221114").glob("final_model_*.pt"))

    # Prepare datasets (DeepPrime-formatted frames with label col)
    dp_datasets = {
        "DP_base": (load_dp_dataset(DATASETS["DP_base"]), "Measured_PE_efficiency"),
        "DP_A549": (load_dp_dataset(DATASETS["DP_A549"]), "Measured_PE_efficiency"),
        "DP_DLD1": (load_dp_dataset(DATASETS["DP_DLD1"]), "Measured_PE_efficiency"),
    }

    # PE7 datasets (mapped to DeepPrime schema)
    dp_pe7 = prepare_pe7_deepprime(args.pe7_path, args.pe7_source)
    for name, col in PE7_LABEL_MAP.items():
        df = dp_pe7.copy()
        df[col] = df[col]  # ensure column exists
        dp_datasets[name] = (df, col)

    if args.mathis_path:
        mathis_df = pd.read_csv(args.mathis_path)
        if args.mathis_skipped and args.mathis_skipped.exists() and "idx" in mathis_df.columns:
            skipped = pd.read_csv(args.mathis_skipped)
            if "idx" in skipped.columns:
                mathis_df = mathis_df.loc[~mathis_df["idx"].isin(skipped["idx"])].reset_index(drop=True)
        dp_mathis = rdo.convert_to_deepprime_schema(mathis_df)
        dp_mathis["HEKaverageedited"] = mathis_df["HEKaverageedited"].values
        dp_mathis["K562averageedited"] = mathis_df["K562averageedited"].values
        if "PRIDICT2_0_editing_Score_deep_HEK_mean5fold" in mathis_df.columns:
            dp_mathis["PRIDICT2_HEK"] = mathis_df["PRIDICT2_0_editing_Score_deep_HEK_mean5fold"].values
        if "PRIDICT2_0_editing_Score_deep_K562_mean5fold" in mathis_df.columns:
            dp_mathis["PRIDICT2_K562"] = mathis_df["PRIDICT2_0_editing_Score_deep_K562_mean5fold"].values
        dp_datasets["PRIDICT_HEK"] = (dp_mathis, "HEKaverageedited")
        dp_datasets["PRIDICT_K562"] = (dp_mathis, "K562averageedited")

    pridict_map = None
    if args.pridict2_preds:
        pridict_df = pd.read_csv(args.pridict2_preds)
        if "seq_id" not in pridict_df.columns:
            raise KeyError("PRIDICT2 preds must include seq_id column.")
        pridict_map = pridict_df.set_index("seq_id")[["PRIDICT2_HEK", "PRIDICT2_K562"]]

    overall_heat = []
    pam_heat_ngg = []
    pam_heat_non = []
    logs = []
    n_map: Dict[str, int] = {}

    # Length plots only for PE7 datasets
    length_curves: Dict[str, Dict[int, Dict[str, float]]] = {}
    scatter_targets = {
        "PE7_DN": ["DP_base", "DP_DLD1", "DP7_DN", "PRIDICT2_HEK", "PRIDICT2_K562"],
        "DP_DLD1": ["DP_base", "DP_DLD1", "DP7_DN"],
        "DP_base": ["DP_base", "DP_DLD1", "DP7_DN"],
    }

    for dname, (df_dp, label_col) in dp_datasets.items():
        eval_res = evaluate_dataset(
            dname,
            df_dp,
            label_col,
            device,
            base_models,
            dld1_models,
            a549_models,
            args.dp7_path,
            args.dp7_scaler,
            batch_size=args.batch_size,
        )
        metrics = eval_res["metrics"]
        preds = eval_res["preds"]

        if pridict_map is not None and dname.startswith("PE7"):
            if "idx" not in df_dp.columns:
                raise KeyError("PE7 datasets must include idx for PRIDICT2 merge.")
            idx_series = df_dp["idx"]
            extra_preds = {
                "PRIDICT2_HEK": idx_series.map(pridict_map["PRIDICT2_HEK"]).to_numpy(),
                "PRIDICT2_K562": idx_series.map(pridict_map["PRIDICT2_K562"]).to_numpy(),
            }
            preds.update(extra_preds)
            y = df_dp[label_col].to_numpy()
            mask_valid = ~pd.isna(y)
            seq_col = "WT74_On" if "WT74_On" in df_dp.columns else "Target"
            pam_masks = get_pam_mask(df_dp, seq_col)
            for model_name, vec in extra_preds.items():
                metrics["overall"][model_name] = spearman_with_mask(y, vec, mask_valid)
                metrics["PAM_NGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["NGG"]
                )
                metrics["PAM_nonNGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["Non-NGG"]
                )
        if dname == "PRIDICT_HEK":
            y = df_dp[label_col].to_numpy()
            seq_col = "WT74_On" if "WT74_On" in df_dp.columns else "Target"
            pam_masks = get_pam_mask(df_dp, seq_col)
            for model_name, col in [("PRIDICT2_HEK", "PRIDICT2_HEK"), ("PRIDICT2_K562", "PRIDICT2_K562")]:
                if col not in df_dp.columns:
                    continue
                vec = df_dp[col].to_numpy()
                mask_valid = (~pd.isna(y)) & (~pd.isna(vec))
                preds[model_name] = vec
                metrics["overall"][model_name] = spearman_with_mask(y, vec, mask_valid)
                metrics["PAM_NGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["NGG"]
                )
                metrics["PAM_nonNGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["Non-NGG"]
                )
        if dname == "PRIDICT_K562":
            y = df_dp[label_col].to_numpy()
            seq_col = "WT74_On" if "WT74_On" in df_dp.columns else "Target"
            pam_masks = get_pam_mask(df_dp, seq_col)
            for model_name, col in [("PRIDICT2_K562", "PRIDICT2_K562"), ("PRIDICT2_HEK", "PRIDICT2_HEK")]:
                if col not in df_dp.columns:
                    continue
                vec = df_dp[col].to_numpy()
                mask_valid = (~pd.isna(y)) & (~pd.isna(vec))
                preds[model_name] = vec
                metrics["overall"][model_name] = spearman_with_mask(y, vec, mask_valid)
                metrics["PAM_NGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["NGG"]
                )
                metrics["PAM_nonNGG"][model_name] = spearman_with_mask(
                    y, vec, mask_valid & pam_masks["Non-NGG"]
                )

        overall_heat.append({"dataset": dname, **metrics["overall"]})
        pam_heat_ngg.append({"dataset": dname, **metrics["PAM_NGG"]})
        pam_heat_non.append({"dataset": dname, **metrics["PAM_nonNGG"]})

        logs.append(f"[{DATASET_DISPLAY.get(dname, dname)}] overall: {metrics['overall']}")
        logs.append(f"[{DATASET_DISPLAY.get(dname, dname)}] PAM_NGG: {metrics['PAM_NGG']}")
        logs.append(f"[{DATASET_DISPLAY.get(dname, dname)}] PAM_nonNGG: {metrics['PAM_nonNGG']}")
        n_map[dname] = int((~pd.isna(df_dp[label_col])).sum())

        if dname in scatter_targets:
            models_for_scatter = [m for m in scatter_targets[dname] if m in preds]
            if models_for_scatter:
                scatter_actual_vs_pred(dname, df_dp, label_col, preds, models_for_scatter, plot_dir)

        if dname.startswith("PE7"):
            # Select models per target
            if dname == "PE7_DN":
                allowed = ["DP_base", "DP_DLD1", "DP_A549", "DP7_DN", "DP7_AN"]
            elif dname == "PE7_DC":
                allowed = ["DP_base", "DP_DLD1", "DP_A549", "DP7_DC"]
            elif dname == "PE7_DT":
                allowed = ["DP_base", "DP_DLD1", "DP_A549", "DP7_DT"]
            else:  # PE7_AN or others -> skip length analysis
                allowed = []

            if allowed:
                ins_mask = df_dp.get("type_ins", 0) == 1
                del_mask = df_dp.get("type_del", 0) == 1
                if ins_mask.any():
                    lengths_ins = compute_length_curves(df_dp, preds, label_col, ins_mask)
                    length_curves[f"{dname}_ins"] = lengths_ins
                    line_plot_lengths(plot_dir, f"{dname}_insertion", lengths_ins, allowed, metric="spearman")
                    line_plot_lengths(plot_dir, f"{dname}_insertion", lengths_ins, allowed, metric="mae")
                if del_mask.any():
                    lengths_del = compute_length_curves(df_dp, preds, label_col, del_mask)
                    length_curves[f"{dname}_del"] = lengths_del
                    line_plot_lengths(plot_dir, f"{dname}_deletion", lengths_del, allowed, metric="spearman")
                    line_plot_lengths(plot_dir, f"{dname}_deletion", lengths_del, allowed, metric="mae")

    def save_heatmap(records: List[dict], fname: str, title: str, n_map: Dict[str, int], plot_dir: Path):
        if not records:
            return
        order = [
            "DP_base",
            "DP_DLD1",
            "DP_A549",
            "PE7_DN",
            "PE7_DC",
            "PE7_DT",
            "PE7_AN",
            "PRIDICT_HEK",
            "PRIDICT_K562",
        ]
        df_raw = pd.DataFrame(records).set_index("dataset")
        cols = [m for m in MODEL_NAMES if m in df_raw.columns]
        df = df_raw[cols].reindex(order)
        plt.figure(figsize=(8, 8))
        display_cols = [MODEL_DISPLAY[m] for m in df.columns]
        display_rows = []
        for ds in df.index:
            n_val = n_map.get(ds, np.nan)
            label = DATASET_DISPLAY.get(ds, ds)
            if not np.isnan(n_val):
                label = f"{label} (n={n_val})"
            display_rows.append(label)
        sns.heatmap(
            df.values,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=display_cols,
            yticklabels=display_rows,
            cbar=False,
        )
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        sns.despine(left=False, bottom=False, ax=plt.gca())
        plt.tight_layout()
        plt.savefig(plot_dir / fname, dpi=200)
        plt.close()

    save_heatmap(overall_heat, "heatmap_overall.png", "Spearman (overall)", n_map, plot_dir)

    log_path = out_dir / "eval_grid_log.txt"
    log_path.write_text("\n".join(logs))
    print(f"Wrote logs to {log_path}")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
