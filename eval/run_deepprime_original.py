"""
Run the original DeepPrime ensembles on PE7's processed_test.csv.

Steps
- Convert PE7 column names to the DeepPrime feature schema.
- Normalize with DeepPrime mean/std.
- Run two ensembles:
  * DeepPrime base (5 models under DeepPrime/models/ontarget/final)
  * DeepPrime DLD1 variant (20 models under DeepPrime/models/ontarget_variants/DP_variant_DLD1_PE2max_Opti_221114)
- Optionally evaluate against available PE7 measurement columns (DN/DC/DT/AN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Make DeepPrime utils importable
REPO_ROOT = Path(__file__).resolve().parent.parent
DEEPPRIME_ROOT = REPO_ROOT.parent / "DeepPrime"
sys.path.insert(0, str(DEEPPRIME_ROOT))
from utils.data import seq_concat  # type: ignore  # noqa: E402
from utils.model import GeneInteractionModel  # type: ignore  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from model import DeepPrime7  # type: ignore  # noqa: E402


COLUMN_MAP: Dict[str, str] = {
    "PBS_len": "PBSlen",
    "RTT_len": "RTlen",
    "RT-PBS_len": "RT-PBSlen",
    "Edit_pos": "Edit_pos",
    "Edit_len": "Edit_len",
    "RHA_len_nn": "RHA_len",
    "type_sub": "type_sub",
    "type_ins": "type_ins",
    "type_del": "type_del",
    "Tm1_PBS": "Tm1",
    "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4",
    "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1",
    "GC_count_RTT": "nGCcnt2",
    "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1",
    "GC_contents_RTT": "fGCcont2",
    "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3",
    "MFE_Spacer": "MFE4",
    "DeepSpCas9_score": "DeepSpCas9_score",
}

TARGET_COLUMNS = ["DN_M_EF", "DC_M_EF", "DT_M_EF", "AN_M_EF"]

DP7_SELECTED_FEATURES = [
    "Tm1_PBS",
    "Tm2_RTT_cTarget_sameLength",
    "Tm3_RTT_cTarget_replaced",
    "Tm4_cDNA_PAM-oppositeTarget",
    "Tm5_RTT_cDNA",
    "GC_contents_PBS",
    "GC_contents_RTT",
    "MFE_RT-PBS-polyT",
    "MFE_Spacer",
]

DP7_PARAMS = {
    "conv1_out": 64,
    "conv2_out": 72,
    "conv3_out": 72,
    "dropout1": 0.25,
    "dropout2": 0.03,
    "dropout3": 0.25,
    "fc1_hidden": 512,
    "fc2_hidden": 192,
    "fc_dropout": 0.15,
    "num_heads": 4,
    "use_attention": True,
}

DP7_FEATURE_MAP_FROM_DP = {
    "Tm1_PBS": "Tm1",
    "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4",
    "GC_contents_PBS": "fGCcont1",
    "GC_contents_RTT": "fGCcont2",
    "MFE_RT-PBS-polyT": "MFE3",
    "MFE_Spacer": "MFE4",
}


def _load_pe7(
    pe7_path: Path, source_path: Path, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PE7 processed test set and merge with the raw table to recover missing features."""
    pe7 = pd.read_csv(pe7_path)
    source = pd.read_csv(source_path)

    if verbose:
        print(f"Loaded PE7 test: {pe7.shape}, raw: {source.shape}")

    if "idx" not in pe7.columns:
        raise KeyError("processed_test.csv must contain an 'idx' column for merging.")

    pe7_idx = pe7.set_index("idx")
    source_idx = source.set_index("idx")

    missing_cols = [c for c in COLUMN_MAP if c not in pe7_idx.columns and c not in source_idx.columns]
    if missing_cols:
        raise KeyError(f"Columns missing from both processed_test and source CSV: {missing_cols}")

    pull_from_source = [c for c in COLUMN_MAP if c not in pe7_idx.columns]
    seq_from_source = [c for c in ["Target", "Masked_EditSeq"] if c not in pe7_idx.columns]

    missing_in_source = [c for c in pull_from_source + seq_from_source if c not in source_idx.columns]
    if missing_in_source:
        raise KeyError(f"Columns missing in source CSV: {missing_in_source}")

    merged = pe7_idx.copy()
    for col in seq_from_source + pull_from_source:
        merged[col] = source_idx.loc[merged.index, col]

    merged.reset_index(inplace=True)

    required_cols = set(COLUMN_MAP.keys()) | {"Target", "Masked_EditSeq"}
    missing_mask = merged[list(required_cols)].isna().any(axis=1)
    if missing_mask.any():
        missing_idx = merged.loc[missing_mask, "idx"].tolist()
        if verbose:
            print(f"Dropping {len(missing_idx)} rows with missing features (idx: {missing_idx})")
        merged = merged.loc[~missing_mask].reset_index(drop=True)
        pe7 = pe7.loc[pe7["idx"].isin(merged["idx"])].reset_index(drop=True)

    return pe7, merged


def convert_to_deepprime_schema(merged: pd.DataFrame) -> pd.DataFrame:
    """Return a DeepPrime-shaped dataframe (column names + order)."""
    converted = pd.DataFrame()
    converted["WT74_On"] = merged["Target"]
    converted["Edited74_On"] = merged["Masked_EditSeq"]
    converted["Fold"] = merged.get("Split", "Test")

    for src, dst in COLUMN_MAP.items():
        converted[dst] = merged[src]

    if converted.isna().any().any():
        missing_cols = [c for c in converted.columns if converted[c].isna().any()]
        raise ValueError(f"Converted DeepPrime frame has NaNs in columns: {missing_cols}")

    # DeepPrime training target column placeholder (not used for inference)
    converted["Measured_PE_efficiency"] = np.nan
    return converted


def _load_normalization() -> Tuple[pd.Series, pd.Series]:
    mean = pd.read_csv(DEEPPRIME_ROOT / "data/mean.csv", header=None, index_col=0).iloc[:, 0]
    std = pd.read_csv(DEEPPRIME_ROOT / "data/std.csv", header=None, index_col=0).iloc[:, 0]
    return mean, std


def _tensorize_inputs(df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = _load_normalization()

    features = df.loc[:, mean.index]
    x = (features - mean) / std
    x_tensor = torch.tensor(x.to_numpy(), dtype=torch.float32, device=device)

    g = seq_concat(df)  # expects WT74_On / Edited74_On
    g_tensor = torch.tensor(g, dtype=torch.float32, device=device).permute((0, 3, 1, 2))
    return g_tensor, x_tensor


def ensemble_predict(
    model_paths: Iterable[Path],
    g: torch.Tensor,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int | None = 1024,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    n = g.shape[0]
    for path in model_paths:
        model = GeneInteractionModel(hidden_size=128, num_layers=1).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            if batch_size:
                chunk_preds = []
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    pred = model(g[start:end], x[start:end]).detach().cpu().numpy()
                    chunk_preds.append(pred)
                pred = np.concatenate(chunk_preds, axis=0)
            else:
                pred = model(g, x).detach().cpu().numpy()
        preds.append(pred)

    preds_arr = np.squeeze(np.array(preds))  # (n_models, n_samples)
    mean_pred = np.mean(preds_arr, axis=0)
    return np.exp(mean_pred) - 1  # invert log1p used during training


def evaluate_predictions(pred: np.ndarray, df: pd.DataFrame, subset_mask: np.ndarray | None = None) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    base_mask = np.ones(len(df), dtype=bool) if subset_mask is None else subset_mask
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            continue
        mask = base_mask & df[col].notna().to_numpy()
        if mask.sum() < 2:
            continue
        y_true = df[col].to_numpy()[mask]
        y_pred = pred[mask]
        metrics[col] = {
            "spearman": stats.spearmanr(y_true, y_pred).correlation,
            "pearson": stats.pearsonr(y_true, y_pred)[0],
            "mae": float(np.mean(np.abs(y_true - y_pred))),
        }
    return metrics


def evaluate_predictions_multi(pred_map: Dict[str, np.ndarray], df: pd.DataFrame, subset_mask: np.ndarray | None = None) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    base_mask = np.ones(len(df), dtype=bool) if subset_mask is None else subset_mask
    for col, pred in pred_map.items():
        if col not in df.columns:
            continue
        mask = base_mask & df[col].notna().to_numpy()
        if mask.sum() < 2:
            continue
        y_true = df[col].to_numpy()[mask]
        y_pred = pred[mask]
        metrics[col] = {
            "spearman": stats.spearmanr(y_true, y_pred).correlation,
            "pearson": stats.pearsonr(y_true, y_pred)[0],
            "mae": float(np.mean(np.abs(y_true - y_pred))),
        }
    return metrics


def _generate_synthetic_image(dna_seq: str, target_length: int = 128) -> np.ndarray:
    img = np.zeros((target_length, 4, 1), dtype=np.float32)
    for i, base in enumerate(str(dna_seq)):
        if i >= target_length:
            break
        if base == "A":
            img[i, 0, 0] = 1.0
        elif base == "G":
            img[i, 1, 0] = 1.0
        elif base == "T":
            img[i, 2, 0] = 1.0
        elif base == "C":
            img[i, 3, 0] = 1.0
    return img


def deepprime7_predict(
    df: pd.DataFrame,
    device: torch.device,
    model_path: Path,
    scaler_path: Path,
    batch_size: int | None = 1024,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    import joblib

    scaler = joblib.load(scaler_path)

    model = DeepPrime7(use_additional_features=DP7_SELECTED_FEATURES, **DP7_PARAMS).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Ensure required columns
    missing = [c for c in DP7_SELECTED_FEATURES + ["Target", "Masked_EditSeq"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for DeepPrime7 inference: {missing}")

    # Scale selected features
    feats_input = df[DP7_SELECTED_FEATURES].copy()
    if feats_input.isna().any().any():
        # Fill NaNs with scaler mean as a simple imputation to keep transform stable.
        fill_values = {col: scaler.mean_[i] for i, col in enumerate(DP7_SELECTED_FEATURES)}
        feats_input = feats_input.fillna(value=fill_values)
    feats_scaled = scaler.transform(feats_input)
    feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32)

    # Build images
    imgs = []
    for t, m in zip(df["Target"], df["Masked_EditSeq"]):
        img1 = _generate_synthetic_image(t)
        img2 = _generate_synthetic_image(m)
        combined = np.concatenate((img1, img2), axis=2)  # (L,4,2)
        imgs.append(torch.tensor(combined, dtype=torch.float32).permute(1, 0, 2))  # (4,L,2)

    img_tensor = torch.stack(imgs)
    n = img_tensor.shape[0]
    batch_preds = []
    with torch.no_grad():
        if batch_size:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                img_batch = img_tensor[start:end].to(device)
                feat_batch = feats_tensor[start:end].to(device)
                outputs = model(img_batch, feat_batch)
                batch_preds.append(torch.stack(outputs, dim=1).squeeze(-1).cpu())
        else:
            outputs = model(img_tensor.to(device), feats_tensor.to(device))
            batch_preds.append(torch.stack(outputs, dim=1).squeeze(-1).cpu())
    preds = torch.cat(batch_preds, dim=0).numpy()  # (N,4)

    pred_map = {
        "DN_M_EF": preds[:, 0] * 5.0,
        "DC_M_EF": preds[:, 1] * 5.0,
        "DT_M_EF": preds[:, 2] * 5.0,
        "AN_M_EF": preds[:, 3] * 2.0,
    }

    out_df = df.copy()
    out_df["DP7_DN_pred"] = pred_map["DN_M_EF"]
    out_df["DP7_DC_pred"] = pred_map["DC_M_EF"]
    out_df["DP7_DT_pred"] = pred_map["DT_M_EF"]
    out_df["DP7_AN_pred"] = pred_map["AN_M_EF"]

    return out_df, pred_map


def _dp_dataset_to_dp7(df: pd.DataFrame) -> pd.DataFrame:
    """Map DeepPrime-format dataframe into DeepPrime7 expected columns."""
    mapped = pd.DataFrame()
    mapped["Target"] = df["WT74_On"]
    mapped["Masked_EditSeq"] = df["Edited74_On"]
    for new, old in DP7_FEATURE_MAP_FROM_DP.items():
        mapped[new] = df[old] if old in df.columns else np.nan
    return mapped


def evaluate_single_target(
    df: pd.DataFrame,
    device: torch.device,
    base_models: List[Path],
    dld1_models: List[Path],
    a549_models: List[Path],
    dp7_path: Path,
    dp7_scaler: Path,
    batch_size: int | None = 1024,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Prepare tensors for original DeepPrime models
    g_tensor, x_tensor = _tensorize_inputs(df, device)

    base_pred = ensemble_predict(base_models, g_tensor, x_tensor, device, batch_size=batch_size)
    dld1_pred = ensemble_predict(dld1_models, g_tensor, x_tensor, device, batch_size=batch_size)
    a549_pred = ensemble_predict(a549_models, g_tensor, x_tensor, device, batch_size=batch_size)

    dp7_input = _dp_dataset_to_dp7(df)
    dp7_df, dp7_pred_map = deepprime7_predict(
        dp7_input, device, dp7_path, dp7_scaler, batch_size=batch_size
    )
    dp7_pred = dp7_pred_map["DN_M_EF"]  # use head0 for single-target comparison

    out_df = df.copy()
    out_df["Pred_base"] = base_pred
    out_df["Pred_dld1"] = dld1_pred
    out_df["Pred_a549"] = a549_pred
    out_df["Pred_dp7_DN"] = dp7_pred

    y_true = df["Measured_PE_efficiency"].to_numpy()

    def sp(pred):
        return stats.spearmanr(y_true, pred).correlation

    metrics = {
        "DeepPrime_base": sp(base_pred),
        "DeepPrime_DLD1": sp(dld1_pred),
        "DeepPrime_A549": sp(a549_pred),
        "DeepPrime7_DN": sp(dp7_pred),
    }

    return out_df, metrics


def length_analysis(df: pd.DataFrame, preds: Dict[str, np.ndarray], max_len: int = 20) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return Spearman per edit length for insertions and deletions."""
    results: Dict[str, Dict[int, Dict[str, float]]] = {"insertion": {}, "deletion": {}}
    y = df["Measured_PE_efficiency"]

    for typ, mask_col in [("insertion", "type_ins"), ("deletion", "type_del")]:
        type_mask = df[mask_col] == 1
        for L in range(1, max_len + 1):
            m = type_mask & (df["Edit_len"] == L)
            if m.sum() < 3:
                continue
            y_true = y[m]
            results[typ][L] = {}
            for name, pred in preds.items():
                corr = stats.spearmanr(y_true, pred[m]).correlation
                results[typ][L][name] = corr
    return results


def main():
    parser = argparse.ArgumentParser(description="Run original DeepPrime models on PE7 processed_test.csv")
    parser.add_argument("--pe7-path", default=REPO_ROOT / "data/processed_test.csv", type=Path)
    parser.add_argument("--source-path", default=REPO_ROOT / "data/250716_24k_final.csv", type=Path)
    parser.add_argument("--out-csv", default=REPO_ROOT / "results/deepprime_original_predictions.csv", type=Path)
    parser.add_argument("--log-path", default=REPO_ROOT / "results/deepprime_original_log.txt", type=Path)
    parser.add_argument("--dp7-path", default=REPO_ROOT / "models/deepprime7.pth", type=Path)
    parser.add_argument("--dp7-scaler", default=REPO_ROOT / "models/deepprime7_scaler.pth", type=Path)
    parser.add_argument("--cross-eval-dir", default=REPO_ROOT / "eval_cross_models", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)

    pe7_df, merged = _load_pe7(args.pe7_path, args.source_path)
    deepprime_df = convert_to_deepprime_schema(merged)
    g_tensor, x_tensor = _tensorize_inputs(deepprime_df, device)

    base_models = sorted((DEEPPRIME_ROOT / "models/ontarget/final").glob("model_*.pt"))
    variant_models = sorted(
        (DEEPPRIME_ROOT / "models/ontarget_variants/DP_variant_DLD1_PE2max_Opti_221114").glob("final_model_*.pt")
    )
    a549_models = sorted(
        (DEEPPRIME_ROOT / "models/ontarget_variants/DP_variant_A549_PE2max_Opti_221114").glob("final_model_*.pt")
    )

    base_pred = ensemble_predict(base_models, g_tensor, x_tensor, device, batch_size=args.batch_size)
    variant_pred = ensemble_predict(variant_models, g_tensor, x_tensor, device, batch_size=args.batch_size)
    a549_pred = ensemble_predict(a549_models, g_tensor, x_tensor, device, batch_size=args.batch_size)
    dp7_df, dp7_pred_map = deepprime7_predict(
        pe7_df, device, args.dp7_path, args.dp7_scaler, batch_size=args.batch_size
    )

    out = pe7_df.copy()
    out["DeepPrime_base_pred"] = base_pred
    out["DeepPrime_variant_pred"] = variant_pred
    out["DeepPrime_a549_pred"] = a549_pred
    out["DeepPrime7_DN_pred"] = dp7_pred_map["DN_M_EF"]
    out["DeepPrime7_DC_pred"] = dp7_pred_map["DC_M_EF"]
    out["DeepPrime7_DT_pred"] = dp7_pred_map["DT_M_EF"]
    out["DeepPrime7_AN_pred"] = dp7_pred_map["AN_M_EF"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    log_lines: List[str] = []
    log_lines.append(f"Saved predictions to {args.out_csv}")

    def append_metrics(name: str, pred: np.ndarray):
        log_lines.append(f"{name} ensemble metrics (Spearman/Pearson/MAE, NaNs removed per target):")
        for col, vals in evaluate_predictions(pred, pe7_df).items():
            log_lines.append(f"  {col}: {vals}")

    append_metrics("Base", base_pred)
    append_metrics("DLD1 variant", variant_pred)
    append_metrics("A549 variant", a549_pred)

    # PAM stratification
    pam = pe7_df["Target"].str.slice(24, 27)
    pam_masks = {
        "NGG": (pam == "GGG") | (pam == "AGG") | (pam == "TGG") | (pam == "CGG"),
        "Non-NGG": ~(pam == "GGG") & ~(pam == "AGG") & ~(pam == "TGG") & ~(pam == "CGG"),
    }

    log_lines.append("DeepPrime7 metrics (Spearman/Pearson/MAE, NaNs removed per target):")
    for col, vals in evaluate_predictions_multi(dp7_pred_map, pe7_df).items():
        log_lines.append(f"  {col}: {vals}")

    for label, mask in pam_masks.items():
        log_lines.append(f"PAM subset: {label} (n={int(mask.sum())})")
        log_lines.append("  Base:")
        for col, vals in evaluate_predictions(base_pred, pe7_df, mask.to_numpy()).items():
            log_lines.append(f"    {col}: {vals}")
        log_lines.append("  DLD1 variant:")
        for col, vals in evaluate_predictions(variant_pred, pe7_df, mask.to_numpy()).items():
            log_lines.append(f"    {col}: {vals}")
        log_lines.append("  A549 variant:")
        for col, vals in evaluate_predictions(a549_pred, pe7_df, mask.to_numpy()).items():
            log_lines.append(f"    {col}: {vals}")
        log_lines.append("  DeepPrime7:")
        for col, vals in evaluate_predictions_multi(dp7_pred_map, pe7_df, mask.to_numpy()).items():
            log_lines.append(f"    {col}: {vals}")

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("\n".join(log_lines))

    for line in log_lines:
        print(line)

    # Cross-evaluate on original DeepPrime datasets if requested paths exist.
    datasets = [
        ("DeepPrime_base", DEEPPRIME_ROOT / "data/DeepPrime_dataset_final_Feat8.csv"),
        ("Variant_A549", DEEPPRIME_ROOT / "data/DP_variant_A549_PE2max_Opti_221114.csv"),
        ("Variant_DLD1", DEEPPRIME_ROOT / "data/DP_variant_DLD1_PE2max_Opti_221114.csv"),
    ]

    heatmap_records = []
    length_logs: List[str] = []
    args.cross_eval_dir.mkdir(parents=True, exist_ok=True)

    for name, path in datasets:
        if not path.exists():
            length_logs.append(f"[{name}] skipped (missing file {path})")
            continue
        df_full = pd.read_csv(path)
        df_test = df_full[df_full["Fold"] == "Test"].reset_index(drop=True)
        if df_test.empty:
            length_logs.append(f"[{name}] skipped (no Fold == 'Test' rows)")
            continue

        out_df, metrics = evaluate_single_target(
            df_test,
            device,
            base_models,
            variant_models,
            a549_models,
            args.dp7_path,
            args.dp7_scaler,
        )

        out_path = args.cross_eval_dir / f"{name}_predictions.csv"
        out_df.to_csv(out_path, index=False)
        length_logs.append(f"[{name}] saved predictions: {out_path}")
        length_logs.append(f"[{name}] Spearman: {metrics}")

        preds_for_length = {
            "DeepPrime_base": out_df["Pred_base"].to_numpy(),
            "DeepPrime_DLD1": out_df["Pred_dld1"].to_numpy(),
            "DeepPrime_A549": out_df["Pred_a549"].to_numpy(),
            "DeepPrime7_DN": out_df["Pred_dp7_DN"].to_numpy(),
        }
        len_res = length_analysis(out_df, preds_for_length)
        for typ, lens in len_res.items():
            for L, vals in lens.items():
                length_logs.append(f"[{name}] {typ} len={L}: {vals}")

        for model_name, score in metrics.items():
            heatmap_records.append({"dataset": name, "model": model_name, "spearman": score})

    if heatmap_records:
        heat_df = pd.DataFrame(heatmap_records).pivot(index="dataset", columns="model", values="spearman")
        plt.figure(figsize=(8, 4))
        sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
        plt.title("Spearman correlation (Fold == Test)")
        heatmap_path = args.cross_eval_dir / "spearman_heatmap.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        length_logs.append(f"Saved heatmap: {heatmap_path}")

    cross_log_path = args.cross_eval_dir / "cross_eval_log.txt"
    cross_log_path.write_text("\n".join(length_logs))
    for line in length_logs:
        print(line)


if __name__ == "__main__":
    main()
