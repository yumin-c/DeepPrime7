"""
Generate prediction tables for eval_grid plots.

Outputs are written under eval_grid/preds/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import run_deepprime_original as rdo

REPO_ROOT = Path(__file__).resolve().parent.parent
DEEPPRIME_ROOT = REPO_ROOT.parent / "DeepPrime"
OUT_DIR_DEFAULT = REPO_ROOT / "eval_grid"

DATASETS = {
    "DP_base": DEEPPRIME_ROOT / "data" / "DeepPrime_dataset_final_Feat8.csv",
    "DP_A549": DEEPPRIME_ROOT / "data" / "DP_variant_A549_PE2max_Opti_221114.csv",
    "DP_DLD1": DEEPPRIME_ROOT / "data" / "DP_variant_DLD1_PE2max_Opti_221114.csv",
    "DP_DLD1_NRCHPE4max": DEEPPRIME_ROOT / "data" / "DP_variant_DLD1_NRCHPE4max_Opti_220728.csv",
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
    "DP_DLD1_NRCHPE4max",
    "DP_A549",
    "DP7_DN",
    "DP7_DC",
    "DP7_DT",
    "DP7_AN",
    "PRIDICT2_HEK",
    "PRIDICT2_K562",
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


def evaluate_dataset(
    df_dp: pd.DataFrame,
    label_col: str,
    device: torch.device,
    base_models: List[Path],
    dld1_models: List[Path],
    dld1_nrch_models: List[Path],
    a549_models: List[Path],
    dp7_path: Path,
    dp7_scaler: Path,
    batch_size: int | None = 1024,
) -> Dict[str, np.ndarray]:
    """Return dict: model -> predictions."""
    g_tensor, x_tensor = rdo._tensorize_inputs(df_dp, device)
    pred_dp_base = rdo.ensemble_predict(base_models, g_tensor, x_tensor, device, batch_size=batch_size)
    pred_dp_dld1 = rdo.ensemble_predict(dld1_models, g_tensor, x_tensor, device, batch_size=batch_size)
    pred_dp_dld1_nrch = rdo.ensemble_predict(dld1_nrch_models, g_tensor, x_tensor, device, batch_size=batch_size)
    pred_dp_a549 = rdo.ensemble_predict(a549_models, g_tensor, x_tensor, device, batch_size=batch_size)

    dp7_input = rdo._dp_dataset_to_dp7(df_dp)
    _, dp7_pred_map = rdo.deepprime7_predict(
        dp7_input, device, dp7_path, dp7_scaler, batch_size=batch_size
    )

    return {
        "DP_base": pred_dp_base,
        "DP_DLD1": pred_dp_dld1,
        "DP_DLD1_NRCHPE4max": pred_dp_dld1_nrch,
        "DP_A549": pred_dp_a549,
        "DP7_DN": dp7_pred_map["DN_M_EF"],
        "DP7_DC": dp7_pred_map["DC_M_EF"],
        "DP7_DT": dp7_pred_map["DT_M_EF"],
        "DP7_AN": dp7_pred_map["AN_M_EF"],
    }


def save_preds(
    out_dir: Path,
    dataset_name: str,
    df_dp: pd.DataFrame,
    label_col: str,
    preds: Dict[str, np.ndarray],
) -> int:
    base_cols = []
    for col in ["idx", "WT74_On", "Target", "Edit_len", "type_ins", "type_del"]:
        if col in df_dp.columns:
            base_cols.append(col)
    out = df_dp[base_cols].copy() if base_cols else pd.DataFrame(index=df_dp.index)
    out["y_true"] = df_dp[label_col].to_numpy()
    for model_name, vec in preds.items():
        out[model_name] = vec

    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_dir / f"{dataset_name}_preds.csv", index=False)
    return int((~pd.isna(out["y_true"])).sum())


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for eval_grid plots.")
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

    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_models = sorted((DEEPPRIME_ROOT / "models" / "ontarget" / "final").glob("model_*.pt"))
    dld1_models = sorted(
        (DEEPPRIME_ROOT / "models" / "ontarget_variants" / "DP_variant_DLD1_PE2max_Opti_221114").glob(
            "final_model_*.pt"
        )
    )
    dld1_nrch_models = sorted(
        (DEEPPRIME_ROOT / "models" / "ontarget_variants" / "DP_variant_DLD1_NRCHPE4max_Opti_220728").glob(
            "final_model_*.pt"
        )
    )
    a549_models = sorted(
        (DEEPPRIME_ROOT / "models" / "ontarget_variants" / "DP_variant_A549_PE2max_Opti_221114").glob(
            "final_model_*.pt"
        )
    )

    dp_datasets = {
        "DP_base": (load_dp_dataset(DATASETS["DP_base"]), "Measured_PE_efficiency"),
        "DP_A549": (load_dp_dataset(DATASETS["DP_A549"]), "Measured_PE_efficiency"),
        "DP_DLD1": (load_dp_dataset(DATASETS["DP_DLD1"]), "Measured_PE_efficiency"),
        "DP_DLD1_NRCHPE4max": (load_dp_dataset(DATASETS["DP_DLD1_NRCHPE4max"]), "Measured_PE_efficiency"),
    }

    dp_pe7 = prepare_pe7_deepprime(args.pe7_path, args.pe7_source)
    for name, col in PE7_LABEL_MAP.items():
        df = dp_pe7.copy()
        df[col] = df[col]
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

    meta_rows = []
    for dname, (df_dp, label_col) in dp_datasets.items():
        preds = evaluate_dataset(
            df_dp,
            label_col,
            device,
            base_models,
            dld1_models,
            dld1_nrch_models,
            a549_models,
            args.dp7_path,
            args.dp7_scaler,
            batch_size=args.batch_size,
        )

        if pridict_map is not None and dname.startswith("PE7"):
            if "idx" not in df_dp.columns:
                raise KeyError("PE7 datasets must include idx for PRIDICT2 merge.")
            idx_series = df_dp["idx"]
            preds["PRIDICT2_HEK"] = idx_series.map(pridict_map["PRIDICT2_HEK"]).to_numpy()
            preds["PRIDICT2_K562"] = idx_series.map(pridict_map["PRIDICT2_K562"]).to_numpy()

        if dname in {"PRIDICT_HEK", "PRIDICT_K562"}:
            if "PRIDICT2_HEK" in df_dp.columns:
                preds["PRIDICT2_HEK"] = df_dp["PRIDICT2_HEK"].to_numpy()
            if "PRIDICT2_K562" in df_dp.columns:
                preds["PRIDICT2_K562"] = df_dp["PRIDICT2_K562"].to_numpy()

        n = save_preds(out_dir, dname, df_dp, label_col, preds)
        seq_col = "WT74_On" if "WT74_On" in df_dp.columns else "Target" if "Target" in df_dp.columns else ""
        meta_rows.append({"dataset": dname, "label_col": label_col, "seq_col": seq_col, "n": n})

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "preds" / "metadata.csv", index=False)
    print(f"Saved prediction tables to {out_dir / 'preds'}")


if __name__ == "__main__":
    main()
