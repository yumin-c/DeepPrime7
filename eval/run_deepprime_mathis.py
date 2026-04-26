import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import run_deepprime_original as rdo


def main():
    parser = argparse.ArgumentParser(description="Run DeepPrime/DeepPrime7 on Mathis 2024 inputs.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--dp7-path", default=rdo.REPO_ROOT / "models/deepprime7.pth", type=Path)
    parser.add_argument("--dp7-scaler", default=rdo.REPO_ROOT / "models/deepprime7_scaler.pth", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.input_csv)

    required = set(rdo.COLUMN_MAP.keys()) | {"Target", "Masked_EditSeq"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    deepprime_df = rdo.convert_to_deepprime_schema(df)
    g_tensor, x_tensor = rdo._tensorize_inputs(deepprime_df, device)

    base_models = sorted((rdo.DEEPPRIME_ROOT / "models/ontarget/final").glob("model_*.pt"))
    variant_models = sorted(
        (rdo.DEEPPRIME_ROOT / "models/ontarget_variants/DP_variant_DLD1_PE2max_Opti_221114").glob("final_model_*.pt")
    )
    a549_models = sorted(
        (rdo.DEEPPRIME_ROOT / "models/ontarget_variants/DP_variant_A549_PE2max_Opti_221114").glob("final_model_*.pt")
    )

    base_pred = rdo.ensemble_predict(base_models, g_tensor, x_tensor, device, batch_size=args.batch_size)
    variant_pred = rdo.ensemble_predict(variant_models, g_tensor, x_tensor, device, batch_size=args.batch_size)
    a549_pred = rdo.ensemble_predict(a549_models, g_tensor, x_tensor, device, batch_size=args.batch_size)

    dp7_df, dp7_pred_map = rdo.deepprime7_predict(
        df, device, args.dp7_path, args.dp7_scaler, batch_size=args.batch_size
    )

    out = df.copy()
    out["DeepPrime_base_pred"] = base_pred
    out["DeepPrime_variant_pred"] = variant_pred
    out["DeepPrime_a549_pred"] = a549_pred
    out["DeepPrime7_DN_pred"] = dp7_pred_map["DN_M_EF"]
    out["DeepPrime7_DC_pred"] = dp7_pred_map["DC_M_EF"]
    out["DeepPrime7_DT_pred"] = dp7_pred_map["DT_M_EF"]
    out["DeepPrime7_AN_pred"] = dp7_pred_map["AN_M_EF"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
