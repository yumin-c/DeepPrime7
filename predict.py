# predict.py
import argparse
import glob
import os
import sys
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_scaler(scaler_path: str):
    try:
        import joblib
        return joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {scaler_path}. Tried torch.load and joblib.load. Error: {e}")

from model import DeepPrime7

CONFIG = {
    "seed": 216,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "training": {
        "batch_size": 512,
        "num_workers": 4,
    },
    "model": {
        "params": {
            'conv1_out': 64, 'conv2_out': 72, 'conv3_out': 72,
            'dropout1': 0.25, 'dropout2': 0.03, 'dropout3': 0.25,
            'fc1_hidden': 512, 'fc2_hidden': 192, 'fc_dropout': 0.15,
            'num_heads': 4,
        },
        "load_path": "models/deepprime7_add6k.pth",
    },
    "ablation": {
        "use_features": {
            'tm_features': True,
            'gc_count_features': False,
            'gc_contents_features': True,
            'mfe_features': True,
            'deepspcas9_score': False,
        },
        "use_attention": True,
    }
}

FEATURE_GROUPS = {
    'tm_features': ['Tm1_PBS', 'Tm2_RTT_cTarget_sameLength', 'Tm3_RTT_cTarget_replaced',
                    'Tm4_cDNA_PAM-oppositeTarget', 'Tm5_RTT_cDNA'],
    'gc_count_features': ['GC_count_PBS', 'GC_count_RTT'],
    'gc_contents_features': ['GC_contents_PBS', 'GC_contents_RTT'],
    'mfe_features': ['MFE_RT-PBS-polyT', 'MFE_Spacer'],
    'deepspcas9_score': ['DeepSpCas9_score']
}

ESSENTIAL_SEQS = ['Target', 'Masked_EditSeq']

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_synthetic_image(dna_seq: str, target_length: int = 128):
    synthetic_image = np.zeros((target_length, 4, 1), dtype=np.float32)
    try:
        for i, base in enumerate(dna_seq):
            if i >= target_length:
                break
            if base == 'A': synthetic_image[i, 0, 0] = 1.0
            elif base == 'G': synthetic_image[i, 1, 0] = 1.0
            elif base == 'T': synthetic_image[i, 2, 0] = 1.0
            elif base == 'C': synthetic_image[i, 3, 0] = 1.0
    except Exception:
        pass
    return synthetic_image

class PE7InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, selected_features: Optional[List[str]]):
        self.df = df.reset_index(drop=True)
        self.selected_features = selected_features or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image1 = generate_synthetic_image(str(row['Target']))
        image2 = generate_synthetic_image(str(row['Masked_EditSeq']))
        combined_image = np.concatenate((image1, image2), axis=2)  # (L, 4, 2)
        combined_image = torch.tensor(combined_image, dtype=torch.float32).permute(1, 0, 2)  # (4, L, 2)

        if self.selected_features:
            feats = torch.tensor(row[self.selected_features].astype(np.float32).values)
            return combined_image, feats
        return combined_image, None

def custom_inference_collate_fn(batch):
    images, features = zip(*batch)
    images = torch.stack(images)
    features = torch.stack([f for f in features if f is not None]) if features[0] is not None else None
    return images, features

def collect_selected_features(cfg) -> List[str]:
    selected = []
    use_map = cfg["ablation"]["use_features"]
    for group, include in use_map.items():
        if include:
            selected.extend(FEATURE_GROUPS[group])
    return selected

def read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    elif lower.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .csv or .parquet")

def save_predictions(df: pd.DataFrame, out_dir: str, src_path: str):
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{base}_pred.csv")
    df.to_csv(out_path, index=False)
    return out_path

def ensure_features(df: pd.DataFrame, needed_cols: List[str]) -> List[str]:
    missing = [c for c in needed_cols if c not in df.columns]
    return missing

def run_inference_on_df(df: pd.DataFrame, model: torch.nn.Module, device: torch.device,
                        selected_features: List[str], scaler, batch_size: int, num_workers: int) -> pd.DataFrame:
    # Drop NA in essential seqs
    init = len(df)
    df = df.dropna(subset=ESSENTIAL_SEQS).copy()
    if len(df) != init:
        print(f" - Dropped {init - len(df)} rows with missing essential sequence columns.")

    # Apply scaler only to selected features
    if selected_features:
        # Reindex to ensure column order matches scaler fit
        df[selected_features] = scaler.transform(df[selected_features])

    ds = PE7InferenceDataset(df, selected_features)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_inference_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for images, feats in dl:
            images = images.to(device, non_blocking=True)
            if feats is not None:
                feats = feats.to(device, non_blocking=True)
            outputs = model(images, feats)  # list of 4 heads
            out_stack = torch.stack(outputs, dim=1)  # (B, 4, ...)
            preds.append(out_stack.cpu().numpy())

    if preds:
        P = np.concatenate(preds, axis=0)  # (N, 4)
        df['DN_EF_Prediction'] = P[:, 0] * 5.0
        df['DC_EF_Prediction'] = P[:, 1] * 5.0
        df['DT_EF_Prediction'] = P[:, 2] * 5.0
        df['AN_EF_Prediction'] = P[:, 3] * 2.0
    else:
        for col in ['DN_EF_Prediction', 'DC_EF_Prediction', 'DT_EF_Prediction', 'AN_EF_Prediction']:
            df[col] = []

    return df

def main():
    parser = argparse.ArgumentParser(description="Batch prediction for .csv/.parquet files in a folder.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder containing input .csv or .parquet files to predict.")
    parser.add_argument("--model_path", type=str, default=CONFIG["model"]["load_path"],
                        help="Path to trained model weights (.pth).")
    parser.add_argument("--scaler_path", type=str, default="models/deepprime7_scaler.pth",
                        help="Path to fitted StandardScaler object (.pth/.pkl).")
    parser.add_argument("--batch_size", type=int, default=CONFIG["training"]["batch_size"])
    parser.add_argument("--num_workers", type=int, default=CONFIG["training"]["num_workers"])
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        print(f"No .csv or .parquet files found under {data_dir}")
        sys.exit(0)

    # Output directory: data_dir/prediction/
    out_dir = os.path.join(data_dir, "prediction")
    os.makedirs(out_dir, exist_ok=True)

    model_params = CONFIG["model"]["params"].copy()
    model_params['use_attention'] = CONFIG["ablation"]["use_attention"]
    selected_features = collect_selected_features(CONFIG)
    model_params['use_additional_features'] = selected_features if selected_features else None

    model = DeepPrime7(**model_params).to(device)
    print(f"Loading model: {args.model_path}")
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    scaler = None
    if selected_features:
        print(f"Loading scaler: {args.scaler_path}")
        scaler = load_scaler(args.scaler_path)

    for path in files:
        print(f"\nProcessing: {os.path.basename(path)}")
        try:
            df = read_table(path)
        except Exception as e:
            warnings.warn(f" - Skipped (read error): {path} ({e})")
            continue

        missing_seq = [c for c in ESSENTIAL_SEQS if c not in df.columns]
        if missing_seq:
            warnings.warn(f" - Skipped (missing essential columns {missing_seq}): {path}")
            continue

        if selected_features:
            missing_feat = ensure_features(df, selected_features)
            if missing_feat:
                warnings.warn(f" - Skipped (missing feature columns {missing_feat}): {path}")
                continue

        try:
            pred_df = run_inference_on_df(
                df=df,
                model=model,
                device=device,
                selected_features=selected_features,
                scaler=scaler,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            out_path = save_predictions(pred_df, out_dir, path)
            print(f" - Saved: {out_path}")
        except Exception as e:
            warnings.warn(f" - Failed on {path}: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()