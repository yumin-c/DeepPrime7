import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIDICT2_ROOT = REPO_ROOT.parent / "PRIDICT2"
sys.path.insert(0, str(PRIDICT2_ROOT))

from pridict.pridictv2.predict_outcomedistrib import PRIEML_Model


DATASET_NAME_REMAP = {"K562MLH1dn": "K562"}


def patch_model_options(model_dir: Path) -> None:
    config_dir = model_dir / "config"
    options_path = config_dir / "exp_options.pkl"
    state_dir = model_dir / "model_statedict"
    if not options_path.exists():
        return
    with options_path.open("rb") as f:
        options = pickle.load(f)

    decoder_names = sorted(
        p.name.replace("decoder_", "").replace(".pkl", "")
        for p in state_dir.glob("decoder_*.pkl")
    )
    if not decoder_names:
        return

    datasets = options.get("datasets_name", [])
    remapped = [DATASET_NAME_REMAP.get(name, name) for name in datasets]
    if set(remapped) == set(decoder_names):
        return

    options["datasets_name"] = decoder_names
    if "trainable_layernames" in options:
        new_layers = []
        for layer in options["trainable_layernames"]:
            for old, new in DATASET_NAME_REMAP.items():
                layer = layer.replace(old, new)
            new_layers.append(layer)
        options["trainable_layernames"] = new_layers
    with options_path.open("wb") as f:
        pickle.dump(options, f)


def load_pridict_models(run_ids):
    script_dir = os.path.abspath("PRIDICT2")
    repo_dir = script_dir
    modellist = [
        ("PRIDICT1_2", "base", "exp_2023-12-22_16-24-32"),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsize = 20
    normalize_opt = "max"
    prieml_model = PRIEML_Model(device, wsize=wsize, normalize=normalize_opt, fdtype=torch.float32)

    models_lst_dict = {}
    for model_desc_tup in modellist:
        models_lst = []
        model_id, __, mfolder = model_desc_tup
        for run_num in run_ids:
            model_dir = os.path.join(
                repo_dir,
                "trained_models",
                model_id.lower(),
                mfolder,
                "train_val",
                f"run_{run_num}",
            )
            patch_model_options(Path(model_dir))
            loaded_model = prieml_model.build_retrieve_models(model_dir)
            models_lst.append((loaded_model, model_dir))
        models_lst_dict[model_id] = models_lst
    return prieml_model, models_lst_dict


def deeppridict(pegdataframe, prieml_model, models_lst_dict):
    deepdfcols = [
        "wide_initial_target",
        "wide_mutated_target",
        "deepeditposition",
        "deepeditposition_lst",
        "Correction_Type",
        "Correction_Length",
        "protospacerlocation_only_initial",
        "PBSlocation",
        "RT_initial_location",
        "RT_mutated_location",
        "RToverhangmatches",
        "RToverhanglength",
        "RTlength",
        "PBSlength",
        "RTmt",
        "RToverhangmt",
        "PBSmt",
        "protospacermt",
        "extensionmt",
        "original_base_mt",
        "edited_base_mt",
        "original_base_mt_nan",
        "edited_base_mt_nan",
    ]

    deepdf = pegdataframe[deepdfcols].copy()
    deepdf.insert(1, "seq_id", list(range(len(deepdf))))
    for col in (
        "protospacerlocation_only_initial",
        "PBSlocation",
        "RT_initial_location",
        "RT_mutated_location",
        "deepeditposition_lst",
    ):
        deepdf[col] = deepdf[col].apply(lambda x: str(x))

    deepdf["edited_base_mt"] = deepdf.apply(
        lambda x: 0 if x.Correction_Type == "Deletion" else x.edited_base_mt, axis=1
    )
    deepdf["original_base_mt"] = deepdf.apply(
        lambda x: 0 if x.Correction_Type == "Insertion" else x.original_base_mt, axis=1
    )

    plain_tcols = ["averageedited", "averageunedited", "averageindel"]
    cell_types = ["HEK", "K562"]
    batch_size = int(1500 / len(cell_types))
    dloader = prieml_model.prepare_data(
        deepdf, None, cell_types=cell_types, y_ref=[], batch_size=batch_size
    )

    all_avg_preds = {}
    for model_id, model_runs_lst in models_lst_dict.items():
        pred_dfs = []
        runs_c = 0
        for loaded_model_lst, model_dir in model_runs_lst:
            pred_df = prieml_model.predict_from_dloader_using_loaded_models(
                dloader, loaded_model_lst, y_ref=plain_tcols
            )
            pred_df["run_num"] = runs_c
            pred_dfs.append(pred_df)
            runs_c += 1
        pred_df_allruns = pd.concat(pred_dfs, axis=0, ignore_index=True)
        avg_preds = prieml_model.compute_avg_predictions(pred_df_allruns)
        avg_preds["model"] = model_id
        all_avg_preds[model_id] = avg_preds
    return all_avg_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="Test")
    parser.add_argument("--use-5folds", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.loc[df["Split"] == args.split].copy()
    if df.empty:
        raise ValueError(f"No rows found for Split == {args.split}")

    run_ids = [0, 1, 2, 3, 4] if args.use_5folds else [0]
    prieml_model, models_lst_dict = load_pridict_models(run_ids)

    all_avg_preds = deeppridict(df, prieml_model, models_lst_dict)
    pred_df = all_avg_preds["PRIDICT1_2"]
    pred_df = pred_df[["seq_id", "dataset_name", "pred_averageedited"]]
    pivot = pred_df.pivot(index="seq_id", columns="dataset_name", values="pred_averageedited")

    df = df.reset_index(drop=True)
    df["PRIDICT2_HEK"] = pivot.reindex(df.index)["HEK"].to_numpy() * 100
    df["PRIDICT2_K562"] = pivot.reindex(df.index)["K562"].to_numpy() * 100

    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
