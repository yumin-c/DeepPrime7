# DeepPrime7

A multi-head CNN model for predicting prime editing (PE7) efficiency across multiple cell lines and PAM variants.

## Model

**DeepPrime7** predicts four efficiency scores simultaneously:

| Output | Cell line | PAM |
|--------|-----------|-----|
| `DN_M_EF` | DLD1 | NGG |
| `DC_M_EF` | DLD1 | NRCH |
| `DT_M_EF` | DLD1 | NRTH |
| `AN_M_EF` | A549 | NGG |

Architecture: Multi-scale CNN with residual blocks, channel attention, and per-target FC branches. See [`model.py`](model.py).

## Repository Structure

```
.
├── model.py             # DeepPrime7 architecture
├── predict.py           # Batch inference (CSV / parquet input)
├── scaler.py            # Refit the StandardScaler from the released splits
├── process_inputs.py    # Build train/valid/test splits from the 24 K library
│
├── data/
│   ├── processed_train.csv     # Training split (~24 K samples)
│   ├── processed_valid.csv     # Validation split (~2.5 K samples)
│   ├── processed_test.csv      # Test split (~2.5 K samples)
│   ├── processed_all.csv       # Full dataset (all splits combined)
│   ├── Result_final.csv        # Raw efficiency measurement data
│   └── Result_final_PAM.csv    # PAM-annotated measurement data
│
├── models/
│   ├── deepprime7.pth           # Final trained model weights
│   └── deepprime7_scaler.pth    # StandardScaler for input features
│
└── eval/                        # Evaluation and comparison scripts
    ├── eval_grid.py             # Full model comparison matrix (7 models × 7 datasets)
    ├── eval_grid_predict.py     # Generate predictions for eval_grid
    ├── run_deepprime_original.py # Run original DeepPrime ensembles on PE7 test data
    ├── run_deepprime_mathis.py  # Run DeepPrime / DP7 on the Mathis 2024 dataset
    ├── run_pridict2_test_preds.py # Run PRIDICT2 predictions
    ├── tools/
    │   ├── convert_mathis_2024_to_deepprime.py  # Convert Mathis 2024 CSV to DeepPrime format
    │   ├── make_pridict_input.py    # Build PRIDICT2 input from the 24 K design library
    │   └── convert_deepprime_to_pridict.py  # Approximate PRIDICT2 input from DeepPrime data
    └── clinvar/
        └── scenario_comparison.py  # DeepPrime vs DeepPrime7 scenario comparison on ClinVar
```

## Requirements

```
torch
scikit-learn
joblib
pandas
numpy
scipy
biopython
ViennaRNA    # provides the RNA package
tqdm
genet        # pegRNA design + feature computation (Tm, GC, MFE, DeepSpCas9)
```

Install with:

```bash
pip install torch scikit-learn joblib pandas numpy scipy biopython tqdm
pip install genet           # https://github.com/Goosang-Yu/genet
pip install viennarna       # or follow ViennaRNA installation docs
```

## Inference

Inference is a two-step process:

1. **Design pegRNAs and compute features** for each target edit (using `genet`)
2. **Score them with DeepPrime7** by running `predict.py` on the resulting CSV

### 1. Creating input files

`predict.py` expects each input file to contain the following columns:

| Column | Description |
|--------|-------------|
| `Target` | 74 nt pre-edit reference target |
| `Masked_EditSeq` | 74 nt sequence with the edit window revealed (rest masked with `x`) |
| `Tm1_PBS` | Melting temperature of the PBS / scaffold duplex |
| `Tm2_RTT_cTarget_sameLength` | Tm of the RTT vs. the same-length cTarget region |
| `Tm3_RTT_cTarget_replaced` | Tm of the RTT vs. the post-edit cTarget |
| `Tm4_cDNA_PAM-oppositeTarget` | Tm of the cDNA vs. the PAM-opposite strand |
| `Tm5_RTT_cDNA` | Tm of the RTT vs. the synthesised cDNA |
| `GC_contents_PBS`, `GC_contents_RTT` | GC content (%) of the PBS and RTT |
| `MFE_RT-PBS-polyT` | Minimum free energy of the RT-PBS extension with a poly-T tail |
| `MFE_Spacer` | Minimum free energy of the spacer hairpin |

The easiest way to generate these is with the [`genet`](https://github.com/Goosang-Yu/genet) package, whose `DeepPrime` class designs candidate pegRNAs and emits a feature DataFrame whose column names already match the table above:

```python
from pathlib import Path
from genet.predict import DeepPrime

# Encode the edit by wrapping (WT/ED) at the edit site.
# Provide ≥24 nt of flanking context on each side so the 74 nt windows
# can be extracted by genet.
seq = (
    "CCGAGTTGGTTCATCGAGCCCCAAAGCGCAA"   # 5' flank
    "(C/T)"                              # the edit (sub: C → T)
    "GAGTTAGGCCAATTGCAGTGCAGTTGCAGCC"   # 3' flank
)

pegrna = DeepPrime(
    sequence=seq,
    name="my_edit_001",
    pam="NGG",          # one of NGG, NGA, NAG, NRCH, NNGG
    pbs_min=7, pbs_max=15,
    rtt_min=0, rtt_max=40,
)

# All candidate pegRNAs for this edit, with features pre-computed.
df = pegrna.features

# Keep only the columns DeepPrime7 needs and write to the inference folder.
needed = [
    "Target", "Masked_EditSeq",
    "Tm1_PBS", "Tm2_RTT_cTarget_sameLength", "Tm3_RTT_cTarget_replaced",
    "Tm4_cDNA_PAM-oppositeTarget", "Tm5_RTT_cDNA",
    "GC_contents_PBS", "GC_contents_RTT",
    "MFE_RT-PBS-polyT", "MFE_Spacer",
]
out_dir = Path("my_inputs"); out_dir.mkdir(exist_ok=True)
df[needed].to_csv(out_dir / "my_edit_001.csv", index=False)
```

For batch design, loop over the function above and write one CSV per edit (or one combined CSV) into the same input folder.

If you already have pegRNAs designed (rather than letting genet enumerate them), use `genet.predict.DeepPrimeGuideRNA` to compute features for a known `(target, PBS, RTT, edit_pos, edit_len, edit_type)` tuple and emit the same feature columns.

### 2. Running predict.py

```bash
python predict.py \
    --data_dir my_inputs \
    --model_path models/deepprime7.pth \
    --scaler_path models/deepprime7_scaler.pth
```

`predict.py` walks every `.csv` / `.parquet` file in `--data_dir`, runs the DeepPrime7 forward pass, and writes a `<basename>_pred.csv` into `<data_dir>/prediction/`. The four prediction columns appended to each output are:

| Column | Cell line | PAM | Output scale |
|--------|-----------|-----|--------------|
| `DN_EF_Prediction` | DLD1 | NGG | model output × 5.0 |
| `DC_EF_Prediction` | DLD1 | NRCH | model output × 5.0 |
| `DT_EF_Prediction` | DLD1 | NRTH | model output × 5.0 |
| `AN_EF_Prediction` | A549 | NGG | model output × 2.0 |

## Evaluation Scripts (`eval/`)

Scripts in `eval/` compare DeepPrime7 against baseline models and run ClinVar scenario analyses. They require the original **DeepPrime** repository as a sibling directory of DeepPrime7:

```
parent_dir/
├── DeepPrime7/    # this repository
├── DeepPrime/     # https://github.com/yumin-c/DeepPrime
└── PRIDICT2/      # https://github.com/uzh-rpg/pridict2  (for run_pridict2_test_preds.py)
```

Run scripts from the DeepPrime7 repo root:

```bash
python eval/run_deepprime_original.py
python eval/eval_grid.py
python eval/clinvar/scenario_comparison.py
```

