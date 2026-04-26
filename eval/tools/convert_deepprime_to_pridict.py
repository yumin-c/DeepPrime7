"""
Utility to approximate PRIDICT2-style inputs from DeepPrime test data.

Assumptions/limitations:
 - `Target` is the pre-edit sequence.
 - `Masked_EditSeq` reveals the edited region; 'x' means use the base from `Target`.
 - Insertions/deletions cannot be perfectly reconstructed from the masked 74 bp
   strings; we therefore:
     * build a "mutated" sequence by overlaying any non-'x' characters onto Target;
     * carry over `edit_type` and `Edit_len` to Correction_Type/Length;
     * mark edit positions based on the first differing base (Target vs overlaid).
 - Many PRIDICT2 handcrafted features (PBS/RT lengths, thermodynamic scores, etc.)
   are not present and are emitted as NaN placeholders.
"""

import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


EDIT_TYPE_MAP = {
    "Sub": "Replacement",
    "MultiSub": "Replacement",
    "Ins": "Insertion",
    "Del": "Deletion",
}


def infer_mutated_sequence(target, masked) -> tuple[str, list[int]]:
    """Overlay masked bases onto target. Return mutated seq and differing indices."""
    target = "" if pd.isna(target) else str(target)
    masked = "" if pd.isna(masked) else str(masked)
    mutated = list(target)
    diff_idx = []
    for i, (t, m) in enumerate(zip(target, masked)):
        if m != "x":
            mutated[i] = m
            if m != t:
                diff_idx.append(i)
    return "".join(mutated), diff_idx


def convert(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        target = row["Target"]
        masked = row["Masked_EditSeq"]
        mutated, diff_idx = infer_mutated_sequence(target, masked)

        if diff_idx:
            edit_start = min(diff_idx)
            edit_positions = diff_idx
        else:
            # fallback: use Edit_pos if overlay found nothing
            edit_pos_val = row.get("Edit_pos", 0)
            edit_len_val = row.get("Edit_len", 0)
            edit_start = 0 if pd.isna(edit_pos_val) else int(edit_pos_val)
            edit_len = 0 if pd.isna(edit_len_val) else int(edit_len_val)
            edit_positions = list(range(edit_start, edit_start + edit_len))

        corr_type = EDIT_TYPE_MAP.get(row.get("edit_type", ""), "Replacement")
        corr_len = row.get("Edit_len", None)

        rows.append(
            {
                # minimal required by PRIDICT2 processing
                "seq_id": f"seq_{row['idx']}",
                "wide_initial_target": target,
                "wide_mutated_target": mutated,
                "Correction_Type": corr_type,
                "Correction_Length": corr_len,
                "deepeditposition": edit_start,
                "deepeditposition_lst": edit_positions,
                # placeholders for features PRIDICT2 typically expects
                "RToverhangmatches": pd.NA,
                "RToverhanglength": pd.NA,
                "RTlength": pd.NA,
                "PBSlength": pd.NA,
                "RTmt": pd.NA,
                "RToverhangmt": pd.NA,
                "PBSmt": pd.NA,
                "protospacermt": pd.NA,
                "extensionmt": pd.NA,
                "original_base_mt": pd.NA,
                "edited_base_mt": pd.NA,
                "protospacerlocation_only_initial": pd.NA,
                "PBSlocation": pd.NA,
                "RT_initial_location": pd.NA,
                "RT_mutated_location": pd.NA,
                # keep original DeepPrime fields for traceability
                "dp_idx": row["idx"],
                "dp_edit_type": row.get("edit_type", pd.NA),
                "dp_Edit_len": corr_len,
                "dp_Edit_pos": row.get("Edit_pos", pd.NA),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Approximate PRIDICT2 input from DeepPrime processed_test.csv"
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "processed_test.csv"),
        help="Path to DeepPrime processed_test.csv",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "processed_test_pridict_like.csv"),
        help="Path to write PRIDICT-like CSV",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    df = pd.read_csv(inp)
    out_df = convert(df)
    out_df.to_csv(out, index=False)
    print(f"✅ Wrote {len(out_df)} rows to {out}")
    print(
        "⚠️ Note: insertion/deletion sequences and thermodynamic features are placeholders; "
        "verify before PRIDICT2 inference."
    )


if __name__ == "__main__":
    main()
