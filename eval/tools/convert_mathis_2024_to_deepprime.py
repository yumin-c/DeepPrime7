import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction as gc
from RNA import fold_compound

REPO_ROOT = Path(__file__).resolve().parent.parent


EDIT_TYPE_MAP = {
    "Replacement": "sub",
    "Insertion": "ins",
    "Deletion": "del",
}


def parse_loc(val):
    if pd.isna(val):
        return None
    if isinstance(val, (list, tuple)):
        return [int(val[0]), int(val[-1])]
    s = str(val).strip()
    if not s:
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and parsed:
            return [int(parsed[0]), int(parsed[-1])]
    except (ValueError, SyntaxError):
        pass
    parts = s.strip("[]").split(",")
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None
    return [int(parts[0]), int(parts[-1])]


def parse_edit_positions(row):
    raw = row.get("deepeditposition_intermediate_individual")
    if isinstance(raw, str) and raw.strip():
        try:
            vals = ast.literal_eval(raw)
            return [int(v) for v in vals]
        except (ValueError, SyntaxError):
            pass
    val = row.get("deepeditposition")
    if pd.isna(val):
        return []
    return [int(val)]


def align_wt_mut(wide_initial, wide_mutated, correction_type, correction_len, editpos_lst):
    if not editpos_lst:
        return wide_initial, wide_mutated
    ewindow_st = editpos_lst[0]
    if correction_type == "Insertion":
        w_repr = wide_initial[:ewindow_st] + "-" * correction_len + wide_initial[ewindow_st:]
        if wide_initial[ewindow_st:] != wide_mutated[ewindow_st + correction_len :]:
            m_repr = wide_mutated + "-" * correction_len
        else:
            m_repr = wide_mutated
    elif correction_type == "Deletion":
        m_repr = wide_mutated[:ewindow_st] + "-" * correction_len + wide_mutated[ewindow_st:]
        if wide_initial[ewindow_st + correction_len :] != wide_mutated[ewindow_st:]:
            w_repr = wide_initial + "-" * correction_len
        else:
            w_repr = wide_initial
    else:
        w_repr = wide_initial
        m_repr = wide_mutated
    return w_repr, m_repr


def build_index_map(aligned_seq):
    mapping = {}
    unaligned_idx = 0
    for aligned_idx, ch in enumerate(aligned_seq):
        if ch != "-":
            mapping[unaligned_idx] = aligned_idx
            unaligned_idx += 1
    return mapping


def compute_tm_features(target, pbs, rtt, edit_type, edit_len):
    seq_tm1 = str(Seq(pbs).transcribe())
    seq_tm2 = target[21 : 21 + len(rtt)]

    if edit_type == "sub":
        seq_tm3 = target[21 : 21 + len(rtt)]
        tm4_anti = str(Seq(target[21 : 21 + len(rtt)]).reverse_complement())
    elif edit_type == "ins":
        seq_tm3 = target[21 : 21 + len(rtt) - edit_len]
        tm4_anti = str(Seq(target[21 : 21 + len(rtt) - edit_len]).reverse_complement())
    else:
        seq_tm3 = target[21 : 21 + len(rtt) + edit_len]
        tm4_anti = str(Seq(target[21 : 21 + len(rtt) + edit_len]).reverse_complement())

    seq_tm4 = [
        str(Seq(rtt).reverse_complement()).replace("T", "U"),
        tm4_anti,
    ]
    seq_tm5 = str(Seq(rtt).transcribe())

    tm1 = mt.Tm_NN(seq=Seq(seq_tm1), nn_table=mt.R_DNA_NN1)
    tm2 = mt.Tm_NN(seq=Seq(seq_tm2), nn_table=mt.DNA_NN3)
    tm3 = mt.Tm_NN(seq=Seq(seq_tm3), nn_table=mt.DNA_NN3)

    tm4 = 0
    for s1, s2 in zip(seq_tm4[0], seq_tm4[1]):
        try:
            tm4 = mt.Tm_NN(seq=Seq(s1), c_seq=s2, nn_table=mt.DNA_NN3)
        except ValueError:
            tm4 = 0

    tm5 = mt.Tm_NN(seq=Seq(seq_tm5), nn_table=mt.R_DNA_NN1)
    delta_tm = tm4 - tm2
    return tm1, tm2, tm3, tm4, tm5, delta_tm


def compute_mfe_features(spacer, rtpbs):
    seq_mfe3 = rtpbs + "TTTTTT"
    _, mfe3 = fold_compound(seq_mfe3).mfe()
    seq_mfe4 = "G" + spacer[1:]
    _, mfe4 = fold_compound(seq_mfe4).mfe()
    return round(mfe3, 1), round(mfe4, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "mathis_2024.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "mathis_2024_deepprime_input.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    rows = []
    skipped = []

    for idx, row in df.iterrows():
        wide_initial = str(row["wide_initial_target"])
        wide_mutated = str(row["wide_mutated_target"])
        correction_type = str(row["Correction_Type"])
        correction_len = int(row["Correction_Length"])
        edit_positions = parse_edit_positions(row)

        pbsloc = parse_loc(row.get("PBSlocation"))
        rtloc = parse_loc(row.get("RT_initial_location"))
        if not pbsloc or not rtloc:
            skipped.append((idx, "missing_locations"))
            continue

        start_pbs, end_pbs = pbsloc
        start_rt, end_rt = rtloc
        target_start = end_pbs - 21
        target_end = target_start + 74
        if target_start < 0 or target_end > len(wide_initial):
            skipped.append((idx, "target_bounds"))
            continue

        w_aln, m_aln = align_wt_mut(
            wide_initial, wide_mutated, correction_type, correction_len, edit_positions
        )
        idx_map = build_index_map(w_aln)

        target = wide_initial[target_start:target_end]
        masked = []
        for pos in range(target_start, target_end):
            if start_pbs <= pos < end_rt:
                aligned_idx = idx_map.get(pos)
                base = m_aln[aligned_idx] if aligned_idx is not None else wide_mutated[pos]
                if base == "-":
                    base = wide_mutated[pos]
                masked.append(base)
            else:
                masked.append("x")
        masked_editseq = "".join(masked)

        pbs = str(row["PBS"])
        rtt = str(row["RTT"])
        edit_type = EDIT_TYPE_MAP.get(correction_type, "sub")
        edit_pos = int(row["Editing_Position"])
        edit_len = int(row["Correction_Length"])

        rtpbs = rtt + pbs
        spacer = str(row.get("spacer", target[4:24]))

        if edit_type == "sub":
            type_sub, type_ins, type_del = 1, 0, 0
            rha_len = len(rtt) - edit_pos - edit_len + 1
        elif edit_type == "ins":
            type_sub, type_ins, type_del = 0, 1, 0
            rha_len = len(rtt) - edit_pos - edit_len + 1
        else:
            type_sub, type_ins, type_del = 0, 0, 1
            rha_len = len(rtt) - edit_pos + 1

        tm1, tm2, tm3, tm4, tm5, delta_tm = compute_tm_features(
            target, pbs, rtt, edit_type, edit_len
        )
        mfe3, mfe4 = compute_mfe_features(spacer, rtpbs)

        deepcas9 = row.get("deepcas9")
        if pd.isna(deepcas9):
            deepcas9 = np.nan

        testset_fold = row.get("testset_fold")
        if pd.isna(testset_fold) or str(testset_fold).strip() == "":
            split = "Test"
        else:
            split = f"Fold{int(testset_fold)}"

        rows.append(
            {
                "idx": idx,
                "Target": target,
                "Masked_EditSeq": masked_editseq,
                "PBS_len": len(pbs),
                "RTT_len": len(rtt),
                "RT-PBS_len": len(rtpbs),
                "Edit_pos": edit_pos,
                "Edit_len": edit_len,
                "RHA_len_nn": rha_len,
                "type_sub": type_sub,
                "type_ins": type_ins,
                "type_del": type_del,
                "Tm1_PBS": tm1,
                "Tm2_RTT_cTarget_sameLength": tm2,
                "Tm3_RTT_cTarget_replaced": tm3,
                "Tm4_cDNA_PAM-oppositeTarget": tm4,
                "Tm5_RTT_cDNA": tm5,
                "deltaTm_Tm4-Tm2": delta_tm,
                "GC_count_PBS": pbs.count("G") + pbs.count("C"),
                "GC_count_RTT": rtt.count("G") + rtt.count("C"),
                "GC_count_RT-PBS": rtpbs.count("G") + rtpbs.count("C"),
                "GC_contents_PBS": 100 * gc(pbs),
                "GC_contents_RTT": 100 * gc(rtt),
                "GC_contents_RT-PBS": 100 * gc(rtpbs),
                "MFE_RT-PBS-polyT": mfe3,
                "MFE_Spacer": mfe4,
                "DeepSpCas9_score": deepcas9,
                "Split": split,
                "HEKaverageedited": row.get("HEKaverageedited"),
                "K562averageedited": row.get("K562averageedited"),
                "PRIDICT2_0_editing_Score_deep_HEK_mean5fold": row.get(
                    "PRIDICT2_0_editing_Score_deep_HEK_mean5fold"
                ),
                "PRIDICT2_0_editing_Score_deep_K562_mean5fold": row.get(
                    "PRIDICT2_0_editing_Score_deep_K562_mean5fold"
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["idx", "reason"])
        skipped_path = Path(args.output).with_suffix("").as_posix() + "_skipped.csv"
        pd.DataFrame(skipped, columns=["idx", "reason"]).to_csv(skipped_path, index=False)


if __name__ == "__main__":
    main()
