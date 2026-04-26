from pathlib import Path
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

REPO_ROOT = Path(__file__).resolve().parent.parent

SUFFIX_4K = "CTACTCTACCACTTGTACTTCAGCGGTCAGCTTACTCGACTTAACGTGCACGTGACACGTTCTAGACCGTACA"
SUFFIX_20K = "GCTCGCGCTACTCTACCACTTGTACTTCAGCGGTCAGCTTACTCGACTTAACGTGCACGTGACACGTTCTAGA"

INPUT_PATH = REPO_ROOT / "data" / "250716_24k_final.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "24k_final_pridict_input.csv"
SPLIT_PATH = REPO_ROOT / "data" / "processed_all.csv"

WIDE_CONTEXT_LEN = 99
PROTO_UPSTREAM = 10  # wide target starts 10 bp upstream of protospacer
RNG = np.random.default_rng(42)


def revcomp(seq):
    return str(Seq(seq).reverse_complement())


def replace_ambiguous(seq):
    if not seq:
        return seq
    out = []
    for ch in seq.upper():
        if ch in "ACGT":
            out.append(ch)
        else:
            out.append(RNG.choice(list("ACGT")))
    return "".join(out)


def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def longest_common_prefix(a, b):
    i = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        i += 1
    return i


def longest_common_suffix(a, b):
    i = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        i += 1
    return i


def find_all(haystack, needle):
    if not needle:
        return []
    hits = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        hits.append(idx)
        start = idx + 1
    return hits


def pick_nearest_index(indices, target):
    if not indices:
        return None
    return min(indices, key=lambda x: abs(x - target))


def build_targets(row):
    combined = row["Combined"]
    subpool = row["Subpool"]
    if subpool == "4K":
        oligo = f"{combined}{SUFFIX_4K}"
    elif subpool == "20K":
        oligo = f"{combined}{SUFFIX_20K}"
    else:
        return None, None, None
    target = revcomp(oligo)
    return oligo, target, subpool


def derive_edit_info(unedited, edited, unedited_start_in_wide):
    if len(unedited) == len(edited):
        diff_positions = [i for i, (a, b) in enumerate(zip(unedited, edited)) if a != b]
        if not diff_positions:
            return None, None, None, None
        deepeditposition = unedited_start_in_wide + diff_positions[0]
        deepeditposition_lst = [unedited_start_in_wide + i for i in diff_positions]
        original_base = "".join(unedited[i] for i in diff_positions)
        edited_base = "".join(edited[i] for i in diff_positions)
        correction_type = "Replacement"
        correction_length = len(diff_positions)
    else:
        lcp = longest_common_prefix(unedited, edited)
        lcs = longest_common_suffix(unedited, edited)
        if len(edited) > len(unedited):
            ins_len = len(edited) - len(unedited)
            inserted = edited[lcp : lcp + ins_len]
            deepeditposition = unedited_start_in_wide + lcp
            deepeditposition_lst = [deepeditposition]
            original_base = "-"
            edited_base = inserted
            correction_type = "Insertion"
            correction_length = ins_len
        else:
            del_len = len(unedited) - len(edited)
            deleted = unedited[lcp : lcp + del_len]
            deepeditposition = unedited_start_in_wide + lcp
            deepeditposition_lst = [deepeditposition]
            original_base = deleted
            edited_base = "-"
            correction_type = "Deletion"
            correction_length = del_len
    return deepeditposition, deepeditposition_lst, original_base, edited_base, correction_type, correction_length


def melting_temperature(protospacer, extension, rt_seq, rtoverhang, pbs, original_base, edited_base):
    protospacermt = mt.Tm_Wallace(Seq(protospacer))
    extensionmt = mt.Tm_Wallace(Seq(extension))
    rtmt = mt.Tm_Wallace(Seq(rt_seq))
    rtoverhangmt = mt.Tm_Wallace(Seq(rtoverhang))
    pbsmt = mt.Tm_Wallace(Seq(pbs))

    if original_base == "-":
        original_base_mt = 0
        original_base_mt_nan = 1
    else:
        original_base_mt = mt.Tm_Wallace(Seq(original_base))
        original_base_mt_nan = 0

    if edited_base == "-":
        edited_base_mt = 0
        edited_base_mt_nan = 1
    else:
        edited_base_mt = mt.Tm_Wallace(Seq(edited_base))
        edited_base_mt_nan = 0

    return (
        protospacermt,
        extensionmt,
        rtmt,
        rtoverhangmt,
        pbsmt,
        original_base_mt,
        edited_base_mt,
        original_base_mt_nan,
        edited_base_mt_nan,
    )


def main():
    df = pd.read_csv(INPUT_PATH)
    split_df = pd.read_csv(SPLIT_PATH, usecols=["idx", "Split"])
    split_map = dict(zip(split_df["idx"], split_df["Split"]))
    records = []
    skipped = []

    for _, row in df.iterrows():
        oligo, target, subpool = build_targets(row)
        if target is None:
            skipped.append((row.get("idx"), "unknown_subpool"))
            continue

        spacer_val = row.get("Spacer")
        if pd.isna(spacer_val):
            skipped.append((row.get("idx"), "spacer_missing"))
            continue
        spacer = str(spacer_val)
        spacer_len = len(spacer)

        unedited = str(row["Unedited (gRNA binding -2 ~ RTT binding +2)"])
        edited = str(row["Prime-edited (gRNA binding -2 ~ RTT binding +2)"])
        unedited_hits = find_all(target, unedited)
        if not unedited_hits:
            skipped.append((row.get("idx"), "unedited_not_found"))
            continue

        spacer_start = target.find(spacer)
        spacer_found = spacer_start != -1
        if spacer_found:
            expected_unedited_start = spacer_start - 2
            unedited_start = pick_nearest_index(unedited_hits, expected_unedited_start)
        else:
            unedited_start = unedited_hits[0]
            spacer_start = unedited_start + 2

        wide_start = spacer_start - PROTO_UPSTREAM
        wide_end = wide_start + WIDE_CONTEXT_LEN
        if wide_start < 0 or wide_end > len(target):
            skipped.append((row.get("idx"), "wide_context_out_of_bounds"))
            continue

        if spacer_start + spacer_len > len(target):
            skipped.append((row.get("idx"), "spacer_out_of_bounds"))
            continue

        target_edited = (
            target[:unedited_start] + edited + target[unedited_start + len(unedited) :]
        )
        wide_initial_target = target[wide_start:wide_end]
        wide_mutated_target = target_edited[wide_start : wide_start + WIDE_CONTEXT_LEN]

        unedited_start_in_wide = unedited_start - wide_start
        edit_info = derive_edit_info(unedited, edited, unedited_start_in_wide)
        if edit_info[0] is None:
            skipped.append((row.get("idx"), "no_edit_detected"))
            continue

        (
            deepeditposition,
            deepeditposition_lst,
            original_base,
            edited_base,
            correction_type,
            correction_length,
        ) = edit_info

        # replace ambiguous bases after all indexing logic
        wide_initial_target = replace_ambiguous(wide_initial_target)
        wide_mutated_target = replace_ambiguous(wide_mutated_target)

        if pd.isna(row.get("RTT_len")) or pd.isna(row.get("PBS_len")) or pd.isna(row.get("RT-PBS")):
            skipped.append((row.get("idx"), "rt_pbs_missing"))
            continue
        rtt_len = int(row["RTT_len"])
        pbs_len = int(row["PBS_len"])
        rt_pbs = replace_ambiguous(str(row["RT-PBS"]))
        rt_seq = rt_pbs[:rtt_len]
        pbs_seq = rt_pbs[rtt_len : rtt_len + pbs_len]
        extension_seq = rt_pbs

        spacer_seq_target = target[spacer_start : spacer_start + spacer_len]
        protospacer_seq = "G" + spacer_seq_target

        # derive genomic RTT for overhang match counting
        rtoverhang_start = deepeditposition + (0 if edited_base == "-" else len(edited_base))
        rtoverhang_seq = wide_mutated_target[
            rtoverhang_start : rtoverhang_start + rtt_len
        ]
        rtoverhangmatches = occurrences(
            wide_mutated_target[rtoverhang_start : rtoverhang_start + rtt_len + 15],
            rtoverhang_seq,
        )

        (
            protospacermt,
            extensionmt,
            rtmt,
            rtoverhangmt,
            pbsmt,
            original_base_mt,
            edited_base_mt,
            original_base_mt_nan,
            edited_base_mt_nan,
        ) = melting_temperature(
            protospacer_seq,
            extension_seq,
            rt_seq,
            rt_seq,
            pbs_seq,
            replace_ambiguous(original_base) if original_base != "-" else original_base,
            replace_ambiguous(edited_base) if edited_base != "-" else edited_base,
        )

        protospacerlocation_only_initial = [PROTO_UPSTREAM, PROTO_UPSTREAM + len(spacer)]
        pbs_location = [
            PROTO_UPSTREAM + len(spacer) - 3 - pbs_len,
            PROTO_UPSTREAM + len(spacer) - 3,
        ]
        rt_anchor = PROTO_UPSTREAM + len(spacer) - 3
        if correction_type == "Replacement":
            rt_initial_location = [rt_anchor, rt_anchor + rtt_len]
            rt_mutated_location = [rt_anchor, rt_anchor + rtt_len]
        elif correction_type == "Deletion":
            rt_initial_location = [rt_anchor, rt_anchor + rtt_len + correction_length]
            rt_mutated_location = [rt_anchor, rt_anchor + rtt_len]
        else:
            rt_initial_location = [rt_anchor, rt_anchor + max(rtt_len - correction_length, 0)]
            rt_mutated_location = [rt_anchor, rt_anchor + rtt_len]

        records.append(
            {
                "seq_id": row.get("idx"),
                "Split": split_map.get(row.get("idx")),
                "wide_initial_target": wide_initial_target,
                "wide_mutated_target": wide_mutated_target,
                "deepeditposition": int(deepeditposition),
                "deepeditposition_lst": str(deepeditposition_lst),
                "Correction_Type": correction_type,
                "Correction_Length": int(correction_length),
                "protospacerlocation_only_initial": str(protospacerlocation_only_initial),
                "PBSlocation": str(pbs_location),
                "RT_initial_location": str(rt_initial_location),
                "RT_mutated_location": str(rt_mutated_location),
                "RToverhangmatches": int(rtoverhangmatches),
                "RToverhanglength": int(rtt_len),
                "RTlength": int(rtt_len),
                "PBSlength": int(pbs_len),
                "RTmt": float(rtmt),
                "RToverhangmt": float(rtoverhangmt),
                "PBSmt": float(pbsmt),
                "protospacermt": float(protospacermt),
                "extensionmt": float(extensionmt),
                "original_base_mt": float(original_base_mt),
                "edited_base_mt": float(edited_base_mt),
                "original_base_mt_nan": int(original_base_mt_nan),
                "edited_base_mt_nan": int(edited_base_mt_nan),
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_PATH, index=False)

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["idx", "reason"])
        skipped_df.to_csv("PE7/data/24k_final_pridict_input_skipped.csv", index=False)


if __name__ == "__main__":
    main()
