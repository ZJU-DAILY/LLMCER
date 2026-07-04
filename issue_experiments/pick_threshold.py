"""
Pick a recall-aware best b_t per dataset from the blocking sweep.

Rule: among the swept thresholds, keep those with PC (recall) >= TARGET_PC,
then choose the one with the highest RR (reduction ratio = efficiency). This
guarantees we never sacrifice blocking recall below TARGET_PC while still
keeping the candidate set as small as possible. If NO threshold reaches
TARGET_PC, fall back to the threshold with the maximum PC.

The sweep numbers are hard-coded from issue_experiments/results (the run already
done) so this is instant and needs no SBERT/LLM. Re-run blocking_recall.py if
the embeddings or blocker change.
"""

TARGET_PC = 0.85

SWEEP = {
    "cora": [
        (0.30, 0.9923, 0.0207, 0.0406, 0.0184),
        (0.40, 0.9980, 0.0206, 0.0403, 0.0046),
        (0.50, 0.9902, 0.0209, 0.0410, 0.0306),
        (0.60, 0.9859, 0.0215, 0.0420, 0.0578),
        (0.70, 0.9248, 0.0258, 0.0503, 0.2657),
        (0.80, 0.8891, 0.2457, 0.3850, 0.9258),
        (0.90, 0.8302, 0.7557, 0.7912, 0.9775),
    ],
    "citeseer": [
        (0.30, 0.9632, 0.0004, 0.0008, 0.1895),
        (0.40, 0.9273, 0.0007, 0.0013, 0.5411),
        (0.50, 0.8854, 0.0050, 0.0099, 0.9422),
        (0.60, 0.8834, 0.1282, 0.2240, 0.9978),
        (0.70, 0.8805, 0.9291, 0.9042, 0.9997),
        (0.80, 0.8661, 0.9891, 0.9235, 0.9997),
        (0.90, 0.8128, 0.9978, 0.8958, 0.9997),
    ],
    "google-DBLP": [
        (0.30, 0.9807, 0.0005, 0.0010, 0.0474),
        (0.40, 0.9561, 0.0005, 0.0010, 0.1185),
        (0.50, 0.8787, 0.0006, 0.0013, 0.3543),
        (0.60, 0.8048, 0.0014, 0.0027, 0.7210),
        (0.70, 0.6951, 0.0641, 0.1174, 0.9949),
        (0.80, 0.6483, 0.5253, 0.5804, 0.9994),
        (0.90, 0.4106, 0.9423, 0.5720, 0.9998),
    ],
    "music20K": [
        (0.30, 0.9902, 0.0001, 0.0002, 0.0144),
        (0.40, 0.9276, 0.0001, 0.0002, 0.1248),
        (0.50, 0.8023, 0.0001, 0.0002, 0.4140),
        (0.60, 0.5891, 0.0005, 0.0009, 0.8896),
        (0.70, 0.5145, 0.0973, 0.1636, 0.9995),
        (0.80, 0.4711, 0.6654, 0.5516, 0.9999),
        (0.90, 0.2539, 0.9586, 0.4015, 1.0000),
    ],
    "sigmod": [
        (0.30, 1.0000, 0.0020, 0.0040, 0.0000),
        (0.40, 0.9999, 0.0020, 0.0040, 0.0005),
        (0.50, 0.9993, 0.0020, 0.0040, 0.0015),
        (0.60, 0.9996, 0.0020, 0.0040, 0.0028),
        (0.70, 0.9919, 0.0025, 0.0050, 0.2032),
        (0.80, 0.9522, 0.0029, 0.0058, 0.3395),
        (0.90, 0.5768, 0.0484, 0.0893, 0.9761),
    ],
    "song": [
        (0.30, 0.9507, 0.0008, 0.0015, 0.0827),
        (0.40, 0.7824, 0.0013, 0.0026, 0.5566),
        (0.50, 0.5959, 0.1410, 0.2281, 0.9969),
        (0.60, 0.6003, 0.8257, 0.6952, 0.9995),
        (0.70, 0.5837, 0.9213, 0.7147, 0.9995),
        (0.80, 0.5222, 0.9723, 0.6795, 0.9996),
        (0.90, 0.4434, 0.9974, 0.6139, 0.9997),
    ],
    "affiliation": [
        (0.30, 0.9225, 0.0069, 0.0138, 0.1267),
        (0.40, 0.8730, 0.0070, 0.0138, 0.1756),
        (0.50, 0.8332, 0.0076, 0.0151, 0.2813),
        (0.60, 0.7705, 0.0131, 0.0257, 0.6117),
        (0.698, 0.7093, 0.0190, 0.0370, 0.7542),
        (0.80, 0.4128, 0.1007, 0.1620, 0.9730),
        (0.90, 0.2502, 0.3397, 0.2882, 0.9952),
    ],
}


def pick(rows, target_pc):
    ok = [r for r in rows if r[1] >= target_pc]
    if ok:
        best = max(ok, key=lambda r: r[4])
        return best, "PC>=target, max RR"
    best = max(rows, key=lambda r: r[1])
    return best, "no threshold hits target; max PC fallback"


def main():
    print(f"Recall-aware threshold selection  (target PC >= {TARGET_PC})")
    print("=" * 78)
    print(f"{'Dataset':<13}{'b_t':>7}{'PC':>9}{'PQ':>9}{'F1':>9}{'RR':>9}   note")
    print("-" * 78)
    chosen = {}
    for name, rows in SWEEP.items():
        (bt, pc, pq, f1, rr), note = pick(rows, TARGET_PC)
        chosen[name] = bt
        print(f"{name:<13}{bt:>7.3f}{pc:>9.4f}{pq:>9.4f}{f1:>9.4f}{rr:>9.4f}   {note}")
    print("=" * 78)
    print("Chosen thresholds dict (paste into run_pipeline BEST_BLOCK_PER_DATASET):")
    print(chosen)


if __name__ == "__main__":
    main()
