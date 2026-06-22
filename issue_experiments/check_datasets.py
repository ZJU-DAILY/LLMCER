"""
Dataset diagnostic: checks each dataset for loadability and data/ground-truth
ID alignment, and recommends which datasets are usable.

A dataset is USABLE only if:
  * its records load and the first column is an integer record id,
  * a ground-truth file exists, and
  * the ground-truth ids reference records that actually exist in the data
    (id range matches -- a two-table benchmark whose second table is missing
    cannot be evaluated).

Run:  python issue_experiments/check_datasets.py
"""

import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "cora":        ("datasets/cora/cora.csv",                "datasets/cora/gt.csv"),
    "citeseer":    ("datasets/citesheer/Citesheer_dblp.csv", "datasets/citesheer/citesheer_gt.txt"),
    "google-DBLP": ("datasets/google-DBLP/data.csv",         "datasets/google-DBLP/gt.csv"),
    "walmart_amz": ("datasets/Walmart_Amazon/tableA.csv",    None),
    "music20K":    ("datasets/music20K/music20K.csv",        "datasets/music20K/ground_truth.txt"),
    "sigmod":      ("datasets/sigmod/alaska.csv",            "datasets/sigmod/alaska_gt.csv"),
    "song":        ("datasets/song/songs.csv",              "datasets/song/gt.txt"),
    "affiliation": ("datasets/affiliation/new_affi_data.csv","datasets/affiliation/new_mapping.csv"),
}


def read_rows(path):
    last = None
    for enc in ("utf-8", "MacRoman", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.reader(f)), enc
        except Exception as e:
            last = e
    return None, str(last)


def diagnose():
    verdicts = {}
    for name, (dp, gp) in DATASETS.items():
        print("=" * 60)
        print(name)
        dp_abs = os.path.join(ROOT, dp)
        if not os.path.exists(dp_abs):
            print("  data file MISSING:", dp)
            verdicts[name] = ("DELETE", "data file missing")
            continue
        rows, enc = read_rows(dp_abs)
        if rows is None:
            print("  data UNREADABLE:", enc)
            verdicts[name] = ("DELETE", "data unreadable")
            continue
        n = len(rows) - 1
        ids = [int(r[0]) for r in rows[1:] if r and r[0].strip().lstrip("-").isdigit()]
        contiguous = ids == list(range(n))
        print(f"  data cols: {rows[0]}")
        print(f"  records={n}  id-range=[{min(ids)},{max(ids)}]  contiguous_0..N-1={contiguous}")

        if gp is None:
            print("  GROUND TRUTH: none provided")
            verdicts[name] = ("DELETE", "no ground-truth file (two-table set incomplete)")
            continue
        gp_abs = os.path.join(ROOT, gp)
        if not os.path.exists(gp_abs):
            print("  GROUND TRUTH file MISSING:", gp)
            verdicts[name] = ("DELETE", "ground-truth file missing")
            continue

        gids = set()
        if gp.endswith(".txt"):
            nclust = 0
            with open(gp_abs, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        nclust += 1
                    for t in parts:
                        if t.lstrip("-").isdigit():
                            gids.add(int(t))
            print(f"  GT format: cluster-per-line, {nclust} clusters")
        else:
            grows, _ = read_rows(gp_abs)
            print(f"  GT format: pair list, cols={grows[0]}, {len(grows)-1} rows")
            for r in grows[1:]:
                for c in r[:2]:
                    if c.strip().lstrip("-").isdigit():
                        gids.add(int(c))

        out_of_range = sum(1 for x in gids if not (0 <= x < n))
        print(f"  GT ids: distinct={len(gids)} range=[{min(gids)},{max(gids)}] "
              f"out-of-data-range={out_of_range}")

        if out_of_range > 0:
            verdicts[name] = ("DELETE",
                              f"{out_of_range} GT ids reference records absent from data "
                              f"(two-table benchmark: second table not provided)")
        else:
            verdicts[name] = ("KEEP", "data & GT ids aligned")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    keep = [n for n, (v, _) in verdicts.items() if v == "KEEP"]
    delete = [n for n, (v, _) in verdicts.items() if v == "DELETE"]
    for name, (v, why) in verdicts.items():
        print(f"  [{v}] {name}: {why}")
    print(f"\nKEEP ({len(keep)}): {keep}")
    print(f"DELETE ({len(delete)}): {delete}")
    return verdicts


if __name__ == "__main__":
    diagnose()
