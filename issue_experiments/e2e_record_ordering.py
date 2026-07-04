import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASETS = {
    "cora":           ("datasets/cora/cora.csv",                     "datasets/cora/gt.csv"),
    "citeseer":       ("datasets/citesheer/Citesheer_dblp.csv",      "datasets/citesheer/citesheer_gt.txt"),
    "google-DBLP":    ("datasets/google-DBLP/data.csv",              "datasets/google-DBLP/gt.csv"),
    "music20K":       ("datasets/music20K/music20K.csv",             "datasets/music20K/ground_truth.txt"),
    "sigmod_subset":  ("datasets/sigmod/alaska.csv",                 "datasets/sigmod/alaska_gt.csv"),
    "song":           ("datasets/song/songs.csv",                    "datasets/song/gt.txt"),
    "affiliation":    ("datasets/affiliation/new_affi_data.csv",     "datasets/affiliation/new_mapping.csv"),
    "walmart-amazon": ("datasets/Walmart_Amazon/walmart_amazon.csv", "datasets/Walmart_Amazon/gt.csv"),
    "amazon-google":  ("datasets/Amazon-Google/amazon_google.csv",   "datasets/Amazon-Google/gt.csv"),
}
BEST_BLOCK = {'cora': 0.89, 'song': 0.70, 'citesheer': 0.70, 'google-dblp': 0.85,
              'music20k': 0.70, 'amazon-google': 0.83, 'affiliation': 0.85,
              'walmart_amazon': 0.82,
              'sigmod': 0.949}
BEST_MERGE = {'sigmod': 0.95, 'affiliation': 0.90}

STRATEGIES = ["similarity_ordered", "weak_ordered", "random_shuffle"]

def install_ordering_patch(strategy):
    from llmcer import record_set
    _orig = record_set._greedy_similarity_order

    if strategy == "similarity_ordered":
        record_set._greedy_similarity_order = _orig
        return

    if strategy == "random_shuffle":
        import random
        def _shuffle(ids, simi):
            xs = list(ids)
            seed = sum(int(x) for x in xs) & 0xffffffff
            rng = random.Random(seed)
            rng.shuffle(xs)
            return xs
        record_set._greedy_similarity_order = _shuffle
        return

    if strategy == "weak_ordered":
        import random
        def _weak(ids, simi):
            xs = list(ids)
            if len(xs) <= 2:
                return xs
            avg = {r: sum(simi[r][r2] for r2 in xs if r2 != r) / max(1, len(xs) - 1)
                   for r in xs}
            ranked = sorted(xs, key=lambda r: -avg[r])
            n = len(ranked); k = max(1, n // 3)
            seed = sum(int(x) for x in xs) & 0xffffffff
            rng = random.Random(seed)
            hi = list(ranked[:k]);       rng.shuffle(hi)
            md = list(ranked[k:n - k]);  rng.shuffle(md)
            lo = list(ranked[n - k:]);   rng.shuffle(lo)
            return hi + md + lo
        record_set._greedy_similarity_order = _weak
        return

    raise ValueError(f"unknown strategy: {strategy}")

def run_one(dataset_name, strategy, sess_dir):
    data_rel, gt_rel = DATASETS[dataset_name]
    os.environ["DATASET_PATH"] = os.path.join(ROOT, data_rel)
    os.environ["GROUND_TRUTH_PATH"] = os.path.join(ROOT, gt_rel)
    pl = data_rel.lower()
    matched_best = False
    for key, thr in BEST_BLOCK.items():
        if key in pl:
            os.environ["BLOCK_THRESHOLD"] = str(thr); matched_best = True; break
    if not matched_best:
        os.environ.pop("BLOCK_THRESHOLD", None)
    os.environ.pop("MERGE_THRESHOLD", None)

    import importlib
    from llmcer import config as _cfg
    importlib.reload(_cfg)

    from llmcer.data_utils import get_ground_truth
    from llmcer.vectorization import cal_total_simi_vector
    from llmcer.clustering import lsh_block
    from llmcer.pipeline import run_blocks
    from llmcer.metrics import (calculate_acc, calculate_fp_measure,
                                calculate_pairwise_metrics, calculate_nmi,
                                calculate_ari, calculate_bcubed_metrics)

    install_ordering_patch(strategy)

    print(f"\n=== {dataset_name} | {strategy} ===", flush=True)
    t0 = time.time()

    gt = get_ground_truth(os.environ["GROUND_TRUTH_PATH"])
    vectors, simi, df = cal_total_simi_vector(os.environ["DATASET_PATH"])
    import numpy as _np
    sim_mean = float(_np.mean(simi)); sim_std = float(_np.std(simi))
    merge_override = None
    for key, thr in BEST_MERGE.items():
        if key in pl:
            merge_override = thr; break
    if merge_override is not None:
        merge_threshold = merge_override
    else:
        merge_threshold = min(sim_mean + 3.0 * sim_std, 0.90)
    os.environ["MERGE_THRESHOLD"] = str(merge_threshold)
    if not matched_best:
        os.environ["BLOCK_THRESHOLD"] = str(min(sim_mean + 2.5 * sim_std, 0.99))
    importlib.reload(_cfg)
    blocks = lsh_block(vectors, df, _cfg.BLOCK_THRESHOLD)
    print(f"  blocking: block={_cfg.BLOCK_THRESHOLD}  merge={merge_threshold:.3f}  "
          f"(mu={sim_mean:.3f}, sigma={sim_std:.3f})  -> {len(blocks)} blocks", flush=True)

    _parallel = os.environ.get("PIPELINE_PARALLEL", "1") == "1"
    pred, stats = run_blocks(vectors, simi, blocks, df,
                             S_s=_cfg.SET_SIZE, S_d=_cfg.SET_DIVERSITY,
                             parallel=_parallel)
    wall = time.time() - t0

    from llmcer.id_utils import get_id_column
    if hasattr(df, "iloc"):
        id_col = get_id_column(df)
        all_ids = df[id_col].tolist() if id_col else df.iloc[:, 0].tolist()
    else:
        all_ids = []
    gt_seen = {str(it).strip() for c in gt for it in c}
    for it in all_ids:
        if str(it).strip() not in gt_seen:
            gt.append([it])

    acc = calculate_acc(gt, pred)
    fp  = calculate_fp_measure(gt, pred)
    pw  = calculate_pairwise_metrics(gt, pred)
    nmi = calculate_nmi(gt, pred)
    ari = calculate_ari(gt, pred)
    bc  = calculate_bcubed_metrics(gt, pred)

    row = dict(
        dataset=dataset_name, strategy=strategy,
        n_records=len(vectors), n_blocks=len(blocks), n_pred_clusters=len(pred),
        acc=round(acc, 4), fp_measure=round(fp, 4),
        pairwise_f1=round(pw.get("f1", 0.0), 4),
        pairwise_p=round(pw.get("precision", 0.0), 4),
        pairwise_r=round(pw.get("recall", 0.0), 4),
        nmi=round(nmi, 4), ari=round(ari, 4),
        bcubed_f1=round(bc.get("f1", 0.0), 4),
        api_calls=stats["api_calls"], in_tokens=stats["in_tokens"],
        out_tokens=stats["out_tokens"], llm_time_s=round(stats["time"], 2),
        wall_s=round(wall, 2),
    )
    print(f"  ACC={row['acc']} FP={row['fp_measure']} PairF1={row['pairwise_f1']} "
          f"NMI={row['nmi']} ARI={row['ari']} BCubedF1={row['bcubed_f1']}", flush=True)
    print(f"  calls={row['api_calls']} in_tok={row['in_tokens']} "
          f"out_tok={row['out_tokens']} llm_time={row['llm_time_s']}s wall={row['wall_s']}s",
          flush=True)

    import pickle
    p = f"{sess_dir}/{dataset_name}__{strategy}.pkl"
    pickle.dump(dict(dataset=dataset_name, strategy=strategy, pred=pred,
                     gt=gt, stats=stats, row=row), open(p, "wb"))
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--sess-dir", default=None,
                    help="reuse an existing session dir; skip runs whose pkl "
                         "already exists (for resuming after interruption)")
    args = ap.parse_args()

    if args.sess_dir:
        sess_dir = args.sess_dir; os.makedirs(sess_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_dir = os.path.join(ROOT, "issue_experiments", "results",
                                "e2e_record_ordering", f"run_{ts}")
        os.makedirs(sess_dir, exist_ok=True)
    print(f"session: {sess_dir}", flush=True)

    rows = []
    import pickle as _pk
    for ds in args.datasets:
        for strat in args.strategies:
            pkl = os.path.join(sess_dir, f"{ds}__{strat}.pkl")
            if os.path.isfile(pkl) and os.path.getsize(pkl) > 0:
                try:
                    rows.append(_pk.load(open(pkl, "rb"))["row"])
                    print(f"[skip existing] {ds} / {strat}", flush=True)
                    continue
                except Exception:
                    pass
            try:
                rows.append(run_one(ds, strat, sess_dir))
            except Exception as e:
                import traceback
                print(f"  ERROR on {ds} / {strat}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    csv_p = os.path.join(sess_dir, "summary.csv")
    if rows:
        with open(csv_p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["strategy"]] = r
    txt_p = os.path.join(sess_dir, "summary.txt")
    with open(txt_p, "w") as fh:
        def w(s=""): print(s, flush=True); fh.write(s + "\n")
        for metric in ["acc", "fp_measure", "pairwise_f1", "nmi", "ari", "bcubed_f1"]:
            w("=" * 78)
            w(f"E2E RECORD-ORDERING (Table 18)   [{metric}]")
            w("=" * 78)
            w(f"{'Dataset':<16}{'simord':>10}{'weak':>10}{'randshuf':>12}{'best':>10}")
            w("-" * 78)
            ns = nw = nr = 0
            for ds in DATASETS:
                if ds not in by_ds: continue
                so = by_ds[ds].get("similarity_ordered", {}).get(metric)
                wo = by_ds[ds].get("weak_ordered", {}).get(metric)
                rs = by_ds[ds].get("random_shuffle", {}).get(metric)
                if so is None or wo is None or rs is None: continue
                m = max(so, wo, rs)
                best = "simord" if so == m else ("weak" if wo == m else "randshuf")
                if best == "simord": ns += 1
                elif best == "weak": nw += 1
                else: nr += 1
                w(f"{ds:<16}{so:>10.4f}{wo:>10.4f}{rs:>12.4f}{best:>10}")
            w("-" * 78)
            w(f"{'best count':<16}{ns:>10}{nw:>10}{nr:>12}")
            w("")
        w("=" * 78)
        w("End-to-end LLMCER pipeline (LSH blocking -> NRS -> in-context "
          "clustering -> CMR),\nrun reps=1 per strategy. Only the ordering of "
          "records inside each record set differs.")
    print(f"\n[wrote {csv_p}]", flush=True)
    print(f"[wrote {txt_p}]", flush=True)

if __name__ == "__main__":
    main()
