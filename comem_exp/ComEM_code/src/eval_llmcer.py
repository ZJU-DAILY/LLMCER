"""
Run ComEM's `selecting` strategy on the LLMCER datasets and report the SAME
metric set LLMCER's pipeline reports (ACC / FP-measure / Purity /
Inverse-Purity / BCubed) plus LLMCER-style efficiency stats (API calls / in-out
tokens / time / official-priced cost).

WHY / HOW
---------
ComEM is pairwise/clean-clean entity MATCHING; LLMCER is single-table dirty-ER
CLUSTERING. To make ComEM comparable as a baseline on LLMCER's datasets:
  1. blocking: reuse LLMCER's SBERT embeddings; for EVERY record (anchor) take
     the top-K most similar records as candidates (batched, no NxN blow-up).
  2. selecting: ComEM's faithful Selecting picks the one candidate that is the
     same entity (gpt-5.4-mini via the packyapi gateway).
  3. clustering: union the (anchor -> selected) edges; connected components are
     the predicted clusters (records in different components stay separate,
     matching LLMCER's hard-partition semantics).
  4. evaluation: LLMCER's own metric functions on (gt_clusters, pred_clusters).

This is a FULL-dataset run (no sampling): every record is an anchor.

USAGE
  python src/eval_llmcer.py --dataset cora          # one dataset
  python src/eval_llmcer.py --all                   # all 7, smallest-first
  python src/eval_llmcer.py --all --topk 10
"""

import os
import sys
import csv
import time
import argparse
import math
import hashlib
from datetime import datetime

THIS = os.path.dirname(os.path.abspath(__file__))
COMEM_ROOT = os.path.dirname(THIS)
# Root of the LLMCER repo (provides datasets/ and optionally a .env with the
# OpenAI key). Override via env var; defaults to the repo this folder lives in
# (ComEM_code is at <repo>/comem_exp/ComEM_code, so repo root is two levels up).
LLMCER_ROOT = os.environ.get(
    "LLMCER_ROOT",
    os.path.dirname(os.path.dirname(COMEM_ROOT)),
)

# --- make the gateway key/base_url available BEFORE importing the OpenAI client.
def _load_env(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(os.path.join(LLMCER_ROOT, ".env"))
sys.path.insert(0, COMEM_ROOT)
sys.path.insert(0, LLMCER_ROOT)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

# datasets, smallest-first (record counts:
# 1295/1808/2161/2260/4854/7626/9127/12010/19375)
DATASETS = [
    ("cora",           "datasets/cora/cora.csv",                  "datasets/cora/gt.csv"),
    ("Walmart_Amazon", "datasets/Walmart_Amazon/walmart_amazon.csv", "datasets/Walmart_Amazon/gt.csv"),
    ("Amazon-Google",  "datasets/Amazon-Google/amazon_google.csv",   "datasets/Amazon-Google/gt.csv"),
    ("affiliation", "datasets/affiliation/new_affi_data.csv", "datasets/affiliation/new_mapping.csv"),
    ("song",        "datasets/song/songs.csv",                "datasets/song/gt.txt"),
    ("google-DBLP", "datasets/google-DBLP/data.csv",          "datasets/google-DBLP/gt.csv"),
    ("citeseer",    "datasets/citesheer/Citesheer_dblp.csv",  "datasets/citesheer/citesheer_gt.txt"),
    ("sigmod",      "datasets/sigmod/alaska.csv",             "datasets/sigmod/alaska_gt.csv"),
    ("music20K",    "datasets/music20K/music20K.csv",         "datasets/music20K/ground_truth.txt"),
]

BEST_BLOCK = {
    "cora": 0.90,
    "song": 0.70,
    "citesheer": 0.70,
    "google-dblp": 0.70,
    "music20k": 0.70,
    "amazon-google": 0.90,
    "affiliation": 0.698,
    "walmart_amazon": 0.487,
}


def build_texts(df):
    """Replicate LLMCER's combine_attributes: drop the id column, join the rest."""
    from llmcer.id_utils import get_id_column
    id_col = get_id_column(df)

    def combine(row):
        if id_col and id_col in row:
            return " ".join(str(v) for k, v in row.items() if k != id_col)
        return " ".join(str(v) for v in row)

    return df.apply(combine, axis=1).tolist()


def topk_candidates(emb, k, batch=512):
    """For each row, indices of the top-k most similar OTHER rows. emb is L2-normalised."""
    import numpy as np
    n = emb.shape[0]
    out = []
    for s in range(0, n, batch):
        block = emb[s:s + batch]
        scores = block @ emb.T
        for r in range(block.shape[0]):
            i = s + r
            row = scores[r].copy()
            row[i] = -1.0
            kk = min(k, n - 1)
            idx = np.argpartition(-row, kk - 1)[:kk] if kk > 0 else np.array([], dtype=int)
            idx = idx[np.argsort(-row[idx])]
            out.append(idx.tolist())
    return out


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _representative_records(block, emb, keep):
    import numpy as np

    if keep >= len(block):
        return list(block)
    vecs = emb[block]
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    order = np.argsort(dists)
    return [int(block[i]) for i in order[:keep]]


def _global_downsample_anchors(anchor_indices, emb, max_total):
    if max_total <= 0 or len(anchor_indices) <= max_total:
        return list(anchor_indices)
    import numpy as np

    chosen = list(anchor_indices)
    vecs = emb[chosen]
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    order = np.argsort(dists)
    return [int(chosen[i]) for i in order[:max_total]]


def _text_hash_downsample_anchors(anchor_indices, texts, max_total):
    if max_total <= 0 or len(anchor_indices) <= max_total:
        return list(anchor_indices)

    def key(i):
        h = hashlib.sha1(texts[i].encode("utf-8", errors="ignore")).hexdigest()
        return (h, i)

    return sorted((int(i) for i in anchor_indices), key=key)[:max_total]


def select_anchor_indices(name, emb, df, args):
    if (
        args.anchor_fraction >= 0.999999
        and (args.max_anchors_per_block or 0) <= 0
    ):
        # Fast path: all records eligible as anchors; downstream downsampling
        # (max_anchors_total + rank_mode) is applied later by
        # finalize_anchor_indices. Return an empty-but-mutable meta dict so
        # callers can still record `n_anchors` / `max_anchors_total` etc.
        return list(range(len(df))), {
            "block_threshold": None,
            "block_threshold_source": "not_used",
            "n_blocks": None,
            "n_anchors": len(df),
            "anchor_fraction": args.anchor_fraction,
            "max_anchors_per_block": args.max_anchors_per_block,
            "max_anchors_total": args.max_anchors_total,
        }

    import numpy as np
    from llmcer.clustering import lsh_block

    path_key = None
    name_key = _norm_name = str(name).strip().lower().replace("_", "-")
    # Env var EVAL_FORCE_DYNAMIC=1 forces dynamic thresholds across all datasets
    # so that scalability-style comparisons (music10K/20K/30K/50K, etc.) share
    # one uniform blocking protocol instead of mixing hard-coded BEST_BLOCK
    # entries with dynamic mu+k*sigma.
    if os.environ.get("EVAL_FORCE_DYNAMIC", "").strip() not in ("", "0", "false", "False"):
        pass  # keep path_key = None -> dynamic path below
    else:
        for key, thr in BEST_BLOCK.items():
            if key in name_key:
                path_key = (thr, key)
                break
    if path_key is None:
        sim = emb @ emb.T
        block_threshold = min(float(np.mean(sim)) + 2.5 * float(np.std(sim)), 0.99)
        matched = "dynamic mu+2.5sigma"
    else:
        block_threshold, matched = path_key

    blocks = lsh_block(emb, df, block_threshold)
    blocks = sorted(blocks, key=lambda b: (-len(b), min(b)))
    chosen = []
    for block in blocks:
        keep = len(block)
        if args.anchor_fraction < 0.999999:
            keep = max(1, int(math.ceil(len(block) * args.anchor_fraction)))
        if (args.max_anchors_per_block or 0) > 0:
            keep = min(keep, args.max_anchors_per_block)
        chosen.extend(_representative_records(block, emb, keep))

    chosen = sorted(set(int(x) for x in chosen))
    return chosen, {
        "block_threshold": round(float(block_threshold), 6),
        "block_threshold_source": matched,
        "n_blocks": len(blocks),
        "n_anchors": len(chosen),
        "anchor_fraction": args.anchor_fraction,
        "max_anchors_per_block": args.max_anchors_per_block,
        "max_anchors_total": args.max_anchors_total,
    }


def finalize_anchor_indices(anchor_indices, emb, texts, args):
    if args.anchor_rank_mode == "centroid":
        return _global_downsample_anchors(anchor_indices, emb, args.max_anchors_total)
    if args.anchor_rank_mode == "text_hash":
        return _text_hash_downsample_anchors(anchor_indices, texts, args.max_anchors_total)
    raise ValueError(f"unknown anchor_rank_mode: {args.anchor_rank_mode}")


def run_dataset(name, data_rel, gt_rel, args, session_dir):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from llmcer.data_utils import get_ground_truth
    from llmcer.config import EMBEDDING_MODEL
    from llmcer.metrics import (calculate_acc, calculate_fp_measure, 
                                 calculate_purity,
                                calculate_inverse_purity, calculate_bcubed_metrics,
                                calculate_pairwise_metrics)
    from tqdm.contrib.concurrent import thread_map
    from src.selecting import Selecting
    from src.matching import Matching

    log_path = os.path.join(session_dir, f"{name}.log")
    log = open(log_path, "w", encoding="utf-8")

    def out(msg=""):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    data_path = os.path.join(LLMCER_ROOT, data_rel)
    gt_path = os.path.join(LLMCER_ROOT, gt_rel)

    full_gt = get_ground_truth(gt_path)
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="MacRoman")
    n = len(df)
    out(
        f"# ComEM selecting | dataset={name} model={MODEL} FULL n={n} topk={args.topk} "
        f"anchor_fraction={args.anchor_fraction:.4f} "
        f"max_anchors_per_block={args.max_anchors_per_block}"
    )

    # ground-truth clusters over record indices + entity map
    gt = [[int(r) for r in c if 0 <= int(r) < n] for c in full_gt]
    gt = [c for c in gt if c]
    entity_of = {}
    for ei, c in enumerate(gt):
        for r in c:
            entity_of[r] = ei
    covered = set(entity_of)
    nxt = len(gt)
    for r in range(n):
        if r not in covered:
            gt.append([r]); entity_of[r] = nxt; nxt += 1

    # blocking: SBERT embed all, batched top-k
    texts = build_texts(df)
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    cand = topk_candidates(emb, args.topk)
    anchor_indices, anchor_meta = select_anchor_indices(name, emb, df, args)
    anchor_indices = finalize_anchor_indices(anchor_indices, emb, texts, args)
    total_anchor_instances = len(anchor_indices)
    anchor_meta["n_anchors"] = total_anchor_instances
    if anchor_meta:
        out(
            f"  anchor selection: {anchor_meta['n_anchors']}/{n} anchors from "
            f"{anchor_meta['n_blocks']} blocks "
            f"(threshold={anchor_meta['block_threshold']} "
            f"{anchor_meta['block_threshold_source']}, "
            f"max_total={anchor_meta['max_anchors_total']}, "
            f"rank_mode={args.anchor_rank_mode})"
        )
    instances = [
        {"anchor": texts[i], "candidates": [texts[j] for j in cand[i]],
         "_idx": cand[i], "_anchor": i}
        for i in anchor_indices
    ]
    if args.limit:
        instances = instances[:args.limit]
        out(f"  [SMOKE] limiting to first {len(instances)} anchors")

    if args.strategy == "matching":
        selector = Matching(model_name=MODEL)
    else:
        selector = Selecting(model_name=MODEL)
    out(f"  blocking done; running {args.strategy} on {len(instances)} anchors "
        f"({args.workers} workers)...")
    t0 = time.time()
    preds_lst = thread_map(selector, instances, max_workers=args.workers,
                           desc=f"{name}:{args.strategy}")
    llm_time = time.time() - t0

    # predicted clusters via transitive closure of (anchor -> selected) edges
    uf = UF(n)
    edges = 0
    tp = fp = fn = tn = 0   # ComEM-native pairwise over candidate pairs (cheap)
    for inst, preds in zip(instances, preds_lst):
        i = inst["_anchor"]
        for p, is_match in enumerate(preds):
            j = inst["_idx"][p]
            gold = (entity_of[i] == entity_of[j])
            if is_match:
                uf.union(i, j); edges += 1
                if gold: tp += 1
                else: fp += 1
            else:
                if gold: fn += 1
                else: tn += 1
    comp = {}
    for r in range(n):
        comp.setdefault(uf.find(r), []).append(r)
    pred_clusters = list(comp.values())

    out(f"  selecting done in {llm_time:.1f}s; match edges={edges}; "
        f"pred clusters={len(pred_clusters)} (gt clusters={len(gt)})")

    # ---- LLMCER metric set ----
    acc = calculate_acc(gt, pred_clusters)
    fp_measure = calculate_fp_measure(gt, pred_clusters)
    
    purity = calculate_purity(gt, pred_clusters)
    inv_purity = calculate_inverse_purity(gt, pred_clusters)
    bcubed = calculate_bcubed_metrics(gt, pred_clusters)
    # Pairwise F1 (paper Equation 6) -- the "F1" column in the paper Table 5.
    # Distinct from FP-measure (Purity/InvPurity F1, Equation 5). Both are needed
    # to match the paper's two-column reporting.
    pw = calculate_pairwise_metrics(gt, pred_clusters)
    pw_p = pw.get("precision", 0.0); pw_r = pw.get("recall", 0.0); pw_f1 = pw.get("f1", 0.0)

    # ComEM-native pairwise (over the top-k candidate pairs) -- cross-check only
    cp = tp / (tp + fp) if (tp + fp) else 0.0
    cr = tp / (tp + fn) if (tp + fn) else 0.0
    cf = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0

    dec = selector.api_cost_decorator
    out("")
    out(f"  ACC={acc:.4f}  FP-measure={fp_measure:.4f} ")
    out(f"  Pairwise F1={pw_f1:.4f} (P={pw_p:.4f} R={pw_r:.4f})   <- paper's F1 column")
    out(f"  Purity={purity:.4f}  InvPurity={inv_purity:.4f}  "
        f"BCubed F1={bcubed['f1']:.4f} (P={bcubed['precision']:.4f} R={bcubed['recall']:.4f})")
    out(f"  [ComEM-native pairwise over candidates] P={cp:.4f} R={cr:.4f} F1={cf:.4f}")
    out(f"  API Calls={dec.n_calls}  InTokens={dec.prompt_tokens}  "
        f"OutTokens={dec.completion_tokens}  Cost=${dec.cost:.4f}  LLM Time={llm_time:.1f}s")
    log.close()

    return dict(
        dataset=name, n_records=n, n_pred_clusters=len(pred_clusters), n_gt_clusters=len(gt),
        n_anchor_instances=len(instances),
        total_anchor_instances=total_anchor_instances,
        acc=round(acc, 4), fp_measure=round(fp_measure, 4),
        purity=round(purity, 4), inv_purity=round(inv_purity, 4),
        bcubed_f1=round(bcubed["f1"], 4), bcubed_p=round(bcubed["precision"], 4),
        bcubed_r=round(bcubed["recall"], 4),
        pairwise_f1=round(pw_f1, 4), pairwise_p=round(pw_p, 4), pairwise_r=round(pw_r, 4),
        pair_p=round(cp, 4), pair_r=round(cr, 4), pair_f1=round(cf, 4),
        api_calls=dec.n_calls, in_tokens=dec.prompt_tokens, out_tokens=dec.completion_tokens,
        cost_usd=round(dec.cost, 4), llm_time_s=round(llm_time, 1),
        log=os.path.relpath(log_path, COMEM_ROOT))


def main():
    ap = argparse.ArgumentParser()
    names = [d[0] for d in DATASETS]
    ap.add_argument("--dataset", default=None, choices=names)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strategy", default="selecting",
                    choices=["selecting", "matching"],
                    help="ComEM strategy. selecting: 1 call per anchor picks "
                         "one candidate; matching: topk calls per anchor each "
                         "asking Y/N -> can produce multiple edges per anchor.")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap #anchors")
    ap.add_argument(
        "--anchor-fraction",
        type=float,
        default=1.0,
        help="fraction of records kept as anchors within each LSH block",
    )
    ap.add_argument(
        "--max-anchors-per-block",
        type=int,
        default=0,
        help="if >0, cap anchors per LSH block after anchor_fraction",
    )
    ap.add_argument(
        "--max-anchors-total",
        type=int,
        default=0,
        help="if >0, globally cap total anchors after block-level selection",
    )
    ap.add_argument(
        "--anchor-rank-mode",
        choices=["centroid", "text_hash"],
        default="centroid",
        help="how to pick final anchors when max_anchors_total is active",
    )
    args = ap.parse_args()

    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-") is False:
        print("ERROR: no OPENAI_API_KEY (expected in LLMCER .env).")
        return 2

    todo = DATASETS if (args.all or not args.dataset) else [d for d in DATASETS if d[0] == args.dataset]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(COMEM_ROOT, "results", "eval_llmcer", f"run_{ts}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"=== ComEM selecting on LLMCER datasets -> {os.path.relpath(session_dir, COMEM_ROOT)} ===")

    rows = []
    for name, data_rel, gt_rel in todo:
        print(f"\n--- {name} ---")
        try:
            r = run_dataset(name, data_rel, gt_rel, args, session_dir)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR on {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    fields = ["dataset", "n_records", "n_pred_clusters", "n_gt_clusters",
              "n_anchor_instances", "total_anchor_instances", "acc",
              "fp_measure", "purity", "inv_purity", "bcubed_f1",
              "bcubed_p", "bcubed_r", "pairwise_f1", "pairwise_p", "pairwise_r",
              "pair_p", "pair_r", "pair_f1", "api_calls",
              "in_tokens", "out_tokens", "cost_usd", "llm_time_s", "log"]
    with open(os.path.join(session_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    txt_path = os.path.join(session_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        def w(s=""):
            print(s); fh.write(s + "\n")
        w("")
        w("=" * 100)
        w(f"ComEM (selecting, {MODEL}) on LLMCER datasets -- FULL runs")
        w("=" * 100)
        w(f"{'Dataset':<13}{'N':>7}{'ACC':>8}{'FP':>8}"
          f"{'Purity':>8}{'InvPur':>8}{'BCubF1':>8}{'Cost$':>8}{'Time s':>8}")
        w("-" * 100)
        for r in rows:
            w(f"{r['dataset']:<13}{r['n_records']:>7}{r['acc']:>8.4f}{r['fp_measure']:>8.4f}"
              f"{r['purity']:>8.4f}{r['inv_purity']:>8.4f}"
              f"{r['bcubed_f1']:>8.4f}{r['cost_usd']:>8.4f}{r['llm_time_s']:>8.1f}")
        w("=" * 100)
        w("Metrics match LLMCER run_pipeline: ACC / FP-measure / Purity /")
        w("Inverse-Purity / BCubed. Efficiency: API calls / in-out tokens / cost / time")
        w("in summary.csv. FULL dataset (every record is an anchor); top-k blocking via")
        w("LLMCER's SBERT; predicted clusters = transitive closure of selecting edges.")

    print(f"\nSummary CSV : {os.path.relpath(os.path.join(session_dir,'summary.csv'), COMEM_ROOT)}")
    print(f"Summary TXT : {os.path.relpath(txt_path, COMEM_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
