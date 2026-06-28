"""
Run ComEM's `selecting` strategy on the LLMCER datasets and report the SAME
metric set LLMCER's pipeline reports (ACC / FP-measure / NMI / ARI / Purity /
Inverse-Purity / BCubed) plus LLMCER-style efficiency stats (API calls / in-out
tokens / time / official-priced cost).

WHY / HOW
---------
ComEM is pairwise/clean-clean entity MATCHING; LLMCER is single-table dirty-ER
CLUSTERING. To make ComEM comparable as a baseline on LLMCER's datasets:
  1. blocking: reuse LLMCER's SBERT embeddings; for EVERY record (anchor) take
     the top-K most similar records as candidates (batched, no NxN blow-up).
  2. selecting: ComEM's faithful Selecting picks the one candidate that is the
     same entity (any chat model reachable via the OpenAI Python SDK, including
     api.openai.com and any OpenAI-compatible gateway).
  3. clustering: union the (anchor -> selected) edges; connected components are
     the predicted clusters (records in different components stay separate,
     matching LLMCER's hard-partition semantics).
  4. evaluation: LLMCER's own metric functions on (gt_clusters, pred_clusters).

This is a FULL-dataset run (no sampling): every record is an anchor.

CONFIGURATION
  OPENAI_API_KEY    -- required, your API key
  OPENAI_MODEL      -- chat model id (default: gpt-4o-mini)
  OPENAI_BASE_URL   -- optional; set to point at any OpenAI-compatible endpoint
  LLMCER_ROOT       -- path to a clone of the LLMCER repository whose datasets/
                       and llmcer/ modules we reuse. Defaults to ../LLMCER
                       relative to this ComEM checkout.

USAGE
  python src/eval_llmcer.py --dataset cora          # one dataset
  python src/eval_llmcer.py --all                   # all datasets, smallest-first
  python src/eval_llmcer.py --all --topk 10
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime

THIS = os.path.dirname(os.path.abspath(__file__))
COMEM_ROOT = os.path.dirname(THIS)
LLMCER_ROOT = os.environ.get(
    "LLMCER_ROOT",
    os.path.abspath(os.path.join(COMEM_ROOT, os.pardir, "LLMCER")),
)
if not os.path.isdir(LLMCER_ROOT):
    raise SystemExit(
        f"LLMCER_ROOT='{LLMCER_ROOT}' is not a directory. Clone "
        "https://github.com/ZJU-DAILY/LLMCER next to this checkout, or set "
        "the LLMCER_ROOT env var to point at your local LLMCER clone."
    )


# Optional .env loader: if LLMCER_ROOT/.env exists, populate OPENAI_* env vars
# from it (without overriding anything already in the environment). The OpenAI
# client itself reads OPENAI_API_KEY (and our utils.py reads OPENAI_BASE_URL).
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

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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


def run_dataset(name, data_rel, gt_rel, args, session_dir):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from llmcer.data_utils import get_ground_truth
    from llmcer.config import EMBEDDING_MODEL
    from llmcer.metrics import (calculate_acc, calculate_fp_measure, calculate_nmi,
                                calculate_ari, calculate_purity,
                                calculate_inverse_purity, calculate_bcubed_metrics)
    from tqdm.contrib.concurrent import thread_map
    from src.selecting import Selecting

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
    out(f"# ComEM selecting | dataset={name} model={MODEL} FULL n={n} topk={args.topk}")

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

    instances = [
        {"anchor": texts[i], "candidates": [texts[j] for j in cand[i]],
         "_idx": cand[i], "_anchor": i}
        for i in range(n)
    ]
    if args.limit:
        instances = instances[:args.limit]
        out(f"  [SMOKE] limiting to first {len(instances)} anchors")

    selector = Selecting(model_name=MODEL)
    out(f"  blocking done; running selecting on {len(instances)} anchors ({args.workers} workers)...")
    t0 = time.time()
    preds_lst = thread_map(selector, instances, max_workers=args.workers,
                           desc=f"{name}:selecting")
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
    nmi = calculate_nmi(gt, pred_clusters)
    ari = calculate_ari(gt, pred_clusters)
    purity = calculate_purity(gt, pred_clusters)
    inv_purity = calculate_inverse_purity(gt, pred_clusters)
    bcubed = calculate_bcubed_metrics(gt, pred_clusters)

    # ComEM-native pairwise (over the top-k candidate pairs) -- cross-check only
    cp = tp / (tp + fp) if (tp + fp) else 0.0
    cr = tp / (tp + fn) if (tp + fn) else 0.0
    cf = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0

    dec = selector.api_cost_decorator
    out("")
    out(f"  ACC={acc:.4f}  FP-measure={fp_measure:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}")
    out(f"  Purity={purity:.4f}  InvPurity={inv_purity:.4f}  "
        f"BCubed F1={bcubed['f1']:.4f} (P={bcubed['precision']:.4f} R={bcubed['recall']:.4f})")
    out(f"  [ComEM-native pairwise over candidates] P={cp:.4f} R={cr:.4f} F1={cf:.4f}")
    out(f"  API Calls={dec.n_calls}  InTokens={dec.prompt_tokens}  "
        f"OutTokens={dec.completion_tokens}  Cost=${dec.cost:.4f}  LLM Time={llm_time:.1f}s")
    log.close()

    return dict(
        dataset=name, n_records=n, n_pred_clusters=len(pred_clusters), n_gt_clusters=len(gt),
        acc=round(acc, 4), fp_measure=round(fp_measure, 4), nmi=round(nmi, 4),
        ari=round(ari, 4), purity=round(purity, 4), inv_purity=round(inv_purity, 4),
        bcubed_f1=round(bcubed["f1"], 4), bcubed_p=round(bcubed["precision"], 4),
        bcubed_r=round(bcubed["recall"], 4),
        pair_p=round(cp, 4), pair_r=round(cr, 4), pair_f1=round(cf, 4),
        api_calls=dec.n_calls, in_tokens=dec.prompt_tokens, out_tokens=dec.completion_tokens,
        cost_usd=round(dec.cost, 4), llm_time_s=round(llm_time, 1),
        log=os.path.relpath(log_path, COMEM_ROOT))


def main():
    ap = argparse.ArgumentParser()
    names = [d[0] for d in DATASETS]
    ap.add_argument("--dataset", default=None, choices=names)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap #anchors")
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

    fields = ["dataset", "n_records", "n_pred_clusters", "n_gt_clusters", "acc",
              "fp_measure", "nmi", "ari", "purity", "inv_purity", "bcubed_f1",
              "bcubed_p", "bcubed_r", "pair_p", "pair_r", "pair_f1", "api_calls",
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
        w(f"{'Dataset':<13}{'N':>7}{'ACC':>8}{'FP':>8}{'NMI':>8}{'ARI':>8}"
          f"{'Purity':>8}{'InvPur':>8}{'BCubF1':>8}{'Cost$':>8}{'Time s':>8}")
        w("-" * 100)
        for r in rows:
            w(f"{r['dataset']:<13}{r['n_records']:>7}{r['acc']:>8.4f}{r['fp_measure']:>8.4f}"
              f"{r['nmi']:>8.4f}{r['ari']:>8.4f}{r['purity']:>8.4f}{r['inv_purity']:>8.4f}"
              f"{r['bcubed_f1']:>8.4f}{r['cost_usd']:>8.4f}{r['llm_time_s']:>8.1f}")
        w("=" * 100)
        w("Metrics match LLMCER run_pipeline: ACC / FP-measure / NMI / ARI / Purity /")
        w("Inverse-Purity / BCubed. Efficiency: API calls / in-out tokens / cost / time")
        w("in summary.csv. FULL dataset (every record is an anchor); top-k blocking via")
        w("LLMCER's SBERT; predicted clusters = transitive closure of selecting edges.")

    print(f"\nSummary CSV : {os.path.relpath(os.path.join(session_dir,'summary.csv'), COMEM_ROOT)}")
    print(f"Summary TXT : {os.path.relpath(txt_path, COMEM_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
