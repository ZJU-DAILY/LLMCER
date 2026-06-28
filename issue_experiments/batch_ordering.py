"""
Batch CONSTRUCTION-strategy experiment  (reviewer issue #3).
============================================================

WHAT THIS ANSWERS
-----------------
The reviewer noted that §7.8 varies the order of RECORDS *within a single record
set*, and §7.7 varies only the BATCH SIZE; neither tests how the TASKS (record
sets) are grouped into a batched prompt. Our Algorithm 5 (CMR / batch building)
claims that packing SIMILAR tasks into the same batch helps. This experiment
tests exactly that claim with three batch-construction strategies:

  * similar     : group the most MUTUALLY-SIMILAR tasks into each batch
                  (this is what Algorithm 5 does).
  * random      : group tasks randomly.
  * dissimilar  : group the LEAST-similar tasks into each batch.

Each strategy partitions the SAME pool of tasks into batches of size K, sends
each batch as one prompt to the LLM, pools the per-task clusterings, and scores
ACC / FP against ground truth. Expected ordering, consistent with Algorithm 5:

        similar  >=  random  >=  dissimilar

It is ORTHOGONAL to §7.8 (records within a set) and §7.7 (batch size): here the
tasks and the batch size are fixed; only HOW tasks are grouped changes.

METRICS (per dataset, per strategy)
-----------------------------------
  ACC mean/std : end-to-end accuracy over `reps` repeats (LLM is non-det).
  FP  mean     : FP-measure.
The summary reports all three strategies side by side so the gradient is visible.

MODES
  --mock     deterministic oracle (no API key). The oracle clusters every task
             perfectly and independently, so all three strategies score the SAME
             -- this only smoke-tests the plumbing, it cannot show the gradient
             (which requires a real LLM with cross-task interference).
  (default)  real LLM via llmcer client (needs OPENAI_API_KEY in .env).

USAGE
  # smoke test (no key): expect identical ACC across strategies
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --mock --all
  # REAL experiment, all datasets (needs key in .env):
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --all
  # single dataset:
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --dataset cora

Outputs (results/batch_ordering/run_<mode>_<ts>/):
  <dataset>.log   full per-dataset trace (every batch, every strategy, every rep)
  summary.csv     machine-readable, one row per dataset
  summary.txt     human-readable table: ACC per strategy (this is what to report)
"""

import os
import sys
import csv
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "batch_ordering")

DATASETS = {
    "cora":           ("datasets/cora/cora.csv",                  "datasets/cora/gt.csv"),
    "citeseer":       ("datasets/citesheer/Citesheer_dblp.csv",   "datasets/citesheer/citesheer_gt.txt"),
    "google-DBLP":    ("datasets/google-DBLP/data.csv",           "datasets/google-DBLP/gt.csv"),
    "music20K":       ("datasets/music20K/music20K.csv",          "datasets/music20K/ground_truth.txt"),
    "sigmod":         ("datasets/sigmod/alaska.csv",              "datasets/sigmod/alaska_gt.csv"),
    "song":           ("datasets/song/songs.csv",                "datasets/song/gt.txt"),
    "affiliation":    ("datasets/affiliation/new_affi_data.csv",  "datasets/affiliation/new_mapping.csv"),
    "walmart-amazon": ("datasets/Walmart_Amazon/walmart_amazon.csv", "datasets/Walmart_Amazon/gt.csv"),
}

# Per-dataset LSH block_threshold used to form tasks (mirrors run_pipeline 'best').
BEST_BLOCK = {'cora': 0.90, 'song': 0.70, 'citesheer': 0.70, 'google-dblp': 0.70,
              'music20k': 0.70, 'amazon-google': 0.90, 'affiliation': 0.698,
              'walmart_amazon': 0.487}

STRATEGIES = ["similar", "random", "dissimilar"]

BATCH_PREPROMPT = (
    "You are given SEVERAL independent clustering tasks. For EACH task, classify "
    "its records into a two-dimensional list of groups by record ID. Treat the "
    "tasks as independent: a record in one task must never be grouped with a "
    "record from another task. Return the answer as a JSON object mapping each "
    "task id (e.g. \"T1\") to its two-dimensional list, with no extra text.\n"
)


# --------------------------- prompt / LLM ---------------------------------
def build_batched_prompt(task_order, task_records, df):
    from llmcer.data_utils import get_prompt_from_indices
    parts = [BATCH_PREPROMPT]
    for tid in task_order:
        parts.append(f"\n### Task T{tid} ###")
        parts.append(get_prompt_from_indices(task_records[tid], df))
    return "\n".join(parts)


def call_real_batch(prompt):
    from llmcer.llm_interaction import client
    from llmcer.config import OPENAI_MODEL
    # temperature omitted: the configured reasoning-style model rejects
    # temperature=0; the client already injects reasoning_effort='none'.
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in Entity Resolution clustering."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def parse_batch_reply(text, task_ids):
    import re, json
    out = {}
    try:
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {}
        for tid in task_ids:
            key = f"T{tid}"
            if key in obj:
                out[tid] = [[int(x) for x in grp] for grp in obj[key]]
    except Exception:
        pass
    return out


class MockBatchOracle:
    """Order/grouping-independent perfect clusterer (smoke test)."""
    def __init__(self, entity_of):
        self.entity_of = entity_of

    def run(self, task_order, task_records):
        result = {}
        for tid in task_order:
            groups = {}
            for r in task_records[tid]:
                groups.setdefault(self.entity_of[r], []).append(r)
            result[tid] = [sorted(v) for v in groups.values()]
        return result


# --------------------------- batch construction ---------------------------
def task_similarity(a, b, task_records, simi):
    """Similarity between two tasks = max pairwise record similarity."""
    return max(simi[i][j] for i in task_records[a] for j in task_records[b])


def build_batches(strategy, task_ids, task_sim, batch_size, seed):
    """
    Partition task_ids into batches of `batch_size` under a strategy.
      similar    : greedily seed a batch with a task, then add the most-similar
                   remaining tasks (Algorithm 5: similar tasks together).
      dissimilar : same but add the LEAST-similar remaining tasks.
      random     : deterministic pseudo-random chunking (varies with seed).
    task_sim[(a,b)] holds the precomputed task-task similarity.
    """
    ids = list(task_ids)
    if strategy == "random":
        # deterministic shuffle (no Math.random): rotate + stride by seed
        k = (seed * 7 + 3) % max(1, len(ids))
        ids = ids[k:] + ids[:k]
        ids = ids[::-1] if seed % 2 == 0 else ids
        return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    want_max = (strategy == "similar")
    remaining = set(ids)
    batches = []
    # seed order is deterministic (sorted) so similar/dissimilar are reproducible
    seed_order = sorted(remaining)
    while remaining:
        # pick the next unused seed task
        anchor = next(t for t in seed_order if t in remaining)
        batch = [anchor]; remaining.discard(anchor)
        while len(batch) < batch_size and remaining:
            # choose remaining task with max/min similarity to the current batch
            def score(t):
                return max(task_sim[(min(t, b), max(t, b))] for b in batch)
            pick = (max if want_max else min)(remaining, key=score)
            batch.append(pick); remaining.discard(pick)
        batches.append(batch)
    return batches


def labels_from_clusters(clusters, items):
    lab = {}
    for cid, c in enumerate(clusters):
        for r in c:
            lab[r] = cid
    nxt = len(clusters); out = []
    for r in items:
        out.append(lab[r] if r in lab else (nxt := nxt + 1))
    return out


def pooled_pred(per_task_all, pool_records):
    pred, seen = [], set()
    for per_task in per_task_all:
        for grps in per_task.values():
            for grp in grps:
                g = [int(x) for x in grp if int(x) in pool_records and int(x) not in seen]
                if g:
                    pred.append(g); seen.update(g)
    for r in pool_records:
        if r not in seen:
            pred.append([r])
    return pred


# --------------------------- per-dataset run ------------------------------
def run_dataset(name, args, session_dir):
    import numpy as np
    from llmcer.data_utils import get_ground_truth
    from llmcer.vectorization import cal_total_simi_vector
    from llmcer.record_set import next_record_set
    from llmcer.clustering import lsh_block
    from llmcer.metrics import calculate_acc, calculate_fp_measure

    data_rel, gt_rel = DATASETS[name]
    data_path = os.path.join(ROOT, data_rel)
    gt_path = os.path.join(ROOT, gt_rel)
    log_path = os.path.join(session_dir, f"{name}.log")
    log = open(log_path, "w", encoding="utf-8")

    def out(msg=""):
        print(msg); log.write(msg + "\n"); log.flush()

    mode = "mock" if args.mock else "real"
    out(f"# batch CONSTRUCTION strategy | dataset={name} mode={mode} "
        f"pool={args.pool} batch={args.batch} reps={args.reps}")

    full_gt = get_ground_truth(gt_path)
    vectors, simi, df = cal_total_simi_vector(data_path)
    n_all = len(vectors)

    entity_of = {}
    for ei, c in enumerate(full_gt):
        for r in c:
            if 0 <= int(r) < n_all:
                entity_of[int(r)] = ei
    nxt = len(full_gt)
    for r in range(n_all):
        if r not in entity_of:
            entity_of[r] = nxt; nxt += 1

    # ----- task pool: real LSH blocking + NRS, one record set per block --------
    pl = data_rel.lower()
    block_threshold, matched = None, None
    for key, thr in BEST_BLOCK.items():
        if key in pl:
            block_threshold, matched = thr, key; break
    if block_threshold is None:
        block_threshold = min(float(np.mean(simi)) + 2.5 * float(np.std(simi)), 0.99)
        matched = "dynamic mu+2.5sigma"
    _bt = os.environ.get("BLOCK_THRESHOLD")
    if _bt:
        block_threshold, matched = float(_bt), "env override"

    np.random.seed(0)
    blocks = lsh_block(vectors, df, block_threshold)
    usable = sorted([b for b in blocks if len(b) >= 2], key=lambda b: (-len(b), min(b)))
    out(f"  blocking: threshold={block_threshold} ({matched}) -> {len(blocks)} blocks, "
        f"{len(usable)} usable; building a pool of up to {args.pool} tasks")
    if len(usable) < args.batch * 2:
        out(f"ERROR: only {len(usable)} usable blocks; need >= {args.batch*2}. "
            f"Lower BLOCK_THRESHOLD.")
        log.close(); return None

    record_sets = []
    for b in usable[:args.pool]:
        rset, _ = next_record_set(b, vectors, simi, 9, 4)
        if rset:
            record_sets.append(rset)
    task_ids = list(range(len(record_sets)))
    task_records = {t: record_sets[t] for t in task_ids}
    pool_records = [r for t in task_ids for r in task_records[t]]
    out(f"  pool: {len(task_ids)} tasks, {len(pool_records)} records total")

    # GT restricted to the pool
    gt_pool = []
    for c in full_gt:
        m = [int(r) for r in c if int(r) in pool_records]
        if m:
            gt_pool.append(m)
    covered = {r for c in gt_pool for r in c}
    for r in pool_records:
        if r not in covered:
            gt_pool.append([r])

    # precompute task-task similarity
    task_sim = {}
    for a in range(len(task_ids)):
        for b in range(a + 1, len(task_ids)):
            task_sim[(a, b)] = task_similarity(a, b, task_records, simi)

    oracle = MockBatchOracle(entity_of) if args.mock else None

    out("")
    out(f"{'strategy':<12}{'rep':>4}{'#batches':>10}{'ACC':>9}{'FP':>9}")
    out("-" * 46)
    result = {}
    for strat in STRATEGIES:
        accs, fps = [], []
        for rep in range(args.reps):
            batches = build_batches(strat, task_ids, task_sim, args.batch, seed=rep)
            per_task_all = []
            for batch in batches:
                if args.mock:
                    per_task_all.append(oracle.run(batch, task_records))
                else:
                    text = call_real_batch(build_batched_prompt(batch, task_records, df))
                    per_task_all.append(parse_batch_reply(text, batch))
            pred = pooled_pred(per_task_all, pool_records)
            acc = calculate_acc(gt_pool, pred)
            fp = calculate_fp_measure(gt_pool, pred)
            accs.append(acc); fps.append(fp)
            out(f"{strat:<12}{rep:>4}{len(batches):>10}{acc:>9.4f}{fp:>9.4f}")
        result[strat] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                             fp_mean=float(np.mean(fps)))

    out("-" * 46)
    s_acc = result["similar"]["acc_mean"]
    r_acc = result["random"]["acc_mean"]
    d_acc = result["dissimilar"]["acc_mean"]
    out(f"ACC  similar={s_acc:.4f}  random={r_acc:.4f}  dissimilar={d_acc:.4f}")
    gradient_ok = s_acc >= r_acc >= d_acc
    out(f"gradient similar>=random>=dissimilar : {'YES' if gradient_ok else 'no'}")
    if args.mock:
        out("(mock oracle is grouping-independent, so the three are equal by "
            "construction — this only validates the harness.)")
    log.close()

    return dict(dataset=name, n_tasks=len(task_ids), n_records=len(pool_records),
                similar_acc=round(s_acc, 4), random_acc=round(r_acc, 4),
                dissimilar_acc=round(d_acc, 4),
                similar_fp=round(result["similar"]["fp_mean"], 4),
                random_fp=round(result["random"]["fp_mean"], 4),
                dissimilar_fp=round(result["dissimilar"]["fp_mean"], 4),
                gradient_ok=gradient_ok, log=os.path.relpath(log_path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DATASETS))
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--pool", type=int, default=9, help="number of tasks in the pool")
    ap.add_argument("--batch", type=int, default=3, help="tasks per batched prompt")
    ap.add_argument("--reps", type=int, default=3, help="repeats per strategy (LLM is non-det)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.mock:
        from llmcer.config import OPENAI_API_KEY
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
            print("ERROR: no OPENAI_API_KEY in .env — set it, or use --mock for a dry run.")
            return 2

    names = list(DATASETS) if (args.all or not args.dataset) else [args.dataset]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "mock" if args.mock else "real"
    session_dir = os.path.join(LOG_ROOT, f"run_{mode}_{ts}")
    os.makedirs(session_dir, exist_ok=True)

    print(f"=== batch construction-strategy ({mode}) -> {os.path.relpath(session_dir, ROOT)} ===")
    rows = []
    for nm in names:
        print(f"\n--- {nm} ---")
        try:
            r = run_dataset(nm, args, session_dir)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR on {nm}: {type(e).__name__}: {e}")

    csv_path = os.path.join(session_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "n_tasks", "n_records",
            "similar_acc", "random_acc", "dissimilar_acc",
            "similar_fp", "random_fp", "dissimilar_fp", "gradient_ok", "log"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    txt_path = os.path.join(session_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        def w(s=""):
            print(s); fh.write(s + "\n")
        w("")
        w("=" * 70)
        w(f"BATCH CONSTRUCTION-STRATEGY SUMMARY  ({mode} mode)   [ACC]")
        w("=" * 70)
        w(f"{'Dataset':<14}{'similar':>10}{'random':>10}{'dissimilar':>12}{'gradient':>10}")
        w("-" * 70)
        for r in rows:
            w(f"{r['dataset']:<14}{r['similar_acc']:>10.4f}{r['random_acc']:>10.4f}"
              f"{r['dissimilar_acc']:>12.4f}{('YES' if r['gradient_ok'] else 'no'):>10}")
        w("=" * 70)
        w("Strategy = how tasks are grouped into batched prompts:")
        w("  similar    : most mutually-similar tasks together (Algorithm 5)")
        w("  random     : random grouping")
        w("  dissimilar : least-similar tasks together")
        w("Expected, consistent with Algorithm 5:  similar >= random >= dissimilar")

    print(f"\nSummary CSV : {os.path.relpath(csv_path, ROOT)}")
    print(f"Summary TXT : {os.path.relpath(txt_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
