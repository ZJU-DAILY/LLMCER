"""
Batch TASK-ordering experiment  (reviewer issue #3).
====================================================

WHAT THIS ANSWERS
-----------------
The reviewer noted that the paper's ordering study (§7.8) varies the order of
RECORDS *within a single record set*, and §7.7 varies only the BATCH SIZE.
Neither isolates the question actually raised: when several record sets (tasks)
are packed into ONE batched prompt, does the ORDER of those tasks within the
batch change the result -- i.e. do earlier tasks influence later ones?

This script isolates exactly that, and ONLY that:
  * a fixed set of K record sets (tasks) is built from each dataset,
  * each task's content is held FIXED,
  * only the ORDER of the K tasks inside the batched prompt is permuted
    (similarity-ordered, its reverse, and several shuffles),
  * the LLM clusters each permutation; we measure whether the produced
    clustering changes.

It is ORTHOGONAL to §7.8 (records within a set) and §7.7 (batch size).

METRICS (per dataset, across the permutations of task order)
------------------------------------------------------------
  ACC mean/std   : end-to-end accuracy, averaged over orderings (+ its std).
  FP  mean/std   : FP-measure, averaged over orderings (+ its std).
  ARI_stability  : mean pairwise Adjusted Rand Index between the clusterings
                   produced by DIFFERENT orderings. ARI = 1.0 means every
                   ordering yields the IDENTICAL clustering (task order has NO
                   effect); < 1.0 quantifies how much the clustering changes
                   with task order. THIS is the direct answer to the reviewer.
  flip_rate      : fraction of records whose assigned cluster differs between
                   the similarity ordering and at least one other ordering.

INTERPRETATION
  ARI_stability ~ 1.0 and std(ACC) ~ 0  -> task order is irrelevant (robust);
  ARI_stability noticeably < 1.0        -> a real ordering effect to report.

MODES
  --mock     deterministic oracle, NO API key (smoke-tests the harness; the
             oracle is order-independent so ARI_stability MUST be 1.0).
  (default)  real LLM via llmcer client (needs OPENAI_API_KEY in .env).

USAGE  (experimenter: just run the last line)
  # smoke test, no key:
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --mock --all
  # REAL experiment, all datasets (needs a working key in .env):
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --all
  # single dataset:
  .venv/Scripts/python.exe issue_experiments/batch_ordering.py --dataset cora

Outputs (per session folder results/batch_ordering/run_<ts>/):
  <dataset>.log         full per-dataset trace
  summary.csv           one row per dataset: ACC/FP mean+std, ARI_stability, flip_rate
  summary.txt           human-readable table (this is what you paste/report)
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
    "cora":        ("datasets/cora/cora.csv",                "datasets/cora/gt.csv"),
    "citeseer":    ("datasets/citesheer/Citesheer_dblp.csv", "datasets/citesheer/citesheer_gt.txt"),
    "google-DBLP": ("datasets/google-DBLP/data.csv",         "datasets/google-DBLP/gt.csv"),
    "music20K":    ("datasets/music20K/music20K.csv",        "datasets/music20K/ground_truth.txt"),
    "sigmod":      ("datasets/sigmod/alaska.csv",            "datasets/sigmod/alaska_gt.csv"),
    "song":        ("datasets/song/songs.csv",              "datasets/song/gt.txt"),
    "affiliation": ("datasets/affiliation/new_affi_data.csv","datasets/affiliation/new_mapping.csv"),
}

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
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
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
    """Order-independent perfect clusterer (smoke test)."""
    def __init__(self, entity_of):
        self.entity_of = entity_of

    def run(self, task_order, task_records, task_ids):
        result = {}
        for tid in task_ids:
            groups = {}
            for r in task_records[tid]:
                groups.setdefault(self.entity_of[r], []).append(r)
            result[tid] = [sorted(v) for v in groups.values()]
        return result


# --------------------------- helpers --------------------------------------
def make_permutations(task_ids, sim_between, n_random, seed_base):
    if len(task_ids) < 2:
        return [("similarity", list(task_ids))]
    order = [task_ids[0]]
    remaining = set(task_ids[1:])
    while remaining:
        last = order[-1]
        nxt = max(remaining, key=lambda t: sim_between(last, t))
        order.append(nxt); remaining.discard(nxt)
    perms = [("similarity", list(order)), ("reverse", list(reversed(order)))]
    for i in range(n_random):
        seq = list(task_ids)
        k = (seed_base + i * 7 + 3) % len(seq)
        seq = seq[k:] + seq[:k]
        if i % 2 == 0:
            seq = seq[::-1]
        perms.append((f"random{i+1}", seq))
    return perms


def labels_from_clusters(clusters, items):
    """item -> cluster-id label vector aligned to `items` order."""
    lab = {}
    for cid, c in enumerate(clusters):
        for r in c:
            lab[r] = cid
    nxt = len(clusters)
    out = []
    for r in items:
        if r in lab:
            out.append(lab[r])
        else:
            out.append(nxt); nxt += 1
    return out


def pooled_pred(per_task, task_ids, batch_records):
    pred, seen = [], set()
    for t in task_ids:
        for grp in per_task.get(t, []):
            g = [int(x) for x in grp if int(x) in batch_records]
            if g:
                pred.append(g); seen.update(g)
    for r in batch_records:
        if r not in seen:
            pred.append([r])
    return pred


# --------------------------- per-dataset run ------------------------------
def run_dataset(name, args, session_dir):
    import numpy as np
    from llmcer.data_utils import get_ground_truth
    from llmcer.vectorization import cal_total_simi_vector
    from llmcer.record_set import create_record_sets
    from llmcer.metrics import calculate_acc, calculate_fp_measure
    from sklearn.metrics import adjusted_rand_score

    data_rel, gt_rel = DATASETS[name]
    data_path = os.path.join(ROOT, data_rel)
    gt_path = os.path.join(ROOT, gt_rel)
    log_path = os.path.join(session_dir, f"{name}.log")
    log = open(log_path, "w", encoding="utf-8")

    def out(msg=""):
        print(msg); log.write(msg + "\n"); log.flush()

    mode = "mock" if args.mock else "real"
    out(f"# batch task-ordering | dataset={name} mode={mode} "
        f"tasks={args.tasks} records~{args.records} perms={args.perms}")

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

    rng = np.random.RandomState(0)
    order = list(range(len(full_gt)))
    rng.shuffle(order)
    keep = set()
    for i in order:
        if len(keep) >= args.records:
            break
        keep.update(int(r) for r in full_gt[i] if 0 <= int(r) < n_all)
    block = sorted(keep)
    if len(block) < args.tasks * 2:
        out(f"ERROR: sampled block too small ({len(block)}). Increase --records.")
        log.close()
        return None

    record_sets = [rs for rs in create_record_sets(block, vectors, simi, 9, 4) if rs]
    record_sets = record_sets[:args.tasks]
    task_ids = list(range(len(record_sets)))
    task_records = {t: record_sets[t] for t in task_ids}
    for t in task_ids:
        out(f"  Task T{t}: {len(task_records[t])} records {task_records[t]}")

    batch_records = [r for t in task_ids for r in task_records[t]]
    gt_batch = []
    for c in full_gt:
        m = [int(r) for r in c if int(r) in batch_records]
        if m:
            gt_batch.append(m)
    covered = {r for c in gt_batch for r in c}
    for r in batch_records:
        if r not in covered:
            gt_batch.append([r])

    def sim_between(a, b):
        return max(simi[i][j] for i in task_records[a] for j in task_records[b])

    perms = make_permutations(task_ids, sim_between, args.perms, seed_base=len(block))
    oracle = MockBatchOracle(entity_of) if args.mock else None

    out("")
    out(f"{'permutation':<13}{'order':<22}{'ACC':>8}{'FP':>8}")
    out("-" * 51)
    accs, fps, label_vecs = [], [], []
    for label, torder in perms:
        if args.mock:
            per_task = oracle.run(torder, task_records, task_ids)
        else:
            text = call_real_batch(build_batched_prompt(torder, task_records, df))
            per_task = parse_batch_reply(text, task_ids)
        pred = pooled_pred(per_task, task_ids, batch_records)
        acc = calculate_acc(gt_batch, pred)
        fp = calculate_fp_measure(gt_batch, pred)
        accs.append(acc); fps.append(fp)
        label_vecs.append(labels_from_clusters(pred, batch_records))
        out(f"{label:<13}{str(torder):<22}{acc:>8.4f}{fp:>8.4f}")

    acc_arr, fp_arr = np.array(accs), np.array(fps)
    # cross-ordering clustering stability: mean pairwise ARI between orderings
    aris = []
    for i in range(len(label_vecs)):
        for j in range(i + 1, len(label_vecs)):
            aris.append(adjusted_rand_score(label_vecs[i], label_vecs[j]))
    ari_stab = float(np.mean(aris)) if aris else 1.0
    # flip rate vs the similarity ordering (index 0)
    base = label_vecs[0]
    flips = 0
    for k in range(len(batch_records)):
        if any(label_vecs[v][k] != base[k] for v in range(1, len(label_vecs))):
            # cluster-id labels are not comparable directly; use co-membership
            pass
    # co-membership flip rate: fraction of record PAIRS whose same/diff-cluster
    # status changes between the base ordering and any other ordering
    npairs = 0; flipped = 0
    for a in range(len(batch_records)):
        for b in range(a + 1, len(batch_records)):
            npairs += 1
            base_same = (base[a] == base[b])
            if any((label_vecs[v][a] == label_vecs[v][b]) != base_same
                   for v in range(1, len(label_vecs))):
                flipped += 1
    flip_rate = flipped / npairs if npairs else 0.0

    out("-" * 51)
    out(f"ACC  mean={acc_arr.mean():.4f}  std={acc_arr.std():.4f}  "
        f"range=[{acc_arr.min():.4f},{acc_arr.max():.4f}]")
    out(f"FP   mean={fp_arr.mean():.4f}  std={fp_arr.std():.4f}")
    out(f"ARI_stability (mean pairwise ARI between orderings) = {ari_stab:.4f}")
    out(f"pair flip_rate vs similarity ordering = {flip_rate:.4f}")
    verdict = ("task order IRRELEVANT (ARI~1, std~0)" if ari_stab > 0.99 and acc_arr.std() < 0.01
               else "task order has a MEASURABLE effect — report it")
    out(f"Verdict: {verdict}")
    log.close()

    return dict(dataset=name, n_tasks=len(task_ids), n_records=len(batch_records),
                acc_mean=round(float(acc_arr.mean()), 4), acc_std=round(float(acc_arr.std()), 4),
                fp_mean=round(float(fp_arr.mean()), 4), fp_std=round(float(fp_arr.std()), 4),
                ari_stability=round(ari_stab, 4), flip_rate=round(flip_rate, 4),
                verdict=verdict, log=os.path.relpath(log_path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DATASETS))
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--records", type=int, default=60)
    ap.add_argument("--perms", type=int, default=5)
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

    print(f"=== batch task-ordering ({mode}) -> {os.path.relpath(session_dir, ROOT)} ===")
    rows = []
    for nm in names:
        print(f"\n--- {nm} ---")
        try:
            r = run_dataset(nm, args, session_dir)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR on {nm}: {type(e).__name__}: {e}")

    # summary.csv + summary.txt
    csv_path = os.path.join(session_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "n_tasks", "n_records",
            "acc_mean", "acc_std", "fp_mean", "fp_std", "ari_stability",
            "flip_rate", "verdict", "log"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    txt_path = os.path.join(session_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        def w(s=""):
            print(s); fh.write(s + "\n")
        w("")
        w("=" * 78)
        w(f"BATCH TASK-ORDERING SUMMARY  ({mode} mode)")
        w("=" * 78)
        w(f"{'Dataset':<13}{'ACC mean':>9}{'ACC std':>9}{'FP mean':>9}"
          f"{'ARI_stab':>10}{'flip':>8}")
        w("-" * 78)
        for r in rows:
            w(f"{r['dataset']:<13}{r['acc_mean']:>9.4f}{r['acc_std']:>9.4f}"
              f"{r['fp_mean']:>9.4f}{r['ari_stability']:>10.4f}{r['flip_rate']:>8.4f}")
        w("=" * 78)
        w("ARI_stab = mean pairwise Adjusted Rand Index between the clusterings")
        w("           produced by different task orderings. 1.0 = task order has")
        w("           NO effect on the clustering. std(ACC)~0 confirms the same.")
        w("flip     = fraction of record pairs whose same/different-cluster status")
        w("           changes when the task order changes.")

    print(f"\nSummary CSV : {os.path.relpath(csv_path, ROOT)}")
    print(f"Summary TXT : {os.path.relpath(txt_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
