"""
Deterministic "oracle" mock LLM for validating the pipeline without an API key.

The real pipeline calls the LLM to in-context cluster a set of records. For
testing we replace that call with an oracle that clusters records by their
ground-truth entity id. This lets us check that the *surrounding algorithms*
(NRS, MDG, CMR, metrics, block isolation) are correct end-to-end, independent of
LLM quality. A separate "noisy" oracle injects controlled misclusterings so we
can show MDG actually fires.
"""

import random


class OracleLLM:
    """Clusters record ids perfectly by a provided id->entity map."""

    def __init__(self, entity_of):
        self.entity_of = entity_of
        self.calls = 0

    def cluster(self, record_ids, df=None):
        self.calls += 1
        groups = {}
        for r in record_ids:
            e = self.entity_of[r]
            groups.setdefault(e, []).append(r)
        clusters = [sorted(v) for v in groups.values()]
        stats = dict(api_calls=1, time=0.0, tokens=10 * len(record_ids),
                     in_tokens=8 * len(record_ids), out_tokens=2 * len(record_ids))
        return clusters, stats


class NoisyOracleLLM:
    """
    Oracle that occasionally merges two different entities into one cluster
    (a misclustering), to exercise the MDG guardrail. `error_rate` is the
    probability a given call produces a wrong cluster.
    """

    def __init__(self, entity_of, error_rate=0.5, seed=0):
        self.entity_of = entity_of
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.calls = 0

    def cluster(self, record_ids, df=None):
        self.calls += 1
        groups = {}
        for r in record_ids:
            e = self.entity_of[r]
            groups.setdefault(e, []).append(r)
        clusters = [list(v) for v in groups.values()]

        if len(clusters) >= 2 and self.rng.random() < self.error_rate:
            clusters[0].extend(clusters[1])
            del clusters[1]

        stats = dict(api_calls=1, time=0.0, tokens=10 * len(record_ids),
                     in_tokens=8 * len(record_ids), out_tokens=2 * len(record_ids))
        return clusters, stats
