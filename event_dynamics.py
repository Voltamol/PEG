#!/usr/bin/env python3
"""
event_dynamics_v1.py

Verb-level event interpolation benchmark.

Evaluates:

    E1 -> ? -> E3

using

    score(E2) =
        log P(E2 | E1)
      + log P(E3 | E2)

Outputs:

- MRR (all queries)
- MRR (recoverable queries)
- Top-1
- Top-5
- Candidate coverage
- Recoverability
- Candidate set statistics
- Baselines

"""

import argparse
import math
import pickle
import random

from collections import defaultdict, Counter
from statistics import mean, median


# --------------------------------------------------
# LOAD
# --------------------------------------------------

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------
# STATE
# --------------------------------------------------

def get_state(event):
    return event["event_type"]


# --------------------------------------------------
# TRANSITIONS
# --------------------------------------------------

def build_transitions(events):

    states = [get_state(e) for e in events]

    trans = defaultdict(lambda: defaultdict(int))

    for i in range(len(states) - 1):
        trans[states[i]][states[i + 1]] += 1

    all_states = set(states)

    global_counts = Counter(states)

    return trans, all_states, global_counts


def p_next(nxt, cur, trans, vocab_size, alpha=1.0):

    numerator = trans[cur].get(nxt, 0) + alpha
    denominator = sum(trans[cur].values()) + alpha * vocab_size

    return numerator / denominator


# --------------------------------------------------
# CANDIDATES
# --------------------------------------------------

def successors(state, trans):
    return set(trans[state].keys())


def predecessors(state, trans):

    preds = set()

    for s, nexts in trans.items():
        if state in nexts:
            preds.add(s)

    return preds


def candidate_set(e1, e3, trans):

    return (
        successors(e1, trans)
        |
        predecessors(e3, trans)
    )


# --------------------------------------------------
# SCORING
# --------------------------------------------------

def score_candidate(
    candidate,
    e1,
    e3,
    trans,
    vocab_size,
    lambda_e3=1.0
):
    """
    score(E2) = log P(E2 | E1)  +  lambda_e3 * log P(E3 | E2)

    lambda_e3 = 1.0  -> original two-term score
    lambda_e3 = 0.0  -> e1-only score (ignores e3 entirely)
    0 < lambda_e3 < 1 -> partial weighting of the e3 term
    """

    p1 = p_next(
        candidate,
        e1,
        trans,
        vocab_size
    )

    log_score = math.log(p1 + 1e-12)

    if lambda_e3 != 0.0:

        p2 = p_next(
            e3,
            candidate,
            trans,
            vocab_size
        )

        log_score += lambda_e3 * math.log(p2 + 1e-12)

    return log_score


# --------------------------------------------------
# BASELINES
# --------------------------------------------------

def random_rank_baseline(
    true_state,
    all_states
):

    candidates = list(all_states)

    random.shuffle(candidates)

    try:
        return candidates.index(true_state) + 1
    except ValueError:
        return len(all_states) + 1


def random_rank_within_candidates_baseline(
    true_state,
    candidates
):
    """
    Random ranking restricted to the same candidate set the model
    actually ranks over. This is the fair baseline for MRR@All /
    MRR@Recoverable -- the model never ranks the full vocabulary,
    so comparing it to full-vocab random ranking overstates its lift.
    """

    cand_list = list(candidates)

    random.shuffle(cand_list)

    try:
        return cand_list.index(true_state) + 1
    except ValueError:
        return len(cand_list) + 1


def global_frequency_baseline(
    true_state,
    global_counts,
    candidates
):
    """
    Rank the candidate set by global frequency (most frequent first)
    and return the rank of true_state within that ordering.

    States with zero global count (shouldn't normally happen, since
    candidates are drawn from observed transitions) are pushed to the
    back, tied at count 0.

    Returns None only if the candidate set is empty.
    """

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda s: -global_counts.get(s, 0)
    )

    if true_state in ranked:
        return ranked.index(true_state) + 1

    # true_state wasn't even in the candidate set -> miss
    return len(ranked) + 1


def most_common_successor_baseline(
    e1,
    true_state,
    trans
):
    """
    Rank e1's observed successors by transition count (most frequent
    first) and return the rank of true_state within that ordering.

    If e1 has no observed successors, or true_state never follows e1
    in training, treat it as a miss: rank = (#successors + 1).
    """

    successors_counts = trans.get(e1)

    if not successors_counts:
        return None

    ranked = sorted(
        successors_counts.items(),
        key=lambda x: -x[1]
    )

    ranked_states = [s for s, _ in ranked]

    if true_state in ranked_states:
        return ranked_states.index(true_state) + 1

    return len(ranked_states) + 1


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    events,
    train_ratio=0.8,
    lambda_variants=None
):
    """
    lambda_variants: dict of {label: lambda_e3} to evaluate side by side,
    e.g. {"full (lambda=1.0)": 1.0, "e1-only (lambda=0.0)": 0.0}.
    All variants share the same train/test split and candidate sets,
    so differences in MRR are attributable purely to the scoring
    function, not to data variation.
    """

    if lambda_variants is None:
        lambda_variants = {"full (lambda=1.0)": 1.0}

    split_idx = int(
        len(events) * train_ratio
    )

    train_events = events[:split_idx]
    test_events = events[split_idx:]

    trans, all_states, global_counts = build_transitions(
        train_events
    )

    vocab_size = len(all_states)

    test_states = [
        get_state(e)
        for e in test_events
    ]

    # Per-variant accumulators
    variant_stats = {
        label: {
            "reciprocal_ranks": [],
            "reciprocal_ranks_recoverable": [],
            "top1": 0,
            "top5": 0,
            "recoverable": 0,
            "examples": []
        }
        for label in lambda_variants
    }

    candidate_generated = 0
    candidate_sizes = []

    random_rr = []
    random_rr_within_candidates = []
    global_rr = []
    mcs_rr = []

    total_queries = 0

    for i in range(
        1,
        len(test_states) - 1
    ):

        total_queries += 1

        e1 = test_states[i - 1]
        e2 = test_states[i]
        e3 = test_states[i + 1]

        candidates = candidate_set(
            e1,
            e3,
            trans
        )

        if not candidates:
            continue

        candidate_generated += 1

        candidate_sizes.append(
            len(candidates)
        )

        # Score every candidate once per variant, sharing the same
        # candidate set across variants for a fair comparison.
        for label, lam in lambda_variants.items():

            scored = []

            for cand in candidates:

                s = score_candidate(
                    cand,
                    e1,
                    e3,
                    trans,
                    vocab_size,
                    lambda_e3=lam
                )

                scored.append(
                    (cand, s)
                )

            scored.sort(
                key=lambda x: -x[1]
            )

            ranked_states = [
                c for c, _
                in scored
            ]

            stats = variant_stats[label]

            if e2 in ranked_states:

                stats["recoverable"] += 1

                rank = (
                    ranked_states.index(e2)
                    + 1
                )

                rr = 1.0 / rank

                stats["reciprocal_ranks"].append(rr)
                stats["reciprocal_ranks_recoverable"].append(rr)

                if rank == 1:
                    stats["top1"] += 1

                if rank <= 5:
                    stats["top5"] += 1

                if len(stats["examples"]) < 10:

                    stats["examples"].append({
                        "e1": e1,
                        "e2_true": e2,
                        "e3": e3,
                        "rank": rank,
                        "top_predictions":
                            ranked_states[:5]
                    })

        # Random ranking baseline (full vocabulary -- NOT a fair
        # comparison to the model, which only ranks the candidate set;
        # kept for reference / comparison to the candidate-scoped version)

        rr_rand = (
            1.0 /
            random_rank_baseline(
                e2,
                all_states
            )
        )

        random_rr.append(rr_rand)

        # Random ranking baseline, scoped to the same candidate set
        # the model ranks over -- this is the fair comparison

        rr_rand_cand = (
            1.0 /
            random_rank_within_candidates_baseline(
                e2,
                candidates
            )
        )

        random_rr_within_candidates.append(rr_rand_cand)

        # Global frequency baseline (full rank over candidate set)

        r = global_frequency_baseline(
            e2,
            global_counts,
            candidates
        )

        if r is not None:
            global_rr.append(1.0 / r)

        # Most common successor baseline (full rank over e1's successors)

        r = most_common_successor_baseline(
            e1,
            e2,
            trans
        )

        if r is not None:
            mcs_rr.append(1.0 / r)

    variant_results = {}

    for label, stats in variant_stats.items():

        mrr_all = (
            sum(stats["reciprocal_ranks"])
            / total_queries
            if total_queries
            else 0
        )

        mrr_recoverable = (
            sum(stats["reciprocal_ranks_recoverable"])
            / stats["recoverable"]
            if stats["recoverable"]
            else 0
        )

        variant_results[label] = {
            "mrr_all": mrr_all,
            "mrr_recoverable": mrr_recoverable,
            "top1": stats["top1"] / total_queries if total_queries else 0,
            "top5": stats["top5"] / total_queries if total_queries else 0,
            "recoverability": stats["recoverable"] / total_queries if total_queries else 0,
            "examples": stats["examples"]
        }

    return {

        "total_queries":
            total_queries,

        "candidate_coverage":
            candidate_generated
            / total_queries,

        "avg_candidate_size":
            mean(candidate_sizes)
            if candidate_sizes else 0,

        "median_candidate_size":
            median(candidate_sizes)
            if candidate_sizes else 0,

        "random_mrr":
            mean(random_rr)
            if random_rr else 0,

        "random_mrr_within_candidates":
            mean(random_rr_within_candidates)
            if random_rr_within_candidates else 0,

        "global_freq_mrr":
            mean(global_rr)
            if global_rr else 0,

        "mcs_mrr":
            mean(mcs_rr)
            if mcs_rr else 0,

        "variants":
            variant_results
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        required=True
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8
    )

    args = parser.parse_args()

    graph = load_graph(
        args.graph
    )

    events = graph["events"]

    print(
        f"Events: {len(events)}"
    )

    lambda_variants = {
        "Full (e1 + e3, lambda=1.0)": 1.0,
        "Partial (e1 + 0.5*e3)": 0.5,
        "e1-only (lambda=0.0)": 0.0,
    }

    results = evaluate(
        events,
        args.train_ratio,
        lambda_variants=lambda_variants
    )

    print("\n=== RESULTS ===\n")

    print(
        f"Total Queries:        {results['total_queries']}"
    )

    print(
        f"Candidate Coverage:   {results['candidate_coverage']:.4f}"
    )

    print("\nCandidate Statistics")

    print(
        f"Average Size:         {results['avg_candidate_size']:.2f}"
    )

    print(
        f"Median Size:          {results['median_candidate_size']:.2f}"
    )

    print("\n=== SCORING ABLATION ===")
    print("(same train/test split and candidate sets across all rows)\n")

    header = f"{'Variant':<32}{'MRR@All':>10}{'MRR@Rec':>10}{'Top-1':>8}{'Top-5':>8}{'Recov.':>9}"
    print(header)
    print("-" * len(header))

    for label, v in results["variants"].items():
        print(
            f"{label:<32}"
            f"{v['mrr_all']:>10.4f}"
            f"{v['mrr_recoverable']:>10.4f}"
            f"{v['top1']:>8.4f}"
            f"{v['top5']:>8.4f}"
            f"{v['recoverability']:>9.4f}"
        )

    print("\nBaselines (for the same row format, MRR@All-equivalent)")
    print("-" * len(header))

    print(
        f"{'Random (full vocab)':<32}{results['random_mrr']:>10.4f}"
        "   <- NOT comparable (ranks full vocab, not candidate set)"
    )

    print(
        f"{'Random (candidates)':<32}{results['random_mrr_within_candidates']:>10.4f}"
        "   <- fair floor for MRR@All rows above"
    )

    print(
        f"{'Global Frequency':<32}{results['global_freq_mrr']:>10.4f}"
        "   (ranked over candidate set, ignores e1 and e3)"
    )

    print(
        f"{'Most Common Successor':<32}{results['mcs_mrr']:>10.4f}"
        "   (ranked over e1's successors only, ignores e3)"
    )

    print(
        "\nRead this as: if 'e1-only' beats 'Full', the e3 term is hurting "
        "the ranking. If 'Full' barely beats 'Most Common Successor', the "
        "model isn't using e1/e3 context beyond what raw frequency gives you."
    )

    print("\nExamples (Full variant)")

    full_label = "Full (e1 + e3, lambda=1.0)"

    for ex in results["variants"][full_label]["examples"]:

        print(
            f"\n{ex['e1']} -> ? -> {ex['e3']}"
        )

        print(
            f"True: {ex['e2_true']} "
            f"(rank={ex['rank']})"
        )

        print(
            "Top predictions:",
            ex["top_predictions"]
        )


if __name__ == "__main__":
    main()