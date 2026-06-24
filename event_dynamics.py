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
    vocab_size
):

    p1 = p_next(
        candidate,
        e1,
        trans,
        vocab_size
    )

    p2 = p_next(
        e3,
        candidate,
        trans,
        vocab_size
    )

    return (
        math.log(p1 + 1e-12)
        +
        math.log(p2 + 1e-12)
    )


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


def global_frequency_baseline(
    true_state,
    global_counts
):

    if not global_counts:
        return None

    prediction = global_counts.most_common(1)[0][0]

    if prediction == true_state:
        return 1

    return None


def most_common_successor_baseline(
    e1,
    true_state,
    trans
):

    if e1 not in trans:
        return None

    if not trans[e1]:
        return None

    prediction = max(
        trans[e1].items(),
        key=lambda x: x[1]
    )[0]

    if prediction == true_state:
        return 1

    return None


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    events,
    train_ratio=0.8
):

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

    reciprocal_ranks = []
    reciprocal_ranks_recoverable = []

    top1 = 0
    top5 = 0

    candidate_generated = 0
    recoverable = 0

    candidate_sizes = []

    random_rr = []
    global_rr = []
    mcs_rr = []

    examples = []

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

        scored = []

        for cand in candidates:

            s = score_candidate(
                cand,
                e1,
                e3,
                trans,
                vocab_size
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

        if e2 in ranked_states:

            recoverable += 1

            rank = (
                ranked_states.index(e2)
                + 1
            )

            rr = 1.0 / rank

            reciprocal_ranks.append(rr)
            reciprocal_ranks_recoverable.append(rr)

            if rank == 1:
                top1 += 1

            if rank <= 5:
                top5 += 1

            if len(examples) < 10:

                examples.append({
                    "e1": e1,
                    "e2_true": e2,
                    "e3": e3,
                    "rank": rank,
                    "top_predictions":
                        ranked_states[:5]
                })

        # Random ranking baseline

        rr_rand = (
            1.0 /
            random_rank_baseline(
                e2,
                all_states
            )
        )

        random_rr.append(rr_rand)

        # Global frequency baseline

        r = global_frequency_baseline(
            e2,
            global_counts
        )

        if r is not None:
            global_rr.append(1.0 / r)

        # Most common successor baseline

        r = most_common_successor_baseline(
            e1,
            e2,
            trans
        )

        if r is not None:
            mcs_rr.append(1.0 / r)

    mrr_all = (
        sum(reciprocal_ranks)
        / total_queries
        if total_queries
        else 0
    )

    mrr_recoverable = (
        sum(reciprocal_ranks_recoverable)
        / recoverable
        if recoverable
        else 0
    )

    return {

        "total_queries":
            total_queries,

        "candidate_coverage":
            candidate_generated
            / total_queries,

        "recoverability":
            recoverable
            / total_queries,

        "mrr_all":
            mrr_all,

        "mrr_recoverable":
            mrr_recoverable,

        "top1":
            top1 / total_queries,

        "top5":
            top5 / total_queries,

        "avg_candidate_size":
            mean(candidate_sizes)
            if candidate_sizes else 0,

        "median_candidate_size":
            median(candidate_sizes)
            if candidate_sizes else 0,

        "random_mrr":
            mean(random_rr)
            if random_rr else 0,

        "global_freq_mrr":
            mean(global_rr)
            if global_rr else 0,

        "mcs_mrr":
            mean(mcs_rr)
            if mcs_rr else 0,

        "examples":
            examples
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

    results = evaluate(
        events,
        args.train_ratio
    )

    print("\n=== RESULTS ===\n")

    print(
        f"Total Queries:        {results['total_queries']}"
    )

    print(
        f"Candidate Coverage:   {results['candidate_coverage']:.4f}"
    )

    print(
        f"Recoverability:       {results['recoverability']:.4f}"
    )

    print(
        f"MRR@All:              {results['mrr_all']:.4f}"
    )

    print(
        f"MRR@Recoverable:      {results['mrr_recoverable']:.4f}"
    )

    print(
        f"Top-1:                {results['top1']:.4f}"
    )

    print(
        f"Top-5:                {results['top5']:.4f}"
    )

    print("\nCandidate Statistics")

    print(
        f"Average Size:         {results['avg_candidate_size']:.2f}"
    )

    print(
        f"Median Size:          {results['median_candidate_size']:.2f}"
    )

    print("\nBaselines")

    print(
        f"Random Ranking MRR:   {results['random_mrr']:.4f}"
    )

    print(
        f"Global Frequency:     {results['global_freq_mrr']:.4f}"
    )

    print(
        f"Most Common Successor:{results['mcs_mrr']:.4f}"
    )

    print("\nExamples")

    for ex in results["examples"]:

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