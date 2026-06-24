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


def candidate_set_union(e1, e3, trans):
    """
    Original candidate set: successors(e1) UNION predecessors(e3).
    Wide net -- includes states plausible from either direction alone,
    even if never jointly consistent with both e1 and e3.
    """

    return (
        successors(e1, trans)
        |
        predecessors(e3, trans)
    )


def candidate_set_intersection(e1, e3, trans):
    """
    Stricter candidate set: successors(e1) INTERSECT predecessors(e3).
    Every candidate is directly observed following e1 AND directly
    observed preceding e3. Much smaller, often empty -- trades
    coverage/recoverability for (hopefully) cleaner ranking when
    candidates do exist.
    """

    return (
        successors(e1, trans)
        &
        predecessors(e3, trans)
    )


# Backwards-compatible alias: existing callers using candidate_set()
# get the original union behavior unchanged.
def candidate_set(e1, e3, trans):
    return candidate_set_union(e1, e3, trans)


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
    lambda_variants=None,
    candidate_set_fns=None
):
    """
    lambda_variants: dict of {label: lambda_e3}, e.g.
        {"Full (lambda=1.0)": 1.0, "e1-only (lambda=0.0)": 0.0}.

    candidate_set_fns: dict of {label: fn(e1, e3, trans) -> set}, e.g.
        {"Union": candidate_set_union, "Intersection": candidate_set_intersection}.

    Every (candidate_set_label, lambda_label) combination is evaluated
    on the same train/test split, so differences are attributable to
    candidate set design and/or scoring weight, not data variation.

    NOTE: different candidate-set types can have different coverage
    (e.g. intersection is often empty when union is not), so
    "total_queries" is shared, but "candidate_generated" / coverage /
    recoverability are tracked PER candidate-set type, not globally.
    """

    if lambda_variants is None:
        lambda_variants = {"full (lambda=1.0)": 1.0}

    if candidate_set_fns is None:
        candidate_set_fns = {"Union": candidate_set_union}

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

    # Per (candidate_set_label, lambda_label) accumulators
    combo_stats = {
        (cs_label, lam_label): {
            "reciprocal_ranks": [],
            "reciprocal_ranks_recoverable": [],
            "top1": 0,
            "top5": 0,
            "recoverable": 0,
            "candidate_generated": 0,
            "candidate_sizes": [],
            "examples": []
        }
        for cs_label in candidate_set_fns
        for lam_label in lambda_variants
    }

    # Baselines are only well-defined for the candidate set actually
    # used to score -- track them per candidate_set_label too, since
    # union vs intersection candidates differ.
    baseline_stats = {
        cs_label: {
            "random_rr": [],
            "random_rr_within_candidates": [],
            "global_rr": [],
            "mcs_rr": []
        }
        for cs_label in candidate_set_fns
    }

    total_queries = 0

    for i in range(
        1,
        len(test_states) - 1
    ):

        total_queries += 1

        e1 = test_states[i - 1]
        e2 = test_states[i]
        e3 = test_states[i + 1]

        for cs_label, cs_fn in candidate_set_fns.items():

            candidates = cs_fn(
                e1,
                e3,
                trans
            )

            bstats = baseline_stats[cs_label]

            if not candidates:
                # No candidates under this candidate-set type for this
                # query: skip scoring/baselines for this type on this
                # query, but other candidate-set types still proceed.
                continue

            for lam_label in lambda_variants:
                combo_stats[(cs_label, lam_label)]["candidate_generated"] += 1
                combo_stats[(cs_label, lam_label)]["candidate_sizes"].append(len(candidates))

            # Score every candidate once per lambda variant, sharing
            # the same candidate set across variants for a fair
            # comparison.
            for lam_label, lam in lambda_variants.items():

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

                stats = combo_stats[(cs_label, lam_label)]

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

            # Baselines, scoped to this candidate_set type

            rr_rand = (
                1.0 /
                random_rank_baseline(
                    e2,
                    all_states
                )
            )

            bstats["random_rr"].append(rr_rand)

            rr_rand_cand = (
                1.0 /
                random_rank_within_candidates_baseline(
                    e2,
                    candidates
                )
            )

            bstats["random_rr_within_candidates"].append(rr_rand_cand)

            r = global_frequency_baseline(
                e2,
                global_counts,
                candidates
            )

            if r is not None:
                bstats["global_rr"].append(1.0 / r)

            r = most_common_successor_baseline(
                e1,
                e2,
                trans
            )

            if r is not None:
                bstats["mcs_rr"].append(1.0 / r)

    combo_results = {}

    for (cs_label, lam_label), stats in combo_stats.items():

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

        combo_results[(cs_label, lam_label)] = {
            "mrr_all": mrr_all,
            "mrr_recoverable": mrr_recoverable,
            "top1": stats["top1"] / total_queries if total_queries else 0,
            "top5": stats["top5"] / total_queries if total_queries else 0,
            "recoverability": stats["recoverable"] / total_queries if total_queries else 0,
            "candidate_coverage": stats["candidate_generated"] / total_queries if total_queries else 0,
            "avg_candidate_size": mean(stats["candidate_sizes"]) if stats["candidate_sizes"] else 0,
            "median_candidate_size": median(stats["candidate_sizes"]) if stats["candidate_sizes"] else 0,
            "examples": stats["examples"]
        }

    baseline_results = {}

    for cs_label, bstats in baseline_stats.items():

        baseline_results[cs_label] = {
            "random_mrr": mean(bstats["random_rr"]) if bstats["random_rr"] else 0,
            "random_mrr_within_candidates": mean(bstats["random_rr_within_candidates"]) if bstats["random_rr_within_candidates"] else 0,
            "global_freq_mrr": mean(bstats["global_rr"]) if bstats["global_rr"] else 0,
            "mcs_mrr": mean(bstats["mcs_rr"]) if bstats["mcs_rr"] else 0,
        }

    return {

        "total_queries":
            total_queries,

        "combos":
            combo_results,

        "baselines":
            baseline_results
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
        "Full (lambda=1.0)": 1.0,
        "Partial (lambda=0.5)": 0.5,
        "e1-only (lambda=0.0)": 0.0,
    }

    candidate_set_fns = {
        "Union": candidate_set_union,
        "Intersection": candidate_set_intersection,
    }

    results = evaluate(
        events,
        args.train_ratio,
        lambda_variants=lambda_variants,
        candidate_set_fns=candidate_set_fns
    )

    print(
        f"\nTotal Queries: {results['total_queries']}"
    )

    print("\n=== CANDIDATE SET COMPARISON ===\n")

    cs_header = f"{'Candidate Set':<16}{'Coverage':>10}{'Avg Size':>10}{'Med Size':>10}"
    print(cs_header)
    print("-" * len(cs_header))

    seen_cs = set()
    for (cs_label, lam_label), v in results["combos"].items():
        if cs_label in seen_cs:
            continue
        seen_cs.add(cs_label)
        print(
            f"{cs_label:<16}"
            f"{v['candidate_coverage']:>10.4f}"
            f"{v['avg_candidate_size']:>10.2f}"
            f"{v['median_candidate_size']:>10.2f}"
        )

    print("\n=== SCORING ABLATION (per candidate set) ===")
    print("(same train/test split within each candidate-set column)\n")

    header = f"{'Candidate Set':<14}{'Lambda Variant':<24}{'MRR@All':>10}{'MRR@Rec':>10}{'Top-1':>8}{'Top-5':>8}{'Recov.':>9}"
    print(header)
    print("-" * len(header))

    for cs_label in candidate_set_fns:
        for lam_label in lambda_variants:
            v = results["combos"][(cs_label, lam_label)]
            print(
                f"{cs_label:<14}"
                f"{lam_label:<24}"
                f"{v['mrr_all']:>10.4f}"
                f"{v['mrr_recoverable']:>10.4f}"
                f"{v['top1']:>8.4f}"
                f"{v['top5']:>8.4f}"
                f"{v['recoverability']:>9.4f}"
            )
        print()

    print("Baselines (per candidate set, MRR@All-equivalent)")
    print("-" * header.__len__())

    for cs_label in candidate_set_fns:
        b = results["baselines"][cs_label]
        print(f"\n[{cs_label}]")
        print(
            f"  {'Random (full vocab)':<28}{b['random_mrr']:>10.4f}"
            "  <- NOT comparable (ranks full vocab)"
        )
        print(
            f"  {'Random (candidates)':<28}{b['random_mrr_within_candidates']:>10.4f}"
            "  <- fair floor for this column"
        )
        print(
            f"  {'Global Frequency':<28}{b['global_freq_mrr']:>10.4f}"
            "  (ignores e1 and e3)"
        )
        print(
            f"  {'Most Common Successor':<28}{b['mcs_mrr']:>10.4f}"
            "  (ignores e3; NOTE: ranked over e1's full successor list,"
            " not restricted to this candidate set -- same across both columns)"
        )

    print(
        "\nRead this as: compare each candidate-set block's best lambda row "
        "against its own 'Random (candidates)' floor and against "
        "'Most Common Successor'. If Intersection's best row clears "
        "Most Common Successor by a real margin (even with lower recoverability), "
        "that's a structural win worth keeping over the wider Union set."
    )

    print("\nExamples (Union, Full variant)")

    for ex in results["combos"][("Union", "Full (lambda=1.0)")]["examples"]:

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

    print("\nExamples (Intersection, Full variant)")

    intersection_examples = results["combos"][("Intersection", "Full (lambda=1.0)")]["examples"]

    if not intersection_examples:
        print("(no recoverable examples captured -- intersection candidate "
              "sets may be empty or rarely contain the true e2 in this run)")
    else:
        for ex in intersection_examples:

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