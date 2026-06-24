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

def get_state_verb_only(event):
    """
    Level A (original): state is just the verb lemma.
    """
    return event["event_type"]


def get_state_verb_roles(event, min_confidence=0.0, include_polarity=False):
    """
    Level B: state is the verb plus the sorted set of distinct semantic
    roles present on the event, e.g. "give(AGENT,PATIENT,RECIPIENT)".

    - Deduplicates role labels (a conjunct AGENT doesn't add a second
      AGENT slot to the state -- presence, not count, is what matters
      here).
    - min_confidence filters out low-confidence role attachments
      (e.g. weak control/conj-inherited AGENTs) before building the
      state, so speculative parses don't fragment the state space.
    - POLARITY is excluded by default since it's closer to a modifier
      on the event (negation) than a participant role, and including
      it would double the state space for every verb without clearly
      helping interpolation; set include_polarity=True to include it.
    """

    verb = event["event_type"]

    role_labels = set()

    for r in event.get("roles", []):

        role = r.get("role")

        if role is None:
            continue

        if role == "POLARITY" and not include_polarity:
            continue

        confidence = r.get("confidence", 1.0)

        if confidence < min_confidence:
            continue

        role_labels.add(role)

    if not role_labels:
        return f"{verb}()"

    return f"{verb}({','.join(sorted(role_labels))})"


# Backwards-compatible default: existing callers of get_state() keep
# the original verb-only behavior unchanged.
def get_state(event):
    return get_state_verb_only(event)


def describe_state_vocab(events, state_fn, label=""):
    """
    Quick diagnostic: how many distinct states does this state_fn
    produce over these events, and how skewed is the distribution?
    Useful for sanity-checking that verb+roles doesn't fragment the
    vocabulary so much that every state becomes a singleton.
    """

    states = [state_fn(e) for e in events]
    counts = Counter(states)

    vocab_size = len(counts)
    singleton_count = sum(1 for c in counts.values() if c == 1)

    print(f"\n[State vocab: {label or state_fn.__name__}]")
    print(f"  Distinct states: {vocab_size}")
    print(f"  Singleton states (count=1): {singleton_count} "
          f"({100 * singleton_count / vocab_size:.1f}% of vocab)" if vocab_size else "")
    print(f"  Top 10 states: {counts.most_common(10)}")

    return counts


# --------------------------------------------------
# TRANSITIONS
# --------------------------------------------------

def build_transitions(events, state_fn=get_state_verb_only):

    states = [state_fn(e) for e in events]

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
    candidate_set_fns=None,
    state_fn=get_state_verb_only
):
    """
    lambda_variants: dict of {label: lambda_e3}, e.g.
        {"Full (lambda=1.0)": 1.0, "e1-only (lambda=0.0)": 0.0}.

    candidate_set_fns: dict of {label: fn(e1, e3, trans) -> set}, e.g.
        {"Union": candidate_set_union, "Intersection": candidate_set_intersection}.

    state_fn: fn(event) -> state string. Determines the granularity of
        states, e.g. get_state_verb_only (Level A) or
        get_state_verb_roles (Level B). This is the main lever for the
        "does richer state granularity help" question -- everything
        else (candidate sets, scoring weights) stays comparable across
        state_fn choices because the harness is identical.

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
        train_events,
        state_fn=state_fn
    )

    vocab_size = len(all_states)

    test_states = [
        state_fn(e)
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

    parser.add_argument(
        "--min-role-confidence",
        type=float,
        default=0.0,
        help="Minimum role confidence to include a role in the verb+roles "
             "state (Level B). 0.0 keeps all roles regardless of confidence."
    )

    args = parser.parse_args()

    graph = load_graph(
        args.graph
    )

    events = graph["events"]

    print(
        f"Events: {len(events)}"
    )

    # --------------------------------------------------
    # State vocab diagnostics (cheap, run before any eval)
    # --------------------------------------------------

    def verb_roles_state(event):
        return get_state_verb_roles(
            event,
            min_confidence=args.min_role_confidence
        )

    describe_state_vocab(events, get_state_verb_only, label="Level A: verb-only")
    describe_state_vocab(events, verb_roles_state, label="Level B: verb+roles")

    # --------------------------------------------------
    # Shared experiment grid
    # --------------------------------------------------

    lambda_variants = {
        "Full (lambda=1.0)": 1.0,
        "Partial (lambda=0.5)": 0.5,
        "e1-only (lambda=0.0)": 0.0,
    }

    candidate_set_fns = {
        "Union": candidate_set_union,
        "Intersection": candidate_set_intersection,
    }

    state_levels = {
        "Level A (verb-only)": get_state_verb_only,
        "Level B (verb+roles)": verb_roles_state,
    }

    all_results = {}

    for level_label, state_fn in state_levels.items():

        all_results[level_label] = evaluate(
            events,
            args.train_ratio,
            lambda_variants=lambda_variants,
            candidate_set_fns=candidate_set_fns,
            state_fn=state_fn
        )

    total_queries = next(iter(all_results.values()))["total_queries"]

    print(
        f"\nTotal Queries: {total_queries}"
    )

    # --------------------------------------------------
    # Candidate set stats per state level
    # --------------------------------------------------

    for level_label, results in all_results.items():

        print(f"\n=== CANDIDATE SET COMPARISON [{level_label}] ===\n")

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

    # --------------------------------------------------
    # Full scoring ablation per state level
    # --------------------------------------------------

    header = f"{'Candidate Set':<14}{'Lambda Variant':<24}{'MRR@All':>10}{'MRR@Rec':>10}{'Top-1':>8}{'Top-5':>8}{'Recov.':>9}"

    for level_label, results in all_results.items():

        print(f"\n=== SCORING ABLATION [{level_label}] ===")
        print("(same train/test split within each candidate-set column)\n")

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

        print(f"Baselines [{level_label}] (per candidate set, MRR@All-equivalent)")
        print("-" * len(header))

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
                "  (ignores e3)"
            )

    # --------------------------------------------------
    # Head-to-head: Level A vs Level B, best config each
    # --------------------------------------------------

    print("\n\n=== HEAD-TO-HEAD: Level A vs Level B ===")
    print("(best MRR@All config found for each state level, vs. that level's own baselines)\n")

    summary_header = f"{'State Level':<22}{'Best Config':<34}{'MRR@All':>10}  {'Best Baseline':<26}{'Baseline MRR':>14}  {'Delta':>8}"
    print(summary_header)
    print("-" * len(summary_header))

    for level_label, results in all_results.items():

        best_combo_key = max(
            results["combos"],
            key=lambda k: results["combos"][k]["mrr_all"]
        )
        best_mrr = results["combos"][best_combo_key]["mrr_all"]
        best_config_label = f"{best_combo_key[0]} / {best_combo_key[1]}"

        # Best baseline across both candidate sets for this level
        best_baseline_label = None
        best_baseline_mrr = -1.0
        for cs_label, b in results["baselines"].items():
            for bname, bval in [
                ("Global Frequency", b["global_freq_mrr"]),
                ("Most Common Successor", b["mcs_mrr"]),
            ]:
                if bval > best_baseline_mrr:
                    best_baseline_mrr = bval
                    best_baseline_label = f"{bname} [{cs_label}]"

        delta = best_mrr - best_baseline_mrr

        print(
            f"{level_label:<22}"
            f"{best_config_label:<34}"
            f"{best_mrr:>10.4f}  "
            f"{best_baseline_label:<26}"
            f"{best_baseline_mrr:>14.4f}  "
            f"{delta:>+8.4f}"
        )

    print(
        "\nThe Delta column is the number that's actually comparable across "
        "state levels (raw MRR is not, since vocab size and chance levels "
        "differ). A more positive Delta for Level B than Level A means "
        "verb+roles is beating its own naive baseline by more than verb-only "
        "beats its own -- that's the real evidence richer states help, not "
        "just a difference in absolute MRR."
    )

    print(
        "\nRead this as: if Level B's best config clears its own best baseline "
        "by a real margin -- something Level A could not do across any "
        "candidate-set or lambda combination tried so far -- that's the first "
        "solid evidence that richer state granularity helps. If Level B's "
        "gap over its baseline looks the same (or worse) than Level A's, "
        "the extra state granularity isn't paying for itself yet, possibly "
        "because verb+roles fragments the transition table faster than it "
        "disambiguates it -- check the vocab diagnostics above for how much "
        "the vocab size grew and how many states became singletons."
    )

    print("\nExamples (Level B, Union, Full variant)")

    level_b_results = all_results["Level B (verb+roles)"]
    examples = level_b_results["combos"][("Union", "Full (lambda=1.0)")]["examples"]

    if not examples:
        print("(no recoverable examples captured)")
    else:
        for ex in examples:
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