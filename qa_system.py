#!/usr/bin/env python3
"""
qa_system.py — Improved QA system using the Graph‑PEG preprocessor.

Improvements over v1:
- Better matching: requires at least one argument overlap.
- Context deduplication.
- Clear "not found" message.
"""

import argparse
import pickle
import sys
from typing import List, Dict, Tuple, Optional, Any

import spacy
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------------------------
# 1. LOAD THE STORY GRAPH
# --------------------------------------------------------------------
def load_graph(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# --------------------------------------------------------------------
# 2. PARSE THE QUESTION
# --------------------------------------------------------------------
def parse_question(text: str) -> Optional[Dict[str, Any]]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    root_verb = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            root_verb = token
            break
    if root_verb is None:
        print("Could not find a main verb in the question.")
        return None

    verb = root_verb.lemma_.lower()

    args = {}
    for child in root_verb.children:
        dep = child.dep_
        if dep == "nsubj":
            args["AGENT"] = child.text
        elif dep == "dobj":
            args["PATIENT"] = child.text
        elif dep == "iobj":
            args["RECIPIENT"] = child.text
        elif dep == "prep":
            for subchild in child.children:
                if subchild.dep_ == "pobj":
                    if child.text.lower() == "to":
                        args["RECIPIENT"] = subchild.text
                    elif child.text.lower() == "with":
                        args["INSTRUMENT"] = subchild.text

    return {"verb": verb, "args": args, "raw_text": text}

# --------------------------------------------------------------------
# 3. MATCHING (improved)
# --------------------------------------------------------------------
def find_best_match(query_event: Dict[str, Any],
                    story_events: List[Dict[str, Any]],
                    story_entities: Dict[int, Dict[str, Any]]) -> Tuple[Optional[int], float]:
    """
    Find the best matching event in the story.
    Requires at least one argument to overlap (or no args specified).
    """
    best_idx = -1
    best_score = 0.0

    query_verb = query_event["verb"]
    query_args = query_event["args"]

    for idx, ev in enumerate(story_events):
        # Skip if verb doesn't match
        if ev.get("event_type", "") != query_verb:
            continue

        # If query has no args, accept any verb match
        if not query_args:
            return idx, 1.0

        # Check argument overlap
        matched_roles = 0
        total_roles = len(query_args)

        for role, q_filler in query_args.items():
            # Look for this role in the event
            for r in ev["roles"]:
                if r["role"] != role:
                    continue
                eid = r.get("entity_id")
                if eid is None:
                    continue
                filler_text = story_entities.get(eid, {}).get("canonical_text", "").lower()

                # Check if the query filler appears in the event filler or vice versa
                if q_filler.lower() in filler_text or filler_text in q_filler.lower():
                    matched_roles += 1
                    break

        # Compute score: verb match (1) + arg overlap (0.5 each)
        score = 1.0 + (matched_roles / total_roles) * 0.5

        # Only consider events with at least one matched arg
        if matched_roles > 0 and score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score

# --------------------------------------------------------------------
# 4. GET CONTEXT (with deduplication)
# --------------------------------------------------------------------
def get_context(story_events: List[Dict[str, Any]], event_idx: int, n: int = 2) -> List[str]:
    if event_idx < 0:
        return []

    sent_idx = story_events[event_idx].get("sentence_idx", -1)
    if sent_idx < 0:
        return []

    seen = set()
    context = []
    for ev in story_events:
        ev_sent_idx = ev.get("sentence_idx", -1)
        if sent_idx - n <= ev_sent_idx <= sent_idx + n:
            sent = ev.get("metadata", {}).get("sentence", "")
            if sent and sent not in seen:
                seen.add(sent)
                context.append(sent)
    return context

# --------------------------------------------------------------------
# 5. MAIN
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QA system using Graph‑PEG.")
    parser.add_argument("--graph", "-g", type=str, required=True,
                        help="Path to the story graph pickle")
    parser.add_argument("--question", "-q", type=str, required=True,
                        help="The user's question")
    parser.add_argument("--context", "-c", type=int, default=2,
                        help="Number of sentences to show before/after the match")
    args = parser.parse_args()

    print("Loading story graph...")
    data = load_graph(args.graph)
    story_events = data.get("events", [])
    story_entities = data.get("entities", {})

    if not story_events:
        print("No events found in the graph.")
        sys.exit(1)

    print(f"Loaded {len(story_events)} events.\n")

    print(f"Question: {args.question}")
    query = parse_question(args.question)
    if query is None:
        sys.exit(1)

    print(f"  Verb: {query['verb']}")
    print(f"  Args: {query['args']}\n")

    idx, score = find_best_match(query, story_events, story_entities)

    if idx == -1:
        print("❌ No matching event found in the story.")
        print("   (Try a different verb or ask about something that happened.)")
        sys.exit(0)

    matched_event = story_events[idx]
    sent = matched_event.get("metadata", {}).get("sentence", "[No sentence]")
    print(f"✅ Best match (score: {score:.2f}):")
    print(f"  Event: {matched_event['event_type']}")
    role_strs = []
    for r in matched_event["roles"]:
        if r["entity_id"] is not None:
            filler = story_entities.get(r["entity_id"], {}).get("canonical_text", "?")
        else:
            filler = "(flag)"
        role_strs.append(f"{r['role']}={filler}")
    print(f"  Roles: {', '.join(role_strs)}")

    print("\nContext:")
    context_sents = get_context(story_events, idx, n=args.context)
    for s in context_sents:
        print(f"  {s}")

if __name__ == "__main__":
    main()