#!/usr/bin/env python3
# preprocess.py — Generate the Graph-PEG dataset from raw text.
#
# IMPORTANT — READ BEFORE TRUSTING THIS FILE:
# I do not have a working spaCy install in my execution sandbox (no network
# route to spaCy's model release assets from here), so the dependency-label
# assumptions below are based on documented spaCy/Universal Dependencies
# conventions, NOT a verified run. The script is written to fail loudly
# and print its own parse trees so you can sanity-check every assumption
# the first time you run it locally. Do not trust the "Entities/Events"
# summary line until you've read the --debug output for at least the
# five demo sentences.
#
# Run with:  python3 preprocess.py --debug

import argparse
import pickle
import sys
from typing import List, Dict, Tuple, Optional, Any

import torch
import spacy
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------
CONFIDENCE_FLOOR = 0.5
EMBEDDING_POLICY = 'most_recent'

# --------------------------------------------------------------------
# 2. ENTITY MERGING — STRING MATCH, MADE EXPLICIT (NOT REAL COREFERENCE)
# --------------------------------------------------------------------
# v1 HONEST LIMITATION:
# We merge entities by exact lowercased string match on the head noun text.
# This is NOT coreference resolution. It will:
#   - correctly merge "John" (sentence 1) with "John" (sentence 5)
#   - INCORRECTLY merge two different people/things that happen to share a
#     name or noun ("the bank" the river vs "the bank" the institution;
#     two different characters both named "John")
#   - FAIL to merge "John" with "he" (pronouns are never resolved in v1 —
#     they become their own, permanently separate entities)
#
# This is a real, known gap, not a hidden one. It is tracked explicitly
# below via the `merge_method` field on every entity, so you can filter
# or audit by it later instead of discovering it by surprise.
MERGE_METHOD = 'exact_string_match_v1'


class EntityRegistry:
    """
    Owns entity creation and lookup. Centralizing this in one class (rather
    than the inline dict-scan from the previous draft) means there is
    exactly one place that defines what "the same entity" means — so when
    real coreference resolution is added later, only this class changes.
    """

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.entities: Dict[int, Dict[str, Any]] = {}
        self._text_to_id: Dict[str, int] = {}  # head-noun-text -> entity_id
        self._next_id = 0

    def resolve_or_create(self, text: str, sent_idx: int, is_pronoun: bool) -> Tuple[int, int]:
        """
        Returns (entity_id, mention_idx).
        Pronouns ALWAYS create a new entity in v1 (no resolution attempted) —
        this is more honest than silently matching "it" to whatever
        head-noun string happens to equal "it" (which would never happen,
        but the previous draft's logic made no distinction and that's the
        kind of silent gap that hides until you test "he"/"she"/"they").
        """
        key = text.lower().strip()

        if is_pronoun:
            eid = self._next_id
            self._next_id += 1
            self._create(eid, text, sent_idx, is_pronoun=True)
            return eid, 0

        if key in self._text_to_id:
            eid = self._text_to_id[key]
            mention_idx = self._add_mention(eid, text, sent_idx, is_pronoun=False)
            return eid, mention_idx

        eid = self._next_id
        self._next_id += 1
        self._text_to_id[key] = eid
        self._create(eid, text, sent_idx, is_pronoun=False)
        return eid, 0

    def _create(self, eid: int, text: str, sent_idx: int, is_pronoun: bool):
        emb = self.encoder.encode([text], convert_to_tensor=True).squeeze(0).cpu()
        self.entities[eid] = {
            'canonical_text': text,
            'merge_method': MERGE_METHOD,
            'mentions': [{
                'text': text,
                'sentence_idx': sent_idx,
                'embedding': emb,
                'gender': 'UNKNOWN',
                'number': 'UNKNOWN',
                'is_pronoun': is_pronoun,
            }],
            'embedding_policy': EMBEDDING_POLICY,
        }

    def _add_mention(self, eid: int, text: str, sent_idx: int, is_pronoun: bool) -> int:
        emb = self.encoder.encode([text], convert_to_tensor=True).squeeze(0).cpu()
        mention = {
            'text': text,
            'sentence_idx': sent_idx,
            'embedding': emb,
            'gender': 'UNKNOWN',
            'number': 'UNKNOWN',
            'is_pronoun': is_pronoun,
        }
        self.entities[eid]['mentions'].append(mention)
        return len(self.entities[eid]['mentions']) - 1


PRONOUNS = {
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'we', 'us', 'our', 'ours', 'ourselves',
}


# --------------------------------------------------------------------
# 3. ROLE MAPPER (spaCy dep label -> our fixed role inventory)
# --------------------------------------------------------------------
DEP_TO_ROLE = {
    'nsubj': 'AGENT',
    'nsubjpass': 'AGENT',
    'dobj': 'PATIENT',
    'iobj': 'RECIPIENT',
    'dative': 'RECIPIENT',   # "gave him a book": spaCy often tags "him" as
                             # `dative`, not `iobj`, depending on model
                             # version. Both map to RECIPIENT. VERIFY THIS
                             # against your installed spaCy's actual output
                             # on a ditransitive sentence before trusting it.
    'pobj': None,            # handled specially — see map_prep_role()
    'advmod': 'MANNER',
    'neg': 'POLARITY',
    'amod': 'MODIFIER',
}

LOCATION_PREPS = {'in', 'at', 'on', 'by', 'near', 'under', 'over', 'behind'}
TIME_PREPS = {'at', 'on', 'in', 'during', 'after', 'before', 'since', 'until'}


def map_prep_role(prep_token) -> str:
    lemma = prep_token.lemma_.lower()
    if lemma in TIME_PREPS:
        return 'TIME'
    if lemma in LOCATION_PREPS:
        return 'LOCATION'
    return 'MODIFIER'


# --------------------------------------------------------------------
# 4. EVENT EXTRACTOR — RECURSIVE, MULTI-CLAUSE
# --------------------------------------------------------------------
# FIX for the bug flagged in review: the previous version only visited
# `root.children` for the ROOT token, so any verb buried under a relative
# clause (`relcl`), clausal complement (`ccomp`/`xcomp`), or conjunct
# (`conj`) was silently dropped — e.g. "The cat that chased the mouse is
# black" produced ONE event (for "is") with zero roles for "chased".
#
# Fix: find ALL verb tokens in the doc (anything with pos_ == 'VERB' or
# 'AUX' acting as a main predicate), and extract an event for each one
# independently, using that verb's own .children — not just the ROOT's.
def find_predicate_tokens(doc):
    """
    Returns every token that should head its own event: main verbs,
    copula-like predicates (ROOT with attr/acomp children even if the
    ROOT itself is a form of "be"), and verbs embedded in relative
    clauses or clausal complements.
    """
    predicates = []
    for tok in doc:
        if tok.pos_ in ('VERB', 'AUX') and tok.dep_ != 'aux':
            predicates.append(tok)
    return predicates


def extract_event_for_predicate(pred_token, sent_idx, doc):
    event_type = pred_token.lemma_
    roles = []

    for child in pred_token.children:
        dep = child.dep_

        if dep in ('nsubj', 'nsubjpass'):
            roles.append({'role': 'AGENT', 'entity_text': child.text,
                          'is_pronoun': child.text.lower() in PRONOUNS})
        elif dep == 'dobj':
            roles.append({'role': 'PATIENT', 'entity_text': child.text,
                          'is_pronoun': child.text.lower() in PRONOUNS})
        elif dep in ('iobj', 'dative'):
            roles.append({'role': 'RECIPIENT', 'entity_text': child.text,
                          'is_pronoun': child.text.lower() in PRONOUNS})
        elif dep == 'attr' or dep == 'acomp':
            # Copula complement: "is black" -> black is the THEME-ish
            # complement. We file it under MODIFIER per the v1 role table
            # (no separate THEME role was locked in the spec — flag this
            # if you want THEME added back as a distinct role).
            roles.append({'role': 'MODIFIER', 'entity_text': child.text,
                          'is_pronoun': False})
        elif dep == 'prep':
            prep_objs = [c for c in child.children if c.dep_ == 'pobj']
            if prep_objs:
                role = map_prep_role(child)
                roles.append({'role': role, 'entity_text': prep_objs[0].text,
                              'is_pronoun': prep_objs[0].text.lower() in PRONOUNS})
        elif dep == 'advmod':
            roles.append({'role': 'MANNER', 'entity_text': child.text,
                          'is_pronoun': False})
        elif dep == 'neg':
            roles.append({'role': 'POLARITY', 'entity_text': 'negative',
                          'is_pronoun': False})

    # Relative clause subject substitution — CONFIRMED BUG, FIXED HERE.
    #
    # Verified against real spaCy output on "The cat that chased the mouse
    # is black": "that" IS tagged nsubj of "chased" (the slot is NOT
    # empty), so the original "only fill AGENT if empty" check never
    # fired. The relative pronoun ("that"/"which"/"who") was being stored
    # as the literal AGENT filler, which is meaningless — it has no
    # real-world referent. The actual agent ("cat") was being dropped.
    #
    # Fix: detect when an AGENT (or any role) filler IS a relative
    # pronoun, and substitute the antecedent (pred_token.head — the noun
    # this relative clause modifies) instead of just checking for an
    # empty slot.
    RELATIVE_PRONOUNS = {'that', 'which', 'who', 'whom', 'whose'}
    if pred_token.dep_ == 'relcl':
        antecedent = pred_token.head.text
        for r in roles:
            if r['entity_text'].lower() in RELATIVE_PRONOUNS:
                r['entity_text'] = antecedent
                r['is_pronoun'] = False  # it's now resolved to the real noun
        if not any(r['role'] == 'AGENT' for r in roles):
            # Truly empty subject slot (different parse shape) — fall
            # back to the antecedent directly.
            roles.append({'role': 'AGENT', 'entity_text': antecedent,
                          'is_pronoun': False})

    mood = 'interrogative' if doc.text.strip().endswith('?') else 'declarative'
    polarity = 'negative' if any(t.dep_ == 'neg' for t in pred_token.subtree) else 'positive'

    return {
        'event_type': event_type,
        'roles': roles,
        'metadata': {
            'sentence': doc.text,
            'mood': mood,
            'polarity': polarity,
            'predicate_dep': pred_token.dep_,  # debug aid: was this ROOT, relcl, ccomp...?
        }
    }


def extract_events(doc, sent_idx):
    predicates = find_predicate_tokens(doc)
    events = []
    for pred in predicates:
        ev = extract_event_for_predicate(pred, sent_idx, doc)
        if ev['roles']:  # skip predicates that yielded nothing (e.g. bare aux)
            events.append(ev)
        else:
            print(f"  [WARN] predicate '{pred.text}' (dep={pred.dep_}) "
                  f"in sentence {sent_idx} produced zero roles — check parse.")
    return events


# --------------------------------------------------------------------
# 5. MAIN PIPELINE
# --------------------------------------------------------------------
def preprocess(corpus: List[str], model_name='all-MiniLM-L6-v2',
               output_file='graph_corpus.pkl', debug=False):
    nlp = spacy.load('en_core_web_sm')
    encoder = SentenceTransformer(model_name)
    registry = EntityRegistry(encoder)

    all_events = []
    unresolved = []
    event_types: Dict[str, int] = {}
    event_id_counter = 0

    for sent_idx, sent in enumerate(corpus):
        doc = nlp(sent)

        if debug:
            print(f"\n--- Sentence {sent_idx}: {sent!r} ---")
            for tok in doc:
                print(f"  {tok.text:12s} dep={tok.dep_:10s} head={tok.head.text:10s} pos={tok.pos_}")

        events = extract_events(doc, sent_idx)

        if debug:
            print(f"  -> extracted {len(events)} event(s): "
                  f"{[e['event_type'] for e in events]}")

        for ev in events:
            event_type = ev['event_type']
            if event_type not in event_types:
                event_types[event_type] = len(event_types)

            role_entries = []
            for r in ev['roles']:
                if r['entity_text'] == 'negative':
                    # POLARITY's filler is a flag, not a real entity —
                    # don't allocate an entity node for it.
                    role_entries.append({'role': r['role'], 'entity_id': None,
                                          'mention_idx': None, 'confidence': 1.0})
                    continue

                eid, mention_idx = registry.resolve_or_create(
                    r['entity_text'], sent_idx, r['is_pronoun'])

                confidence = 0.3 if r['is_pronoun'] else 1.0
                if r['is_pronoun']:
                    unresolved.append({
                        'entity_id': eid, 'text': r['entity_text'],
                        'sentence_idx': sent_idx,
                        'reason': 'pronoun_not_resolved_v1',
                    })

                role_entries.append({
                    'role': r['role'], 'entity_id': eid,
                    'mention_idx': mention_idx, 'confidence': confidence,
                })

            all_events.append({
                'event_id': event_id_counter,
                'event_type': event_type,
                'sentence_idx': sent_idx,
                'roles': role_entries,
                'metadata': ev['metadata'],
            })
            event_id_counter += 1

    output = {
        'entities': registry.entities,
        'events': all_events,
        'event_types': event_types,
        'unresolved': unresolved,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nSaved graph corpus to {output_file}")
    print(f"Entities: {len(registry.entities)}, Events: {len(all_events)}, "
          f"Unresolved pronoun mentions: {len(unresolved)}")
    return output


# --------------------------------------------------------------------
# 6. SELF-CHECK — fails loudly if known structural bugs have regressed
# --------------------------------------------------------------------
def run_self_checks(output):
    events = output['events']
    by_type = {}
    for e in events:
        by_type.setdefault(e['event_type'], []).append(e)

    errors = []

    if 'chase' not in by_type:
        errors.append("Expected an event for 'chase' (from 'chased') — "
                       "relative clause extraction is broken.")
    else:
        chase_roles = by_type['chase'][0]['roles']
        agent_roles = [r for r in chase_roles if r['role'] == 'AGENT']
        if not agent_roles:
            errors.append("'chase' event has no AGENT — implicit relcl subject "
                           "resolution is broken.")
        else:
            # Check the FILLER, not just presence — this is the check that
            # was missing before and let "AGENT=that" pass as correct.
            agent_id = agent_roles[0]['entity_id']
            agent_text = output['entities'][agent_id]['canonical_text'].lower() if agent_id is not None else None
            if agent_text in ('that', 'which', 'who', 'whom'):
                errors.append(f"'chase' event's AGENT filler is the relative "
                               f"pronoun '{agent_text}', not the antecedent noun "
                               f"('cat') — relative pronoun substitution is broken.")
            elif agent_text != 'cat':
                errors.append(f"'chase' event's AGENT filler is '{agent_text}', "
                               f"expected 'cat'.")

    if 'give' not in by_type:
        errors.append("Expected an event for 'give' (from 'gave').")
    else:
        roles_present = {r['role'] for r in by_type['give'][0]['roles']}
        if 'RECIPIENT' not in roles_present:
            errors.append("'give' event has no RECIPIENT — check whether "
                           "your spaCy version tags the indirect object as "
                           "'dative' or 'iobj' and confirm both are in DEP_TO_ROLE.")
        if 'PATIENT' not in roles_present:
            errors.append("'give' event has no PATIENT (the book).")

    if errors:
        print("\n*** SELF-CHECK FAILURES ***")
        for err in errors:
            print(f"  - {err}")
        print("These indicate the script does NOT yet handle the cases "
              "it claims to. Fix before using output for training.\n")
    else:
        print("\nSelf-checks passed (relative clause + ditransitive "
              "extraction produced expected role types).")

    return len(errors) == 0


# --------------------------------------------------------------------
# 7. DEMO
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                         help='Print full dependency parse for every sentence')
    args = parser.parse_args()

    corpus = [
        "John hit the ball.",
        "The ball hit John.",
        "Do you go to church on Sundays?",
        "The cat that chased the mouse is black.",
        "She gave him a book.",
    ]

    output = preprocess(corpus, output_file='demo_graph_corpus.pkl', debug=args.debug)
    ok = run_self_checks(output)
    dump_events(output)
    sys.exit(0 if ok else 1)

# --------------------------------------------------------------------
# Quick standalone inspector — paste this output back for review
# --------------------------------------------------------------------
def dump_events(output):
    print("\n=== FULL EVENT DUMP ===")
    for e in output['events']:
        role_strs = []
        for r in e['roles']:
            if r['entity_id'] is None:
                filler = '(flag)'
            else:
                filler = output['entities'][r['entity_id']]['canonical_text']
            role_strs.append(f"{r['role']}={filler}(conf={r['confidence']})")
        print(f"  [{e['event_id']}] {e['event_type']}: {', '.join(role_strs)} "
              f"| mood={e['metadata']['mood']} polarity={e['metadata']['polarity']}")