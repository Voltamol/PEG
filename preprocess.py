#!/usr/bin/env python3
# preprocess.py — Graph‑PEG dataset generator, v11 (fixed validation + npadvmod)
# VERSION MARKER: v11-entity-validation

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

# v10: source reliability multipliers
SOURCE_RELIABILITY = {
    'direct_nsubj': 1.0,
    'direct_nsubjpass': 1.0,
    'direct_dobj': 1.0,
    'direct_iobj': 0.95,
    'prep_pobj': 0.9,
    'direct_advmod': 0.8,
    'direct_comp': 0.85,
    'subj_control': 0.7,
    'obj_control': 0.65,
    'conj_inherited': 0.6,
    'relcl_inherited': 0.55,
    'relcl_antecedent': 0.75,
    'neg_flag': 1.0,          # polarity flags are exact
}

# v10: blacklist for ambiguous/placeholder fillers
ENTITY_BLACKLIST = {
    '_', 'which', 'what', 'that', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how'
}

# v10: allowed POS tags per role (None means any)
ROLE_POS_ALLOWED = {
    'AGENT':     {'NOUN', 'PROPN', 'PRON'},
    'PATIENT':   {'NOUN', 'PROPN', 'PRON'},
    'RECIPIENT': {'NOUN', 'PROPN', 'PRON'},
    'LOCATION':  {'NOUN', 'PROPN'},
    'TIME':      {'NOUN', 'PROPN'},   # times can be proper nouns (e.g., "Sunday")
    'MANNER':    {'ADV', 'NOUN'},     # adverbs or noun‑phrases (e.g., "with care")
    'ATTRIBUTE': {'ADJ', 'NOUN', 'PROPN'},
    'MODIFIER':  None,                # any, but will be refined later
    'POLARITY':  None,                # flag, no entity
}

# --------------------------------------------------------------------
# 2. ENTITY MERGING (string‑match only)
# --------------------------------------------------------------------
MERGE_METHOD = 'exact_string_match_v1'

class EntityRegistry:
    """
    Owns entity creation and lookup. Merges by exact lowercased string.
    """
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.entities: Dict[int, Dict[str, Any]] = {}
        self._text_to_id: Dict[str, int] = {}
        self._next_id = 0

    def resolve_or_create(self, text: str, sent_idx: int, is_pronoun: bool) -> Tuple[int, int]:
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
# 3. ROLE MAPPER (with provenance)
# --------------------------------------------------------------------
DEP_TO_ROLE = {
    'nsubj': 'AGENT',
    'nsubjpass': 'PATIENT',      # passive subject = patient
    'dobj': 'PATIENT',
    'iobj': 'RECIPIENT',
    'dative': 'RECIPIENT',
    'pobj': None,                # handled specially via prep
    'advmod': 'MANNER',
    'neg': 'POLARITY',
    'amod': 'MODIFIER',
}

LOCATION_PREPS = {'in', 'at', 'on', 'by', 'near', 'under', 'over', 'behind',
                  'to', 'into', 'from', 'through', 'inside', 'outside',
                  'beneath', 'beside', 'within', 'above', 'below'}
TIME_PREPS = {'at', 'on', 'in', 'during', 'after', 'before', 'since', 'until'}

# Expanded TIME noun list (from previous feedback)
TIME_NOUNS = {
    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sundays', 'mondays', 'tuesdays', 'wednesdays',
    'thursdays', 'fridays', 'saturdays',
    'morning', 'afternoon', 'evening', 'night', 'noon', 'midnight',
    'sunrise', 'sunset', 'dawn', 'dusk',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'minute', 'hour', 'day', 'week', 'month', 'year', 'decade',
    'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades',
    'attempt', 'attempts', 'occasion', 'occasions',  # "after several attempts"
    'today', 'tomorrow', 'yesterday', 'tonight',  # NEW: common time nouns
}

def map_prep_role(prep_token, prep_obj_token=None) -> str:
    """
    Determines role of a prepositional phrase.
    TIME fix: only assign TIME if the preposition is in TIME_PREPS AND
    the object lemma is in TIME_NOUNS (or the text is a time noun).
    Otherwise, default to LOCATION (or MODIFIER if not in LOCATION_PREPS).
    """
    lemma = prep_token.lemma_.lower()
    obj_lemma = prep_obj_token.lemma_.lower() if prep_obj_token is not None else ''
    obj_text = prep_obj_token.text.lower() if prep_obj_token is not None else ''

    # First, if it's a clear time expression
    if lemma in TIME_PREPS and (obj_lemma in TIME_NOUNS or obj_text in TIME_NOUNS):
        return 'TIME'

    # If it's a location preposition, even if the object is not in TIME_NOUNS
    if lemma in LOCATION_PREPS:
        return 'LOCATION'

    # Everything else becomes MODIFIER (can be refined later)
    return 'MODIFIER'


def refine_role(event_type: str, role_name: str, token) -> str:
    """
    Refines a generic role (MODIFIER, LOCATION, etc.) into a more specific one
    based on the event type and dependency relation.
    """
    if role_name == 'MODIFIER' and token.dep_ in ('attr', 'acomp'):
        # Copula complement (predicate adjective/nominal)
        if event_type in ('be', 'seem', 'become', 'appear', 'remain', 'stay',
                          'look', 'feel', 'smell', 'taste', 'sound', 'keep'):
            return 'ATTRIBUTE'
    if role_name == 'LOCATION' and token.dep_ == 'prep' and token.lemma_ == 'with':
        # Could be INSTRUMENT (but we need to know the verb)
        # We'll keep as LOCATION for now; can be overridden later.
        pass
    # Keep as is
    return role_name


# --------------------------------------------------------------------
# 4. EVENT EXTRACTION
# --------------------------------------------------------------------
CONTRACTION_LEMMA_MAP = {
    "'s": 'be', "’s": 'be',
    "'m": 'be', "’m": 'be',
    "'re": 'be', "’re": 'be',
    "'ve": 'have', "’ve": 'have',
    "'d": 'have', "’d": 'have',
    'tis': 'be', "'tis": 'be',
    'twas': 'be', "'twas": 'be',
    'doth': 'do', 'dost': 'do',
    'hath': 'have',
    # Add standalone apostrophe (sometimes root)
    "'": 'be', "’": 'be',
}

BARE_MODAL_LEMMAS = {
    'can', 'could', 'will', 'would', 'must', 'should', 'shall', 'might', 'may',
}

def normalize_event_type(raw_lemma: str) -> Optional[str]:
    lemma_lower = raw_lemma.lower()
    if lemma_lower in BARE_MODAL_LEMMAS:
        return None
    return CONTRACTION_LEMMA_MAP.get(lemma_lower, lemma_lower)

OBJECT_CONTROL_VERBS = {
    'tell', 'ask', 'order', 'want', 'allow', 'force', 'persuade',
    'convince', 'remind', 'warn', 'permit', 'instruct', 'urge',
}


def find_predicate_tokens(doc):
    """
    Returns all verb/aux tokens that can be predicates (excluding aux, amod).
    """
    predicates = []
    for tok in doc:
        if tok.pos_ in ('VERB', 'AUX') and tok.dep_ not in ('aux', 'auxpass', 'amod'):
            predicates.append(tok)
    return predicates


# v10: validation function
def is_valid_filler(token, role):
    """Check if a token is a plausible filler for the given role."""
    # First, reject obvious non‑entity POS
    if token.pos_ in {'VERB', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'PUNCT', 'SYM'}:
        return False

    # For POLARITY, the filler is not a real entity, so always valid (handled separately)
    if role == 'POLARITY':
        return True

    # Check POS against allowed list (if any)
    allowed = ROLE_POS_ALLOWED.get(role)
    if allowed is not None and token.pos_ not in allowed:
        return False

    # For concrete roles, reject blacklisted tokens
    if role in {'AGENT', 'PATIENT', 'RECIPIENT', 'LOCATION', 'TIME'}:
        if token.text.lower() in ENTITY_BLACKLIST:
            return False

    return True


def extract_event_for_predicate(pred_token, sent_idx, doc):
    """
    Extracts one event, including provenance (source, origin token, dep).
    Returns a dict with:
      - event_type: str
      - roles: list of dict with keys: role, entity_text, is_pronoun, source, origin_token_idx, origin_dep, weight, confidence
      - metadata: dict
    """
    event_type = normalize_event_type(pred_token.lemma_)
    if event_type is None:
        return None

    roles = []

    # Helper to add a role, tracking provenance and validation
    # FIX: added entity_token parameter for validation (for prep phrases we pass the pobj)
    def add_role(role, entity_text, is_pronoun, source, origin_token, origin_dep, weight=1.0, entity_token=None):
        # If entity_token is not provided, use origin_token for validation
        validate_token = entity_token if entity_token is not None else origin_token
        if not is_valid_filler(validate_token, role):
            return

        refined = refine_role(event_type, role, origin_token)
        # v10: compute confidence from source reliability
        reliability = SOURCE_RELIABILITY.get(source, 0.5)
        confidence = weight * reliability

        roles.append({
            'role': refined,
            'entity_text': entity_text,
            'is_pronoun': is_pronoun,
            'source': source,
            'origin_token_idx': origin_token.i,
            'origin_dep': origin_dep,
            'weight': weight,
            'confidence': confidence,
            'reliability': reliability,
        })

    # Process direct children of the predicate
    for child in pred_token.children:
        dep = child.dep_
        if dep in ('nsubj', 'nsubjpass'):
            add_role('AGENT', child.text, child.text.lower() in PRONOUNS,
                     'direct_nsubj', child, dep)
        elif dep == 'dobj':
            add_role('PATIENT', child.text, child.text.lower() in PRONOUNS,
                     'direct_dobj', child, dep)
        elif dep in ('iobj', 'dative'):
            # If the dative token is a preposition, look for pobj
            prep_objs = [c for c in child.children if c.dep_ == 'pobj']
            filler = prep_objs[0] if prep_objs else child
            add_role('RECIPIENT', filler.text, filler.text.lower() in PRONOUNS,
                     'direct_iobj', child, dep)
        elif dep == 'attr' or dep == 'acomp':
            # Copula complement: will be refined to ATTRIBUTE later
            add_role('MODIFIER', child.text, False,
                     'direct_comp', child, dep)
        elif dep == 'prep':
            prep_objs = [c for c in child.children if c.dep_ == 'pobj']
            if prep_objs:
                obj = prep_objs[0]
                role = map_prep_role(child, obj)
                # FIX: pass obj as entity_token for validation
                add_role(role, obj.text, obj.text.lower() in PRONOUNS,
                         'prep_pobj', child, dep, entity_token=obj)
        elif dep == 'advmod':
            add_role('MANNER', child.text, False,
                     'direct_advmod', child, dep)
        # NEW: handle npadvmod (noun phrase adverbial modifier) – often time expressions
        elif dep == 'npadvmod':
            # Determine if it's a time noun
            if child.lemma_.lower() in TIME_NOUNS or child.text.lower() in TIME_NOUNS:
                role = 'TIME'
            else:
                role = 'MANNER'
            add_role(role, child.text, child.text.lower() in PRONOUNS,
                     'direct_npadvmod', child, dep)
        elif dep == 'neg':
            # POLARITY is a flag, not a real entity
            roles.append({
                'role': 'POLARITY',
                'entity_text': 'negative',
                'is_pronoun': False,
                'source': 'neg_flag',
                'origin_token_idx': child.i,
                'origin_dep': dep,
                'weight': 1.0,
                'confidence': 1.0,
                'reliability': 1.0,
            })

    # ---- Subject inheritance for control/raising/relcl ----
    # Inherit subject from matrix verb if this predicate is a complement or relative clause
    if pred_token.dep_ in ('xcomp', 'ccomp', 'advcl', 'relcl'):
        matrix = pred_token.head
        # Check if we already have an AGENT from a direct child (e.g., relative pronoun)
        has_agent = any(r['role'] == 'AGENT' for r in roles)

        # For relative clauses, if we have an AGENT that is a relative pronoun, replace it with the antecedent
        RELATIVE_PRONOUNS = {'that', 'which', 'who', 'whom', 'whose'}
        if pred_token.dep_ == 'relcl':
            antecedent = matrix.text
            for r in roles:
                if r['role'] == 'AGENT' and r['entity_text'].lower() in RELATIVE_PRONOUNS:
                    # Replace with antecedent, but also validate the antecedent token (matrix)
                    if is_valid_filler(matrix, 'AGENT'):
                        r['entity_text'] = antecedent
                        r['is_pronoun'] = False
                        r['source'] = 'relcl_antecedent'
                        r['origin_token_idx'] = matrix.i
                        r['origin_dep'] = matrix.dep_
                        r['weight'] = 0.8
                        r['reliability'] = SOURCE_RELIABILITY['relcl_antecedent']
                        r['confidence'] = r['weight'] * r['reliability']
                    break
            else:
                # No relative pronoun AGENT, so we might need to inherit
                if not has_agent:
                    # Inherit from matrix subject
                    matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
                    if matrix_subj and is_valid_filler(matrix_subj, 'AGENT'):
                        add_role('AGENT', matrix_subj.text, matrix_subj.text.lower() in PRONOUNS,
                                 'relcl_inherited', matrix_subj, matrix_subj.dep_, weight=0.7)
        else:
            # Control/raising: inherit subject from matrix
            if not has_agent:
                matrix_dobj = next((c for c in matrix.children if c.dep_ == 'dobj'), None)
                matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
                # Object control vs subject control
                if matrix.lemma_.lower() in OBJECT_CONTROL_VERBS and matrix_dobj is not None:
                    inherited = matrix_dobj
                    source = 'obj_control'
                else:
                    inherited = matrix_subj
                    source = 'subj_control'
                if inherited is not None and is_valid_filler(inherited, 'AGENT'):
                    add_role('AGENT', inherited.text, inherited.text.lower() in PRONOUNS,
                             source, inherited, inherited.dep_, weight=0.7)

    # Conjunction handling: if this predicate is a conj, inherit subject from the head verb's subject
    if pred_token.dep_ == 'conj' and not any(r['role'] == 'AGENT' for r in roles):
        matrix = pred_token.head
        matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
        if matrix_subj is not None and is_valid_filler(matrix_subj, 'AGENT'):
            add_role('AGENT', matrix_subj.text, matrix_subj.text.lower() in PRONOUNS,
                     'conj_inherited', matrix_subj, matrix_subj.dep_, weight=0.7)

    # Mood and polarity
    mood = 'interrogative' if doc.text.strip().endswith('?') else 'declarative'
    polarity = 'negative' if any(t.dep_ == 'neg' for t in pred_token.subtree) else 'positive'

    return {
        'event_type': event_type,
        'roles': roles,
        'metadata': {
            'sentence': doc.text,
            'mood': mood,
            'polarity': polarity,
            'predicate_dep': pred_token.dep_,
        }
    }


def extract_events(doc, sent_idx):
    predicates = find_predicate_tokens(doc)
    events = []
    for pred in predicates:
        ev = extract_event_for_predicate(pred, sent_idx, doc)
        if ev is None:
            continue
        if ev['roles']:
            events.append(ev)
        else:
            # Warn only if it's not a bare modal (already filtered)
            if pred.lemma_.lower() not in BARE_MODAL_LEMMAS:
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
                    # POLARITY flag: no real entity
                    role_entries.append({
                        'role': r['role'],
                        'entity_id': None,
                        'mention_idx': None,
                        'confidence': r.get('confidence', 1.0),
                        'source': r.get('source', 'unknown'),
                        'origin_token_idx': r.get('origin_token_idx', -1),
                        'origin_dep': r.get('origin_dep', ''),
                        'weight': r.get('weight', 1.0),
                        'reliability': r.get('reliability', 1.0),
                    })
                    continue

                eid, mention_idx = registry.resolve_or_create(
                    r['entity_text'], sent_idx, r['is_pronoun'])

                # Confidence already computed in add_role; store it
                confidence = r.get('confidence', r.get('weight', 0.5))
                if r['is_pronoun']:
                    unresolved.append({
                        'entity_id': eid, 'text': r['entity_text'],
                        'sentence_idx': sent_idx,
                        'reason': 'pronoun_not_resolved_v1',
                    })

                role_entries.append({
                    'role': r['role'],
                    'entity_id': eid,
                    'mention_idx': mention_idx,
                    'confidence': confidence,
                    'source': r.get('source', 'unknown'),
                    'origin_token_idx': r.get('origin_token_idx', -1),
                    'origin_dep': r.get('origin_dep', ''),
                    'weight': r.get('weight', 1.0),
                    'reliability': r.get('reliability', 1.0),
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
# 6. SELF-CHECKS (unchanged)
# --------------------------------------------------------------------
def run_self_checks(output, is_demo_corpus=True):
    if not is_demo_corpus:
        print("\n(Self-checks skipped — not the demo corpus. "
              "Inspect the event dump manually instead.)")
        return True

    events = output['events']
    by_type = {}
    for e in events:
        by_type.setdefault(e['event_type'], []).append(e)

    errors = []
    # Check for 'chase' event (relative clause test)
    if 'chase' not in by_type:
        errors.append("Expected an event for 'chase' (from 'chased').")
    else:
        chase_roles = by_type['chase'][0]['roles']
        agent_roles = [r for r in chase_roles if r['role'] == 'AGENT']
        if not agent_roles:
            errors.append("'chase' event has no AGENT.")
        else:
            agent_id = agent_roles[0]['entity_id']
            agent_text = output['entities'][agent_id]['canonical_text'].lower() if agent_id is not None else None
            if agent_text not in ('that', 'which', 'who', 'whom') and agent_text != 'cat':
                errors.append(f"'chase' AGENT is '{agent_text}', expected 'cat'.")

    if 'give' not in by_type:
        errors.append("Expected an event for 'give'.")
    else:
        roles_present = {r['role'] for r in by_type['give'][0]['roles']}
        if 'RECIPIENT' not in roles_present:
            errors.append("'give' has no RECIPIENT.")
        if 'PATIENT' not in roles_present:
            errors.append("'give' has no PATIENT.")

    if errors:
        print("\n*** SELF-CHECK FAILURES ***")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("\nSelf-checks passed.")
        return True


# --------------------------------------------------------------------
# 7. DUMP EVENTS (updated to show confidence)
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
            # Show source, weight, and confidence
            src = r.get('source', '?')
            wt = r.get('weight', 1.0)
            conf = r.get('confidence', wt)
            role_strs.append(f"{r['role']}={filler}(src={src},wt={wt:.2f},conf={conf:.2f})")
        print(f"  [{e['event_id']}] {e['event_type']}: {', '.join(role_strs)} "
              f"| mood={e['metadata']['mood']} polarity={e['metadata']['polarity']}")


# --------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print(">>> RUNNING preprocess.py VERSION: v11-entity-validation <<<")
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                         help='Print full dependency parse for every sentence')
    parser.add_argument('--corpus-file', type=str, default=None,
                         help='Path to a text file with one sentence per line. '
                              'If omitted, runs the 5-sentence demo corpus.')
    parser.add_argument('--text', type=str, default=None,
                         help='Direct sentence(s) to process. Provide as a quoted string. '
                              'If given, --corpus-file is ignored.')
    parser.add_argument('--output', type=str, default=None,
                         help='Output pickle path.')
    args = parser.parse_args()

    # Determine input source
    if args.text is not None:
        corpus = [args.text]
        is_demo = False
        output_file = args.output or 'inline_graph.pkl'
    elif args.corpus_file is not None:
        with open(args.corpus_file) as f:
            corpus = [line.strip() for line in f if line.strip()]
        is_demo = False
        output_file = args.output or (args.corpus_file.rsplit('.', 1)[0] + '_graph.pkl')
    else:
        corpus = [
            "John hit the ball.",
            "The ball hit John.",
            "Do you go to church on Sundays?",
            "The cat that chased the mouse is black.",
            "She gave him a book.",
        ]
        is_demo = True
        output_file = args.output or 'demo_graph_corpus.pkl'

    output = preprocess(corpus, output_file=output_file, debug=args.debug)
    ok = run_self_checks(output, is_demo_corpus=is_demo)
    dump_events(output)
    sys.exit(0 if ok else 1)