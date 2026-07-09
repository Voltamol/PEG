#!/usr/bin/env python3
# preprocess.py — Graph‑PEG dataset generator, v17 (probabilistic coref)
# VERSION MARKER: v17-prob-coref

import argparse
import pickle
import sys
from typing import List, Dict, Tuple, Optional, Any, Generator

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
    'neg_flag': 1.0,
    'direct_npadvmod': 0.8,
    # v12: sources for conjuncts
    'direct_nsubj_conj': 0.9,
    'direct_dobj_conj': 0.9,
    'prep_pobj_conj': 0.85,
    'direct_iobj_conj': 0.85,
    'appos_inherited': 0.8,
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
    'TIME':      {'NOUN', 'PROPN'},
    'MANNER':    {'ADV', 'NOUN'},
    'ATTRIBUTE': {'ADJ', 'NOUN', 'PROPN'},
    'MODIFIER':  None,
    'POLARITY':  None,
}

# --------------------------------------------------------------------
# 2a. PRONOUN METADATA AND COREF RESOLVER
# --------------------------------------------------------------------
# Pronoun attribute table: (gender, number)
PRONOUN_ATTRS = {
    'he': ('MASC', 'SING'), 'him': ('MASC', 'SING'), 'his': ('MASC', 'SING'),
    'himself': ('MASC', 'SING'),
    'she': ('FEM', 'SING'), 'her': ('FEM', 'SING'), 'hers': ('FEM', 'SING'),
    'herself': ('FEM', 'SING'),
    'it': ('NEUT', 'SING'), 'its': ('NEUT', 'SING'), 'itself': ('NEUT', 'SING'),
    'they': ('PLUR', 'PLUR'), 'them': ('PLUR', 'PLUR'), 'their': ('PLUR', 'PLUR'),
    'theirs': ('PLUR', 'PLUR'), 'themselves': ('PLUR', 'PLUR'),
    'i': ('1ST', 'SING'), 'me': ('1ST', 'SING'), 'my': ('1ST', 'SING'),
    'mine': ('1ST', 'SING'), 'myself': ('1ST', 'SING'),
    'you': ('2ND', 'ANY'), 'your': ('2ND', 'ANY'), 'yours': ('2ND', 'ANY'),
    'yourself': ('2ND', 'SING'),
    'we': ('1ST', 'PLUR'), 'us': ('1ST', 'PLUR'), 'our': ('1ST', 'PLUR'),
    'ours': ('1ST', 'PLUR'), 'ourselves': ('1ST', 'PLUR'),
}

# Interrogative pronouns — never resolve these in narrative
INTERROGATIVE_PRONOUNS = {'who', 'whom', 'whose', 'what', 'which', 'where', 'when', 'why', 'how'}

# Small proper-noun gender dictionary (extend as needed)
PROPER_NOUN_GENDER = {
    'alice': 'FEM', 'queen': 'FEM', 'duchess': 'FEM', 'mary': 'FEM',
    'john': 'MASC', 'rabbit': 'MASC', 'king': 'MASC', 'hatter': 'MASC',
    'cat': 'MASC', 'mouse': 'MASC', 'bill': 'MASC',
}

# Feature weights — hand-tuned starting point.
# Can later be learned from labeled data via logistic regression.
FEATURE_WEIGHTS = {
    'recency':        2.5,   # exponential decay over sentence distance
    'parallelism':    3.0,   # same grammatical role (esp. subject↔subject)
    'salience':       1.0,   # log(mention_count)
    'subject_bias':   1.5,   # fraction of mentions as subject
    'semantic':       2.0,   # cosine similarity to local context
    'first_mention':  0.3,   # bonus for being introduced early (protagonist bias)
}

COREF_THRESHOLD = 0.55   # minimum P(best candidate) to commit a link


class CachingEncoder:
    """
    Wraps a SentenceTransformer with:
      1. An exact-string cache — repeated mentions (pronouns like "he"/"she"/"it",
         or names like "Kevin" appearing dozens of times in a transcript) are only
         embedded once, not once per mention.
      2. A true batch-encode path — callers with a known list of strings up front
         (e.g. all sentences in a corpus) get one model call instead of N calls with
         batch size 1, which is where most of the real throughput was being lost.
    Cached by exact string (not lowercased) to avoid conflating case-sensitive
    proper nouns with common words that happen to share a lowercase form.
    """
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
        self._cache: Dict[str, torch.Tensor] = {}

    def encode_one(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        emb = self._model.encode([text], convert_to_tensor=True).squeeze(0).cpu()
        self._cache[text] = emb
        return emb

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Returns a stacked tensor, one row per input text, in the given order.
        Only strings not already cached are sent to the model, and they're sent
        together in a single batched call."""
        to_encode = [t for t in texts if t not in self._cache]
        if to_encode:
            embs = self._model.encode(to_encode, convert_to_tensor=True).cpu()
            for t, e in zip(to_encode, embs):
                self._cache[t] = e
        return torch.stack([self._cache[t] for t in texts])

    # Kept for any external code that still expects the raw .encode() signature.
    def encode(self, texts: List[str], convert_to_tensor: bool = True):
        return self.encode_batch(texts)


class CorefResolver:
    """
    Maintains a discourse model of entities and resolves pronouns
    by computing P(e_k | pronoun, context) via a log-linear model.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        # candidate pool: list of dicts, one per entity bucket
        self.candidates = []
        self._eid_to_idx = {}

    # ---------- attribute inference ----------
    def pronoun_attrs(self, text: str):
        return PRONOUN_ATTRS.get(text.lower(), ('UNKNOWN', 'UNKNOWN'))

    def infer_entity_attrs(self, token):
        """Infer (gender, number) from a spaCy token."""
        # Number
        number = 'UNKNOWN'
        morph_num = token.morph.get("Number")
        if morph_num:
            number = 'PLUR' if 'Plur' in morph_num else 'SING'
        elif token.text.lower().endswith('s') and not token.text.lower().endswith('ss'):
            number = 'PLUR'

        # Gender
        gender = 'UNKNOWN'
        morph_gen = token.morph.get("Gender")
        if morph_gen:
            if 'Masc' in morph_gen: gender = 'MASC'
            elif 'Fem' in morph_gen: gender = 'FEM'
            elif 'Neut' in morph_gen: gender = 'NEUT'
        elif token.pos_ == 'PROPN':
            gender = PROPER_NOUN_GENDER.get(token.text.lower(), 'UNKNOWN')
        return gender, number

    # ---------- candidate management ----------
    def register_entity(self, eid: int, text: str, sent_idx: int, dep: str,
                        gender: str, number: str, embedding, token):
        """Add a new non-pronoun entity to the candidate pool."""
        if eid in self._eid_to_idx:
            # update existing candidate's state
            cand = self.candidates[self._eid_to_idx[eid]]
            cand['last_seen_sent'] = sent_idx
            cand['mention_count'] += 1
            cand['last_dep'] = dep
            if dep in ('nsubj', 'nsubjpass'):
                cand['subject_count'] += 1
            cand['embedding'] = embedding  # refresh to most recent
            cand['last_token'] = token
        else:
            idx = len(self.candidates)
            self._eid_to_idx[eid] = idx
            self.candidates.append({
                'eid': eid,
                'text': text,
                'first_seen_sent': sent_idx,
                'last_seen_sent': sent_idx,
                'mention_count': 1,
                'subject_count': 1 if dep in ('nsubj', 'nsubjpass') else 0,
                'last_dep': dep,
                'gender': gender,
                'number': number,
                'embedding': embedding,
                'last_token': token,
            })

    # ---------- feature functions ----------
    def _f_recency(self, cand, sent_idx):
        """Exponential decay: 1.0 if just mentioned, → 0 as distance grows."""
        dist = sent_idx - cand['last_seen_sent']
        if dist < 0: return 0.0
        import math
        return math.exp(-0.4 * dist)

    def _f_parallelism(self, cand, pronoun_dep):
        """Bonus if pronoun and candidate share grammatical role.
        Subject↔subject gets the biggest bonus (Centering Theory)."""
        cand_dep = cand['last_dep']
        if pronoun_dep == cand_dep:
            return 1.0
        subj_set = {'nsubj', 'nsubjpass'}
        if pronoun_dep in subj_set and cand_dep in subj_set:
            return 1.0
        obj_set = {'dobj', 'pobj', 'iobj', 'dative'}
        if pronoun_dep in obj_set and cand_dep in obj_set:
            return 0.6
        return 0.0

    def _f_salience(self, cand):
        import math
        return math.log1p(cand['mention_count'])

    def _f_subject_bias(self, cand):
        if cand['mention_count'] == 0: return 0.0
        return cand['subject_count'] / cand['mention_count']

    def _f_semantic(self, cand, context_embedding):
        """Cosine similarity between entity embedding and local sentence."""
        if context_embedding is None or cand['embedding'] is None:
            return 0.0
        from sentence_transformers import util
        return float(util.cos_sim(cand['embedding'], context_embedding)[0][0])

    def _f_first_mention(self, cand, sent_idx):
        """Entities introduced early in the text are often protagonists."""
        if sent_idx == 0: return 0.0
        return 1.0 / (1.0 + cand['first_seen_sent'])

    # ---------- the inference ----------
    def resolve(self, pronoun_text: str, pronoun_dep: str,
                sent_idx: int, context_embedding=None):
        """
        Returns (best_eid, probability) or (None, 0.0) if no good candidate.
        """
        # Never resolve interrogative pronouns
        if pronoun_text.lower() in INTERROGATIVE_PRONOUNS:
            return None, 0.0

        p_gender, p_number = self.pronoun_attrs(pronoun_text)
        if not self.candidates:
            return None, 0.0

        # Compute log-score for each candidate
        log_scores = []
        for cand in self.candidates:
            # HARD FILTERS — gender / number mismatch → -inf
            if (p_gender != 'UNKNOWN' and cand['gender'] != 'UNKNOWN'
                    and p_gender != cand['gender']):
                log_scores.append((cand['eid'], float('-inf')))
                continue
            if (p_number != 'UNKNOWN' and cand['number'] != 'UNKNOWN'
                    and p_number != cand['number']):
                log_scores.append((cand['eid'], float('-inf')))
                continue

            # SOFT FEATURES
            f = {
                'recency':       self._f_recency(cand, sent_idx),
                'parallelism':   self._f_parallelism(cand, pronoun_dep),
                'salience':      self._f_salience(cand),
                'subject_bias':  self._f_subject_bias(cand),
                'semantic':      self._f_semantic(cand, context_embedding),
                'first_mention': self._f_first_mention(cand, sent_idx),
            }
            log_score = sum(FEATURE_WEIGHTS[k] * f[k] for k in FEATURE_WEIGHTS)
            log_scores.append((cand['eid'], log_score))

        # Softmax → proper probabilities
        import math
        finite = [(eid, s) for eid, s in log_scores if s != float('-inf')]
        if not finite:
            return None, 0.0
        max_s = max(s for _, s in finite)
        exp_scores = [(eid, math.exp(s - max_s)) for eid, s in finite]
        total = sum(e for _, e in exp_scores)
        probs = [(eid, e / total) for eid, e in exp_scores]
        probs.sort(key=lambda x: -x[1])

        best_eid, best_p = probs[0]
        if best_p < COREF_THRESHOLD:
            return None, 0.0
        return best_eid, best_p


# --------------------------------------------------------------------
# 2b. ENTITY MERGING (string‑match only) + coref integration
# --------------------------------------------------------------------
MERGE_METHOD = 'exact_string_match_v1'

class EntityRegistry:
    """
    Owns entity creation and lookup. Merges by exact lowercased string.
    Integrates with CorefResolver for pronoun resolution.
    """
    def __init__(self, encoder: 'CachingEncoder'):
        self.encoder = encoder
        self.entities: Dict[int, Dict[str, Any]] = {}
        self._text_to_id: Dict[str, int] = {}
        self._next_id = 0
        self.coref = CorefResolver(encoder)

    def resolve_or_create(self, text: str, sent_idx: int, is_pronoun: bool,
                          dep: str = None, token=None,
                          context_embedding=None) -> Tuple[int, int, bool]:
        """Returns (eid, mention_idx, was_resolved)."""
        key = text.lower().strip()

        # --- PRONOUN PATH: ask the resolver first ---
        if is_pronoun:
            best_eid, prob = self.coref.resolve(
                text, dep or '', sent_idx, context_embedding)
            if best_eid is not None:
                mention_idx = self._add_mention(best_eid, text, sent_idx,
                                                is_pronoun=True)
                return best_eid, mention_idx, True
            # fallback: unresolved pronoun becomes its own entity
            eid = self._next_id
            self._next_id += 1
            self._create(eid, text, sent_idx, is_pronoun=True)
            return eid, 0, False

        # --- NON-PRONOUN PATH: exact string match or create ---
        if key in self._text_to_id:
            eid = self._text_to_id[key]
            mention_idx = self._add_mention(eid, text, sent_idx, is_pronoun=False)
        else:
            eid = self._next_id
            self._next_id += 1
            self._text_to_id[key] = eid
            self._create(eid, text, sent_idx, is_pronoun=False)
            mention_idx = 0

        # Register as a candidate for future pronouns
        if token is not None:
            gender, number = self.coref.infer_entity_attrs(token)
            emb = self.entities[eid]['mentions'][-1]['embedding']
            self.coref.register_entity(
                eid, text, sent_idx, dep or '', gender, number, emb, token)

        return eid, mention_idx, False

    def _create(self, eid: int, text: str, sent_idx: int, is_pronoun: bool):
        emb = self.encoder.encode_one(text)
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
        emb = self.encoder.encode_one(text)
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
    'nsubjpass': 'PATIENT',
    'dobj': 'PATIENT',
    'iobj': 'RECIPIENT',
    'dative': 'RECIPIENT',
    'pobj': None,
    'advmod': 'MANNER',
    'neg': 'POLARITY',
    'amod': 'MODIFIER',
}

LOCATION_PREPS = {'in', 'at', 'on', 'by', 'near', 'under', 'over', 'behind',
                  'to', 'into', 'from', 'through', 'inside', 'outside',
                  'beneath', 'beside', 'within', 'above', 'below'}
TIME_PREPS = {'at', 'on', 'in', 'during', 'after', 'before', 'since', 'until'}

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
    'attempt', 'attempts', 'occasion', 'occasions',
    'today', 'tomorrow', 'yesterday', 'tonight',
}

def map_prep_role(prep_token, prep_obj_token=None) -> str:
    lemma = prep_token.lemma_.lower()
    obj_lemma = prep_obj_token.lemma_.lower() if prep_obj_token is not None else ''
    obj_text = prep_obj_token.text.lower() if prep_obj_token is not None else ''

    if lemma in TIME_PREPS and (obj_lemma in TIME_NOUNS or obj_text in TIME_NOUNS):
        return 'TIME'
    if lemma in LOCATION_PREPS:
        return 'LOCATION'
    return 'MODIFIER'


def refine_role(event_type: str, role_name: str, token) -> str:
    if role_name == 'MODIFIER' and token.dep_ in ('attr', 'acomp'):
        if event_type in ('be', 'seem', 'become', 'appear', 'remain', 'stay',
                          'look', 'feel', 'smell', 'taste', 'sound', 'keep'):
            return 'ATTRIBUTE'
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
    predicates = []
    for tok in doc:
        if tok.pos_ in ('VERB', 'AUX') and tok.dep_ not in ('aux', 'auxpass', 'amod'):
            predicates.append(tok)
    return predicates


def is_valid_filler(token, role):
    if token.pos_ in {'VERB', 'AUX', 'PART', 'SCONJ', 'CCONJ', 'PUNCT', 'SYM'}:
        return False
    if role == 'POLARITY':
        return True
    allowed = ROLE_POS_ALLOWED.get(role)
    if allowed is not None and token.pos_ not in allowed:
        return False
    if role in {'AGENT', 'PATIENT', 'RECIPIENT', 'LOCATION', 'TIME'}:
        if token.text.lower() in ENTITY_BLACKLIST:
            return False
    return True


# v12: recursive expansion of conjunctions and appositives
def expand_filler(token, visited=None) -> Generator:
    """
    Recursively yield the token and all its coordinated/apposited children.
    Handles: 'John, Mary, and Sam' and 'John (my brother)'.
    """
    if visited is None:
        visited = set()
    if token.i in visited:
        return
    visited.add(token.i)
    yield token
    for child in token.children:
        if child.dep_ in ('conj', 'appos'):
            yield from expand_filler(child, visited)


# v13: syntactic interrogative detection (works with Doc or Span)
def is_interrogative(doc_or_span) -> bool:
    """
    Returns True if the sentence is an interrogative (question).
    Checks for:
    1. Trailing '?'.
    2. Subject-auxiliary inversion: AUX before nsubj for the ROOT verb.
    """
    text = doc_or_span.text.strip()
    if text.endswith('?'):
        return True

    # Find the root verb
    root = None
    for token in doc_or_span:
        if token.dep_ == 'ROOT':
            root = token
            break
    if root is None:
        return False

    # Check for auxiliary before subject
    aux_token = None
    subj_token = None

    for token in doc_or_span:
        if token.dep_ == 'aux' and token.head == root:
            aux_token = token
        if token.dep_ == 'nsubj' and token.head == root:
            subj_token = token

    if aux_token is not None and subj_token is not None:
        if aux_token.i < subj_token.i:
            return True

    return False


# v15: punctuation normalisation to help sentence segmentation
def normalise_punctuation(text: str) -> str:
    """Normalise ambiguous punctuation that confuses spaCy's sentence boundary detection."""
    text = text.replace("?,", "? ").replace("?.", "? ")
    text = text.replace("!,", "! ").replace("!.", "! ")
    return text


# v16: local negation detection (does NOT descend into clausal complements)
def has_local_negation(pred_token) -> bool:
    """
    Check for negation directly attached to the predicate or its auxiliary chain.
    Does NOT descend into clausal complements (ccomp, advcl, xcomp).
    """
    # 1. Check if the predicate itself has a 'neg' child
    for child in pred_token.children:
        if child.dep_ == 'neg':
            return True
        # 2. Check if an auxiliary (aux) has a 'neg' child (e.g., "doesn't")
        if child.dep_ == 'aux':
            for grandchild in child.children:
                if grandchild.dep_ == 'neg':
                    return True
            # 3. Check if the auxiliary token ends with "n't" (e.g., "don't" as a single token)
            if child.text.lower().endswith("n't"):
                return True

    # 4. Check if the predicate text itself ends with "n't" (e.g., "can't")
    if pred_token.text.lower().endswith("n't"):
        return True

    return False


def extract_event_for_predicate(pred_token, sent_idx, doc, feature_log=None):
    """
    Extracts one event, including provenance and feature logging.
    """
    event_type = normalize_event_type(pred_token.lemma_)
    if event_type is None:
        return None

    roles = []
    if feature_log is None:
        feature_log = []

    def add_role(role, entity_text, is_pronoun, source, origin_token, origin_dep,
                 weight=1.0, entity_token=None, is_conjunct=False):
        validate_token = entity_token if entity_token is not None else origin_token
        if not is_valid_filler(validate_token, role):
            return

        refined = refine_role(event_type, role, origin_token)
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
            'is_conjunct': is_conjunct,
            'entity_token': entity_token,   # store token for coref registration
        })

        feature_log.append({
            'verb': event_type,
            'role': refined,
            'filler': entity_text,
            'filler_pos': validate_token.pos_,
            'dep': origin_dep,
            'source': source,
            'is_pronoun': is_pronoun,
            'is_conjunct': is_conjunct,
            'weight': weight,
            'confidence': confidence,
            'sentence_idx': sent_idx,
        })

    # Process direct children of the predicate
    for child in pred_token.children:
        dep = child.dep_

        if dep in ('nsubj', 'nsubjpass'):
            source = 'direct_nsubj'
            for idx, filler in enumerate(expand_filler(child)):
                weight = 1.0 if idx == 0 else 0.9
                source_actual = source if idx == 0 else f"{source}_conj"
                add_role('AGENT', filler.text, filler.text.lower() in PRONOUNS,
                         source_actual, child, dep, weight=weight, entity_token=filler,
                         is_conjunct=(idx > 0))

        elif dep == 'dobj':
            source = 'direct_dobj'
            for idx, filler in enumerate(expand_filler(child)):
                weight = 1.0 if idx == 0 else 0.9
                source_actual = source if idx == 0 else f"{source}_conj"
                add_role('PATIENT', filler.text, filler.text.lower() in PRONOUNS,
                         source_actual, child, dep, weight=weight, entity_token=filler,
                         is_conjunct=(idx > 0))

        elif dep in ('iobj', 'dative'):
            prep_objs = [c for c in child.children if c.dep_ == 'pobj']
            filler = prep_objs[0] if prep_objs else child
            add_role('RECIPIENT', filler.text, filler.text.lower() in PRONOUNS,
                     'direct_iobj', child, dep, entity_token=filler)

        elif dep == 'attr' or dep == 'acomp':
            add_role('MODIFIER', child.text, False,
                     'direct_comp', child, dep, entity_token=child)

        elif dep == 'prep':
            prep_objs = [c for c in child.children if c.dep_ == 'pobj']
            if prep_objs:
                obj = prep_objs[0]
                role = map_prep_role(child, obj)
                source = 'prep_pobj'
                for idx, filler in enumerate(expand_filler(obj)):
                    weight = 1.0 if idx == 0 else 0.9
                    source_actual = source if idx == 0 else f"{source}_conj"
                    add_role(role, filler.text, filler.text.lower() in PRONOUNS,
                             source_actual, child, dep, weight=weight, entity_token=filler,
                             is_conjunct=(idx > 0))

        elif dep == 'advmod':
            add_role('MANNER', child.text, False,
                     'direct_advmod', child, dep, entity_token=child)

        elif dep == 'npadvmod':
            if child.lemma_.lower() in TIME_NOUNS or child.text.lower() in TIME_NOUNS:
                role = 'TIME'
            else:
                role = 'MANNER'
            add_role(role, child.text, child.text.lower() in PRONOUNS,
                     'direct_npadvmod', child, dep, entity_token=child)

        elif dep == 'neg':
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
                'is_conjunct': False,
                'entity_token': None,
            })

    # ---- Subject inheritance for control/raising/relcl ----
    if pred_token.dep_ in ('xcomp', 'ccomp', 'advcl', 'relcl'):
        matrix = pred_token.head
        has_agent = any(r['role'] == 'AGENT' for r in roles)

        RELATIVE_PRONOUNS = {'that', 'which', 'who', 'whom', 'whose'}
        if pred_token.dep_ == 'relcl':
            antecedent = matrix.text
            for r in roles:
                if r['role'] == 'AGENT' and r['entity_text'].lower() in RELATIVE_PRONOUNS:
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
                if not has_agent:
                    matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
                    if matrix_subj and is_valid_filler(matrix_subj, 'AGENT'):
                        add_role('AGENT', matrix_subj.text, matrix_subj.text.lower() in PRONOUNS,
                                 'relcl_inherited', matrix_subj, matrix_subj.dep_, weight=0.7,
                                 entity_token=matrix_subj)
        else:
            if not has_agent:
                matrix_dobj = next((c for c in matrix.children if c.dep_ == 'dobj'), None)
                matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
                if matrix.lemma_.lower() in OBJECT_CONTROL_VERBS and matrix_dobj is not None:
                    inherited = matrix_dobj
                    source = 'obj_control'
                else:
                    inherited = matrix_subj
                    source = 'subj_control'
                if inherited is not None and is_valid_filler(inherited, 'AGENT'):
                    add_role('AGENT', inherited.text, inherited.text.lower() in PRONOUNS,
                             source, inherited, inherited.dep_, weight=0.7,
                             entity_token=inherited)

    # Conjunction handling for verbs (inherit subject)
    if pred_token.dep_ == 'conj' and not any(r['role'] == 'AGENT' for r in roles):
        matrix = pred_token.head
        matrix_subj = next((c for c in matrix.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
        if matrix_subj is not None and is_valid_filler(matrix_subj, 'AGENT'):
            add_role('AGENT', matrix_subj.text, matrix_subj.text.lower() in PRONOUNS,
                     'conj_inherited', matrix_subj, matrix_subj.dep_, weight=0.7,
                     entity_token=matrix_subj)

    # v14: mood detection per sentence (using pred_token.sent, not the full doc)
    mood = 'interrogative' if is_interrogative(pred_token.sent) else 'declarative'

    # v16: use local negation detection (does not leak from embedded clauses)
    polarity = 'negative' if has_local_negation(pred_token) else 'positive'

    return {
        'event_type': event_type,
        'roles': roles,
        'metadata': {
            'sentence': doc.text,
            'mood': mood,
            'polarity': polarity,
            'predicate_dep': pred_token.dep_,
        },
        'feature_log': feature_log,
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
    encoder = CachingEncoder(model_name)
    registry = EntityRegistry(encoder)

    all_events = []
    unresolved = []
    event_types: Dict[str, int] = {}
    event_id_counter = 0
    global_feature_log = []

    # --- Batch what can be batched, before the per-sentence loop ---
    # 1. Normalize all sentences up front (needed before both batch steps below).
    normalised_corpus = [normalise_punctuation(s) for s in corpus]

    # 2. One batched embedding call for every sentence's context embedding, instead
    #    of N calls with batch size 1. This was the single biggest fixed cost for
    #    long transcripts (hundreds of sentences => hundreds of individual model
    #    calls previously).
    sentence_embeddings = encoder.encode_batch(normalised_corpus)

    # 3. nlp.pipe batches spaCy's own internal processing across sentences too —
    #    same reasoning as above, applied to the parser instead of the encoder.
    docs = list(nlp.pipe(normalised_corpus))

    for sent_idx, (sent, doc, sent_embedding) in enumerate(
            zip(normalised_corpus, docs, sentence_embeddings)):

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
            event_features = ev.get('feature_log', [])

            for r in ev['roles']:
                if r['entity_text'] == 'negative':
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
                        'is_conjunct': r.get('is_conjunct', False),
                    })
                    continue

                # Use the resolver
                eid, mention_idx, was_resolved = registry.resolve_or_create(
                    r['entity_text'], sent_idx, r['is_pronoun'],
                    dep=r.get('origin_dep'),
                    token=r.get('entity_token'),
                    context_embedding=sent_embedding,
                )

                confidence = r.get('confidence', r.get('weight', 0.5))

                # Only log as unresolved if it was a pronoun AND the resolver failed
                if r['is_pronoun'] and not was_resolved:
                    unresolved.append({
                        'entity_id': eid, 'text': r['entity_text'],
                        'sentence_idx': sent_idx,
                        'reason': 'pronoun_not_resolved_probabilistic',
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
                    'is_conjunct': r.get('is_conjunct', False),
                })

            all_events.append({
                'event_id': event_id_counter,
                'event_type': event_type,
                'sentence_idx': sent_idx,
                'roles': role_entries,
                'metadata': ev['metadata'],
            })
            event_id_counter += 1

            global_feature_log.extend(event_features)

    output = {
        'entities': registry.entities,
        'events': all_events,
        'event_types': event_types,
        'unresolved': unresolved,
        'feature_log': global_feature_log,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nSaved graph corpus to {output_file}")
    print(f"Entities: {len(registry.entities)}, Events: {len(all_events)}, "
          f"Unresolved pronoun mentions: {len(unresolved)}")
    print(f"Feature vectors logged: {len(global_feature_log)}")
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
# 7. DUMP EVENTS (updated to show is_conjunct)
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
            src = r.get('source', '?')
            wt = r.get('weight', 1.0)
            conf = r.get('confidence', wt)
            conj = ' C' if r.get('is_conjunct', False) else ''
            role_strs.append(f"{r['role']}={filler}(src={src}{conj},wt={wt:.2f},conf={conf:.2f})")
        print(f"  [{e['event_id']}] {e['event_type']}: {', '.join(role_strs)} "
              f"| mood={e['metadata']['mood']} polarity={e['metadata']['polarity']}")


# --------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print(">>> RUNNING preprocess.py VERSION: v17-prob-coref <<<")
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
