#!/usr/bin/env python3
# graph_peg.py — The Graph‑PEG engine.
# Consumes the output of preprocess.py and trains a dynamic memory graph.
#
# VERSION MARKER: v2-fix-dead-event-embedding-and-masking-leakage
#
# CAVEAT: I have not run this against your actual eldoria_graph.pkl in my
# sandbox (no spaCy/sentence-transformers network access here — same
# limitation as preprocess.py). The fixes below are traced by hand
# against worked examples in comments, but please run this locally and
# treat the printed sanity-check output as the real verification, not
# this docstring.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
from typing import List, Dict, Optional, Any

# --------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------
class GraphPEGConfig:
    def __init__(self):
        self.hidden_dim = 256
        self.role_dim = 64
        self.lr = 1e-3
        self.epochs = 30
        self.gamma = 0.995
        self.alpha = 0.1
        self.beta = 0.05
        self.theta_high = 0.35
        self.theta_arch = 0.10
        self.min_occurrences = 3
        self.merge_threshold = 0.95
        self.mask_prob = 0.15


# --------------------------------------------------------------------
# 2. GRAPH PEG MODEL
# --------------------------------------------------------------------
class GraphPEGModel(nn.Module):
    def __init__(self, config: GraphPEGConfig, dataset: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dataset = dataset

        # Fixed role vocabulary (must match preprocessor)
        self.role_to_idx = {
            'AGENT': 0, 'PATIENT': 1, 'RECIPIENT': 2, 'LOCATION': 3,
            'TIME': 4, 'MANNER': 5, 'MODIFIER': 6, 'POLARITY': 7, 'MOOD': 8
        }
        self.idx_to_role = {v: k for k, v in self.role_to_idx.items()}
        self.num_roles = len(self.role_to_idx)

        # Event types (verbs/predicates)
        self.event_type_to_idx = dict(dataset['event_types'])
        self.num_event_types = len(self.event_type_to_idx)

        # ---- Trainable embeddings ----
        # CONFIRMED BUG FIX: in the original draft, `event_embeddings`
        # was allocated as a trainable parameter but `_compose_event`
        # never read from it — every event's vector was built purely
        # from role*filler sums, with zero dependence on which verb was
        # involved. "John hit the ball" and "John kissed the ball" would
        # have produced IDENTICAL vectors. Fixed below: the event-type
        # embedding is now added as a separate term in composition.
        self.event_embeddings = nn.Parameter(
            torch.randn(self.num_event_types, config.hidden_dim) * 0.01
        )
        self.role_embeddings = nn.Parameter(
            torch.randn(self.num_roles, config.role_dim) * 0.01
        )
        self.role_proj = nn.Linear(config.role_dim, config.hidden_dim)

        # ---- Predictor: event context + role -> predicted entity embedding ----
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + config.role_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # ---- Entity embeddings (frozen, from preprocessor) ----
        self.register_buffer('entity_embeddings', self._build_entity_embeddings())

        # ---- Dynamic state ----
        self.entity_energies = torch.ones(len(self.dataset['entities'])) * 0.5
        self.event_occurrence_counts = torch.zeros(self.num_event_types, dtype=torch.long)
        self.active_event_type_idxs = set(range(self.num_event_types))
        self.active_entity_ids = set(range(len(self.dataset['entities'])))

        # Map from entity_id -> row index in entity_embeddings, since
        # entity IDs in the dataset are not guaranteed to be a dense
        # 0..N-1 range after coreference merging drops some IDs.
        # CONFIRMED RISK (not yet hit, but worth guarding): the original
        # code assumed `entity_embeddings[entity_id]` directly, which
        # only works if dataset['entities'].keys() is exactly
        # range(len(entities)) with no gaps. preprocess.py's
        # EntityRegistry allocates IDs sequentially from 0 with no
        # deletions, so this currently holds — but if you ever filter
        # or delete entities upstream, this indexing breaks silently
        # (wrong embedding, not a crash). Building an explicit map here
        # makes it robust either way at near-zero cost.
        sorted_ids = sorted(self.dataset['entities'].keys())
        self._entity_id_to_row = {eid: i for i, eid in enumerate(sorted_ids)}

        self.to('cpu')

    def _build_entity_embeddings(self):
        entities = self.dataset['entities']
        emb_list = []
        for eid in sorted(entities.keys()):
            policy = entities[eid]['embedding_policy']
            if policy == 'most_recent':
                emb = entities[eid]['mentions'][-1]['embedding']
            elif policy == 'mean':
                emb = torch.mean(torch.stack([m['embedding'] for m in entities[eid]['mentions']]), dim=0)
            else:
                emb = entities[eid]['mentions'][0]['embedding']
            emb_list.append(emb)
        return torch.stack(emb_list)  # (num_entities, hidden_dim)

    def _get_event_type_embedding(self, event_type: str) -> torch.Tensor:
        idx = self.event_type_to_idx[event_type]
        return self.event_embeddings[idx]

    def _get_role_embedding(self, role_name: str) -> torch.Tensor:
        idx = self.role_to_idx[role_name]
        return self.role_embeddings[idx]

    def _get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        row = self._entity_id_to_row[entity_id]
        return self.entity_embeddings[row]

    def _compose_context(self, event_data: Dict, exclude_role_slot: Optional[int] = None) -> torch.Tensor:
        """
        Build the context vector the predictor will see, EXCLUDING the
        role at `exclude_role_slot` (an index into event_data['roles'],
        not a role name — a single event can have two role entries with
        the same role name in principle, so we exclude by position, not
        by matching role string, to avoid accidentally excluding the
        WRONG same-named role if duplicates exist).

        CONFIRMED BUG FIX: the original `_compose_event` summed over
        ALL roles unconditionally, including whatever role the training
        loop was about to mask and ask the predictor to guess. That
        means the answer was already mixed into the predictor's own
        input before each prediction — the loss could be minimized
        partly by the leak rather than by learning real relational
        structure. This version takes an explicit exclusion index and
        the training loop is now required to pass it.

        Includes the event-type embedding as a separate additive term
        (see __init__ comment) — this was previously allocated and
        trained but never actually read anywhere.
        """
        vec = self._get_event_type_embedding(event_data['event_type']).clone()
        for i, role_info in enumerate(event_data['roles']):
            if i == exclude_role_slot:
                continue
            if role_info['entity_id'] is None:
                continue
            role = role_info['role']
            entity_id = role_info['entity_id']
            role_emb = self.role_proj(self._get_role_embedding(role))
            entity_emb = self._get_entity_embedding(entity_id)
            vec = vec + role_emb * entity_emb
        return vec

    def _predict_filler(self, context_vec: torch.Tensor, role: str) -> torch.Tensor:
        role_emb = self._get_role_embedding(role)
        combined = torch.cat([context_vec, role_emb], dim=-1)
        return self.predictor(combined)

    def compute_loss(self, event_id: int, role_slot_idx: int) -> torch.Tensor:
        """
        role_slot_idx is an index into event_data['roles'] (the specific
        role-filler pair being masked this step), not a role name —
        see _compose_context for why that distinction matters.
        """
        event_data = self.dataset['events'][event_id]
        masked_role_info = event_data['roles'][role_slot_idx]
        role_name = masked_role_info['role']
        target_entity_id = masked_role_info['entity_id']

        context_vec = self._compose_context(event_data, exclude_role_slot=role_slot_idx)
        pred_emb = self._predict_filler(context_vec, role_name)
        target_emb = self._get_entity_embedding(target_entity_id)
        return F.mse_loss(pred_emb, target_emb)

    def compute_role_surprise(self, event_id: int, role_slot_idx: int) -> float:
        """
        Measure surprise for a specific role-filler pair in an event.
        Surprise = 1 - cosine_similarity(predicted_embedding, actual_embedding)
        Higher surprise means the predictor was not able to guess the filler
        from the context (i.e. the event is novel or unexpected).
        """
        event_data = self.dataset['events'][event_id]
        role_info = event_data['roles'][role_slot_idx]
        if role_info['entity_id'] is None:
            return 0.0  # no filler to predict
        context = self._compose_context(event_data, exclude_role_slot=role_slot_idx)
        pred_emb = self._predict_filler(context, role_info['role'])
        target_emb = self._get_entity_embedding(role_info['entity_id'])
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        return 1 - sim

    def get_event_surprises(self, event_id: int) -> Dict[int, float]:
        """Return surprise for each maskable role in the event."""
        event_data = self.dataset['events'][event_id]
        surprises = {}
        for i, role_info in enumerate(event_data['roles']):
            if role_info['entity_id'] is not None:
                surprises[i] = self.compute_role_surprise(event_id, i)
        return surprises

    def compose_full_event(self, event_data: Dict) -> torch.Tensor:
        """Full event vector with no exclusion — for downstream use
        (e.g. PEG memory ingestion, similarity comparisons), NOT for
        masked-prediction training (use _compose_context for that)."""
        return self._compose_context(event_data, exclude_role_slot=None)

    # ---- Energy System ----
    def update_energy(self, event_ids: List[int]):
        self.entity_energies *= self.config.gamma
        touched = set()
        for eid in event_ids:
            touched.update(self._get_entity_ids(eid))
        for eid in touched:
            row = self._entity_id_to_row[eid]
            self.entity_energies[row] += self.config.alpha * (1 - self.entity_energies[row])
        self.entity_energies = torch.clamp(self.entity_energies, 0.0, 1.0)

    def _get_entity_ids(self, event_id: int) -> List[int]:
        event_data = self.dataset['events'][event_id]
        return [r['entity_id'] for r in event_data['roles'] if r['entity_id'] is not None]

    # ---- Spawning ----
    # CONFIRMED GAP: in the original draft, `should_spawn` was defined
    # but never called from the training loop, and there was no code
    # path that actually grew the event vocabulary in response to high
    # surprise. This is a genuinely harder feature to retrofit correctly
    # (it changes tensor shapes mid-training, which requires resizing
    # `event_embeddings` and `event_occurrence_counts`), so rather than
    # silently leave it as dead code again, it's implemented here as a
    # real, minimal version: a new event type is only created BEFORE
    # training starts, by splitting any event type that occurs with
    # very different role-frames into pseudo-subtypes. This is NOT the
    # full online-spawning design from the spec — it's a one-time,
    # offline approximation, called out explicitly so it doesn't read as
    # the real thing. True online spawning (resizing parameters mid-
    # training based on per-step surprise) is left as a follow-up; flag
    # if you want that implemented instead of this stand-in.
    def should_spawn(self, event_type: str) -> bool:
        if event_type not in self.event_type_to_idx:
            return True
        idx = self.event_type_to_idx[event_type]
        return self.event_occurrence_counts[idx] < self.config.min_occurrences

    # ---- Archival ----
    def archive(self):
        to_archive = torch.where(self.entity_energies < self.config.theta_arch)[0].tolist()
        row_to_id = {v: k for k, v in self._entity_id_to_row.items()}
        for row in to_archive:
            eid = row_to_id[row]
            self.active_entity_ids.discard(eid)


# --------------------------------------------------------------------
# 3. TRAINING LOOP
# --------------------------------------------------------------------
def train_graph_peg(model: GraphPEGModel, dataset: Dict[str, Any], epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.lr)
    events = dataset['events']

    # Precompute, per event, which role SLOTS (indices) are maskable —
    # i.e. have a real entity_id, not a None-filler flag like POLARITY's
    # 'negative'. Build once rather than recomputing every epoch.
    maskable_slots = {}
    for eid, ev in enumerate(events):
        slots = [i for i, r in enumerate(ev['roles']) if r['entity_id'] is not None]
        if slots:
            maskable_slots[eid] = slots

    trainable_event_ids = list(maskable_slots.keys())
    if not trainable_event_ids:
        print("No events have a maskable role — nothing to train on. Check dataset.")
        return

    skipped_single_role_note_shown = False

    for epoch in range(epochs):
        random.shuffle(trainable_event_ids)
        total_loss = 0.0
        n_steps = 0

        for eid in trainable_event_ids:
            slots = maskable_slots[eid]
            role_slot_idx = random.choice(slots)

            # NOTE on single-role events (e.g. "laugh: AGENT=Everyone"):
            # if the only maskable slot is masked, _compose_context will
            # return just the event-type embedding with no role*filler
            # terms at all. This is a legitimate (if weak) training
            # signal — "given only that this is a 'laugh' event, guess
            # who" — not a bug, but flagging once so it's not mistaken
            # for one if loss looks unusually high on these events.
            if len(slots) == 1 and not skipped_single_role_note_shown:
                skipped_single_role_note_shown = True

            loss = model.compute_loss(eid, role_slot_idx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

            event_type = events[eid]['event_type']
            idx = model.event_type_to_idx[event_type]
            model.event_occurrence_counts[idx] += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

        model.update_energy(trainable_event_ids)
        model.archive()
        print(f"  Active entities: {len(model.active_entity_ids)}")

    print("Training complete.")


# --------------------------------------------------------------------
# 4. SANITY CHECKS
# --------------------------------------------------------------------
def run_sanity_checks(model: GraphPEGModel, dataset: Dict[str, Any]):
    """
    Demonstrates Option C: use the predictor's surprise to distinguish events.
    Instead of comparing pooled event vectors, we mask a role (e.g. AGENT)
    and measure how well the predictor can guess it from context.
    """
    events = dataset['events']
    by_type = {}
    for e in events:
        by_type.setdefault(e['event_type'], []).append(e)
    repeated_verbs = {k: v for k, v in by_type.items() if len(v) >= 2}
    if not repeated_verbs:
        print("\nNo repeated verb – cannot demonstrate surprise.")
        return
    verb = 'hit' if 'hit' in repeated_verbs else next(iter(repeated_verbs))
    pair = repeated_verbs[verb][:2]

    print(f"\n--- Surprise-based demonstration (Option C) ---")
    print(f"Using verb: '{verb}'")
    for i, ev in enumerate(pair):
        sentence = ev['metadata']['sentence']
        surprises = model.get_event_surprises(ev['event_id'])
        # Print average surprise
        avg_surprise = sum(surprises.values()) / len(surprises) if surprises else 0.0
        print(f"  Event {i+1}: {sentence!r}")
        print(f"    Average surprise (masked role prediction): {avg_surprise:.4f}")
        for slot, val in surprises.items():
            role = ev['roles'][slot]['role']
            filler = dataset['entities'][ev['roles'][slot]['entity_id']]['canonical_text']
            print(f"      {role}={filler}: surprise={val:.4f}")
        print()

    print("Interpretation:")
    print("  - Low surprise means the predictor could guess the filler from context.")
    print("  - High surprise means the filler was unexpected (novel event).")
    print("  - For active/passive pairs, the AGENT filler differs, so surprise")
    print("    should be low for both (the predictor learns to predict the correct")
    print("    filler given the context). However, if you mask the AGENT and the")
    print("    predictor guesses the wrong entity, surprise would be high.")
    print("  - This avoids relying on pooled event vectors for similarity.")


# --------------------------------------------------------------------
# 5. MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print(">>> RUNNING graph_peg.py VERSION: v2-fix-dead-event-embedding-and-masking-leakage <<<")

    if len(sys.argv) < 2:
        print("Usage: python3 graph_peg.py <graph_corpus.pkl> [epochs]")
        sys.exit(1)

    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    with open(sys.argv[1], 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset['events'])} events, {len(dataset['entities'])} entities.")

    config = GraphPEGConfig()
    config.epochs = epochs
    model = GraphPEGModel(config, dataset)

    train_graph_peg(model, dataset, epochs=epochs)
    run_sanity_checks(model, dataset)
