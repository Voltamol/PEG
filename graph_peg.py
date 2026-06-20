#!/usr/bin/env python3
# graph_peg.py — Graph‑PEG engine v11 (no verb embedding, pure roles)
# VERSION MARKER: v11-no-verb-embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np
from typing import List, Dict, Optional, Any
from torch.optim.lr_scheduler import StepLR

# --------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------
class GraphPEGConfig:
    def __init__(self):
        self.hidden_dim = 512
        self.role_dim = 64
        self.lr = 5e-4
        self.weight_decay = 1e-5
        self.epochs = 200
        self.gamma = 0.995
        self.alpha = 0.1
        self.beta = 0.05
        self.theta_high = 0.35
        self.theta_arch = 0.10
        self.min_occurrences = 3
        self.merge_threshold = 0.95
        self.mask_prob = 0.15
        self.num_negatives = 20
        self.dropout = 0.2
        self.temperature = 0.05
        self.noise_std = 0.05
        self.augment_prob = 0.5


# --------------------------------------------------------------------
# 2. MODEL
# --------------------------------------------------------------------
class GraphPEGModel(nn.Module):
    def __init__(self, config: GraphPEGConfig, dataset: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dataset = dataset

        # vocab
        self.role_to_idx = {
            'AGENT': 0, 'PATIENT': 1, 'RECIPIENT': 2, 'LOCATION': 3,
            'TIME': 4, 'MANNER': 5, 'MODIFIER': 6, 'POLARITY': 7, 'MOOD': 8
        }
        self.idx_to_role = {v: k for k, v in self.role_to_idx.items()}
        self.num_roles = len(self.role_to_idx)

        self.event_type_to_idx = dict(dataset['event_types'])
        self.num_event_types = len(self.event_type_to_idx)

        # embeddings – event embeddings are still kept for compatibility but NOT used in context
        self.event_embeddings = nn.Parameter(
            torch.randn(self.num_event_types, config.hidden_dim) * 0.01
        )
        self.role_embeddings = nn.Parameter(
            torch.randn(self.num_roles, config.role_dim) * 0.01
        )
        self.role_proj = nn.Linear(config.role_dim, config.hidden_dim)

        # ---- Bilinear predictor ----
        self.role_predictors = nn.ModuleDict({
            role: nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
            for role in self.role_to_idx.keys()
        })

        # entity projection
        sample_ent = next(iter(self.dataset['entities'].values()))
        sample_emb = sample_ent['mentions'][0]['embedding']
        self.embed_dim = sample_emb.shape[0]
        self.entity_proj = nn.Linear(self.embed_dim, config.hidden_dim)

        # entity embeddings buffer
        self.register_buffer('entity_embeddings', self._build_entity_embeddings())

        # dynamic state
        self.entity_energies = torch.ones(len(self.dataset['entities'])) * 0.5
        self.event_occurrence_counts = torch.zeros(self.num_event_types, dtype=torch.long)
        self.active_event_type_idxs = set(range(self.num_event_types))
        self.active_entity_ids = set(range(len(self.dataset['entities'])))

        sorted_ids = sorted(self.dataset['entities'].keys())
        self._entity_id_to_row = {eid: i for i, eid in enumerate(sorted_ids)}
        self.all_entity_ids = list(self.dataset['entities'].keys())

        # pre‑compute role‑entity map for hard negatives and augmentation
        self._build_role_entity_map()
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
        return torch.stack(emb_list)

    def _build_role_entity_map(self):
        self.role_entity_map = {}
        self.role_to_entities = {}
        for event in self.dataset['events']:
            verb = event['event_type']
            for role_info in event['roles']:
                if role_info['entity_id'] is None:
                    continue
                role = role_info['role']
                eid = role_info['entity_id']
                key = (verb, role)
                if key not in self.role_entity_map:
                    self.role_entity_map[key] = []
                self.role_entity_map[key].append(eid)
                if role not in self.role_to_entities:
                    self.role_to_entities[role] = []
                self.role_to_entities[role].append(eid)
        for k in self.role_entity_map:
            self.role_entity_map[k] = list(set(self.role_entity_map[k]))
        for role in self.role_to_entities:
            self.role_to_entities[role] = list(set(self.role_to_entities[role]))

    def _get_role_embedding(self, role_name: str) -> torch.Tensor:
        idx = self.role_to_idx[role_name]
        return self.role_embeddings[idx]

    def _get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        row = self._entity_id_to_row[entity_id]
        raw_emb = self.entity_embeddings[row]
        return self.entity_proj(raw_emb)

    def _compose_context(self, event_data: Dict, exclude_role_slot: Optional[int] = None,
                         augment: bool = False) -> torch.Tensor:
        """
        Build context from ONLY role-entity products – NO verb embedding.
        """
        vec = torch.zeros(self.config.hidden_dim, device=self.event_embeddings.device)
        for i, role_info in enumerate(event_data['roles']):
            if i == exclude_role_slot:
                continue
            if role_info['entity_id'] is None:
                continue
            role = role_info['role']
            original_eid = role_info['entity_id']
            if augment and random.random() < self.config.augment_prob:
                candidates = [eid for eid in self.role_to_entities.get(role, []) if eid != original_eid]
                if candidates:
                    original_eid = random.choice(candidates)
            entity_emb = self._get_entity_embedding(original_eid)
            role_emb = self.role_proj(self._get_role_embedding(role))
            vec = vec + role_emb * entity_emb
        return vec

    def _predict_filler(self, context_vec: torch.Tensor, role: str) -> torch.Tensor:
        return self.role_predictors[role](context_vec)

    # ---- InfoNCE loss with hard negatives and augmentation ----
    def compute_loss(self, event_id: int, role_slot_idx: int) -> torch.Tensor:
        event_data = self.dataset['events'][event_id]
        masked_role = event_data['roles'][role_slot_idx]
        role_name = masked_role['role']
        target_entity_id = masked_role['entity_id']
        verb = event_data['event_type']   # only for hard negative selection

        context = self._compose_context(event_data, exclude_role_slot=role_slot_idx,
                                        augment=self.training)
        if self.training:
            noise = torch.randn_like(context) * self.config.noise_std
            context = context + noise

        pred = self._predict_filler(context, role_name)   # (hidden_dim,)
        pos_emb = self._get_entity_embedding(target_entity_id)

        # Hard negatives: same verb-role (still use verb-role mapping)
        hard_pool = self.role_entity_map.get((verb, role_name), [])
        hard_neg_ids = [eid for eid in hard_pool if eid != target_entity_id]
        if len(hard_neg_ids) > self.config.num_negatives:
            hard_neg_ids = random.sample(hard_neg_ids, self.config.num_negatives)
        elif len(hard_neg_ids) > 0:
            random_pool = [eid for eid in self.all_entity_ids if eid != target_entity_id and eid not in hard_neg_ids]
            if random_pool:
                extra = random.sample(random_pool, min(self.config.num_negatives - len(hard_neg_ids), len(random_pool)))
                hard_neg_ids.extend(extra)
        else:
            random_pool = [eid for eid in self.all_entity_ids if eid != target_entity_id]
            hard_neg_ids = random.sample(random_pool, min(self.config.num_negatives, len(random_pool)))

        neg_embs = torch.stack([self._get_entity_embedding(eid) for eid in hard_neg_ids])

        # Normalize
        pred_norm = F.normalize(pred, dim=0)
        pos_norm = F.normalize(pos_emb, dim=0)
        neg_norm = F.normalize(neg_embs, dim=1)

        pos_sim = torch.dot(pred_norm, pos_norm)
        neg_sim = torch.mv(neg_norm, pred_norm)

        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim]) / self.config.temperature
        labels = torch.zeros(1, dtype=torch.long, device=pred.device)
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss

    # ---- Surprise ----
    def compute_role_surprise(self, event_id: int, role_slot_idx: int) -> float:
        event_data = self.dataset['events'][event_id]
        role_info = event_data['roles'][role_slot_idx]
        if role_info['entity_id'] is None:
            return 0.0
        context = self._compose_context(event_data, exclude_role_slot=role_slot_idx, augment=False)
        pred = self._predict_filler(context, role_info['role'])
        target = self._get_entity_embedding(role_info['entity_id'])
        sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
        return 1 - sim

    def get_event_surprises(self, event_id: int) -> Dict[int, float]:
        event_data = self.dataset['events'][event_id]
        surprises = {}
        for i, role_info in enumerate(event_data['roles']):
            if role_info['entity_id'] is not None:
                surprises[i] = self.compute_role_surprise(event_id, i)
        return surprises

    def compose_full_event(self, event_data: Dict) -> torch.Tensor:
        return self._compose_context(event_data, exclude_role_slot=None, augment=False)

    # ---- Energy ----
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

    def should_spawn(self, event_type: str) -> bool:
        if event_type not in self.event_type_to_idx:
            return True
        idx = self.event_type_to_idx[event_type]
        return self.event_occurrence_counts[idx] < self.config.min_occurrences

    def archive(self):
        to_archive = torch.where(self.entity_energies < self.config.theta_arch)[0].tolist()
        row_to_id = {v: k for k, v in self._entity_id_to_row.items()}
        for row in to_archive:
            eid = row_to_id[row]
            self.active_entity_ids.discard(eid)


# --------------------------------------------------------------------
# 3. TRAINING LOOP
# --------------------------------------------------------------------
def train_graph_peg(model: GraphPEGModel, dataset: Dict[str, Any], epochs=200):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=model.config.lr,
                                 weight_decay=model.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.7)

    events = dataset['events']
    maskable_slots = {}
    for eid, ev in enumerate(events):
        slots = [i for i, r in enumerate(ev['roles']) if r['entity_id'] is not None]
        if slots:
            maskable_slots[eid] = slots

    trainable_event_ids = list(maskable_slots.keys())
    if not trainable_event_ids:
        print("No events have a maskable role — nothing to train on.")
        return

    for epoch in range(epochs):
        random.shuffle(trainable_event_ids)
        total_loss = 0.0
        n_steps = 0

        for eid in trainable_event_ids:
            slots = maskable_slots[eid]
            role_slot_idx = random.choice(slots)

            model.train()
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
        scheduler.step()

        model.update_energy(trainable_event_ids)
        model.archive()
        print(f"  Active entities: {len(model.active_entity_ids)}")

    print("Training complete.")


# --------------------------------------------------------------------
# 4. SANITY CHECKS (unchanged)
# --------------------------------------------------------------------
def run_sanity_checks(model: GraphPEGModel, dataset: Dict[str, Any]):
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
        avg_surprise = sum(surprises.values()) / len(surprises) if surprises else 0.0
        print(f"  Event {i+1}: {sentence!r}")
        print(f"    Average surprise: {avg_surprise:.4f}")
        for slot, val in surprises.items():
            role = ev['roles'][slot]['role']
            filler = dataset['entities'][ev['roles'][slot]['entity_id']]['canonical_text']
            print(f"      {role}={filler}: surprise={val:.4f}")
        print()

    print("Interpretation:")
    print("  - Low surprise means the predictor could guess the filler from context.")
    print("  - High surprise means the filler was unexpected (novel event).")
    print("  - This avoids relying on pooled event vectors for similarity.")


# --------------------------------------------------------------------
# 5. NOVELTY TEST (fixed)
# --------------------------------------------------------------------
def test_novelty(model: GraphPEGModel, dataset: Dict[str, Any]):
    import spacy
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*60)
    print("NOVELTY TEST (Option C in action)")
    print("="*60)

    nlp = spacy.load('en_core_web_sm')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_surprise_for_new_event(event_type: str, roles: List[Dict]):
        with torch.no_grad():
            # Note: verb embedding is NOT used in context, but we need it for status
            if event_type in model.event_type_to_idx:
                verb_status = "KNOWN"
            else:
                verb_status = "NEW (using average)"
            print(f"  Verb: '{event_type}' [{verb_status}]")

            entity_projs = {}
            for r in roles:
                text = r['entity_text']
                raw_emb = encoder.encode([text], convert_to_tensor=True).squeeze(0)
                raw_emb = raw_emb.clone()
                proj_emb = model.entity_proj(raw_emb.to(model.event_embeddings.device))
                entity_projs[text] = proj_emb

            surprises = {}
            for i, r in enumerate(roles):
                # Build context from other roles ONLY (no verb embedding)
                context_vec = torch.zeros(model.config.hidden_dim, device=model.event_embeddings.device)
                for j, other in enumerate(roles):
                    if i == j:
                        continue
                    role_emb = model.role_proj(model._get_role_embedding(other['role']))
                    entity_emb = entity_projs[other['entity_text']]
                    context_vec = context_vec + role_emb * entity_emb

                pred = model._predict_filler(context_vec, r['role'])
                target = entity_projs[r['entity_text']]
                sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
                surprise = 1 - sim
                surprises[r['role']] = surprise

            return surprises, verb_status

    # --- Test 1: Known verb + new modifier ---
    print("\n--- Test 1: Known verb 'be' + NEW modifier 'carpenter' ---")
    new_sentence1 = "Leo was a carpenter."
    doc1 = nlp(new_sentence1)
    root1 = [t for t in doc1 if t.dep_ == 'ROOT'][0]
    event_type1 = root1.lemma_
    roles1 = []
    for child in root1.children:
        if child.dep_ == 'nsubj':
            roles1.append({'role': 'AGENT', 'entity_text': child.text})
        elif child.dep_ in ('attr', 'acomp'):
            roles1.append({'role': 'MODIFIER', 'entity_text': child.text})
    print(f"  Sentence: {new_sentence1!r}")
    surprises1, status1 = compute_surprise_for_new_event(event_type1, roles1)
    for role, val in surprises1.items():
        print(f"    {role}: surprise={val:.4f}")

    # --- Test 2: New verb + known filler ---
    print("\n--- Test 2: NEW verb 'dance' + known filler 'John' ---")
    new_sentence2 = "John danced."
    doc2 = nlp(new_sentence2)
    root2 = [t for t in doc2 if t.dep_ == 'ROOT'][0]
    event_type2 = root2.lemma_
    roles2 = []
    for child in root2.children:
        if child.dep_ == 'nsubj':
            roles2.append({'role': 'AGENT', 'entity_text': child.text})
    print(f"  Sentence: {new_sentence2!r}")
    surprises2, status2 = compute_surprise_for_new_event(event_type2, roles2)
    for role, val in surprises2.items():
        print(f"    {role}: surprise={val:.4f}")

    # --- Test 3: NEW verb + NEW entities (manual) ---
    print("\n--- Test 3: NEW verb 'arrive' + NEW entities 'Zorp' and 'alien' ---")
    new_sentence3 = "Zorp the alien arrived."
    event_type3 = "arrive"
    roles3 = [
        {'role': 'AGENT', 'entity_text': 'Zorp'},
        {'role': 'MODIFIER', 'entity_text': 'alien'}
    ]
    print(f"  Sentence: {new_sentence3!r}")
    print(f"  Extracted roles: {[(r['role'], r['entity_text']) for r in roles3]}")
    surprises3, status3 = compute_surprise_for_new_event(event_type3, roles3)
    for role, val in surprises3.items():
        print(f"    {role}: surprise={val:.4f}")

    # --- Interpretation ---
    print("\n--- Interpretation ---")
    print(f"  Test 1 (carpenter): MODIFIER surprise = {surprises1.get('MODIFIER', 0.0):.4f}")
    print(f"  Test 2 (dance):    AGENT surprise    = {surprises2.get('AGENT', 0.0):.4f}")
    print(f"  Test 3 (Zorp):     AGENT surprise    = {surprises3.get('AGENT', 0.0):.4f}")
    print(f"  Test 3 (alien):    MODIFIER surprise = {surprises3.get('MODIFIER', 0.0):.4f}")
    print()
    print("  Expected: Test 1 < 0.2 (known verb, new modifier should generalise).")
    print("            Test 2 > 0.2 (new verb, surprise should be moderate).")
    print("            Test 3 > 0.3 (completely new event – highest surprise).")
    print("  This version removes the verb embedding from context to force role-based generalisation.")


# --------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print(">>> RUNNING graph_peg.py VERSION: v11-no-verb-embedding <<<")

    if len(sys.argv) < 2:
        print("Usage: python3 graph_peg.py <graph_corpus.pkl> [epochs]")
        sys.exit(1)

    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    with open(sys.argv[1], 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset['events'])} events, {len(dataset['entities'])} entities.")

    config = GraphPEGConfig()
    config.epochs = epochs
    model = GraphPEGModel(config, dataset)

    train_graph_peg(model, dataset, epochs=epochs)
    run_sanity_checks(model, dataset)
    test_novelty(model, dataset)