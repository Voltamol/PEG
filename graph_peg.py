#!/usr/bin/env python3
# graph_peg_v16.py — Linear predictor + InfoNCE cosine similarity
# VERSION MARKER: v16-linear-infonce

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
        self.hidden_dim = 256
        self.role_dim = 64
        self.lr = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 100
        self.dropout = 0.1
        self.noise_std = 0.05
        self.batch_size = 32
        self.verb_weight = 0.2
        self.temperature = 0.1
        self.num_negatives = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        # embeddings
        self.event_embeddings = nn.Parameter(
            torch.randn(self.num_event_types, config.hidden_dim) * 0.01
        )
        self.role_embeddings = nn.Parameter(
            torch.randn(self.num_roles, config.role_dim) * 0.01
        )
        self.role_proj = nn.Linear(config.role_dim, config.hidden_dim)

        # Linear predictor: context+role -> hidden_dim
        self.predictor = nn.Linear(config.hidden_dim + config.role_dim, config.hidden_dim)

        # entity projection
        sample_ent = next(iter(self.dataset['entities'].values()))
        sample_emb = sample_ent['mentions'][0]['embedding']
        self.embed_dim = sample_emb.shape[0]
        self.entity_proj = nn.Linear(self.embed_dim, config.hidden_dim)

        # move to device
        self.to(config.device)

        # precompute entity embeddings
        self._precompute_entity_embeddings()

        # role‑entity map for hard negatives
        self._build_role_entity_map()

    def _precompute_entity_embeddings(self):
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
        raw_embs = torch.stack(emb_list)
        with torch.no_grad():
            proj_embs = self.entity_proj(raw_embs.to(self.config.device))
        self.register_buffer('entity_embeddings', proj_embs)
        sorted_ids = sorted(entities.keys())
        self._entity_id_to_row = {eid: i for i, eid in enumerate(sorted_ids)}
        self.all_entity_ids = list(entities.keys())

    def _build_role_entity_map(self):
        self.role_entity_map = {}
        for event in self.dataset['events']:
            verb = event['event_type']
            for role_info in event['roles']:
                if role_info['entity_id'] is None:
                    continue
                role = role_info['role']
                key = (verb, role)
                if key not in self.role_entity_map:
                    self.role_entity_map[key] = []
                self.role_entity_map[key].append(role_info['entity_id'])
        for k in self.role_entity_map:
            self.role_entity_map[k] = list(set(self.role_entity_map[k]))

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
        verb_emb = self._get_event_type_embedding(event_data['event_type'])
        role_sum = torch.zeros(self.config.hidden_dim, device=self.config.device)
        for i, role_info in enumerate(event_data['roles']):
            if i == exclude_role_slot:
                continue
            if role_info['entity_id'] is None:
                continue
            role = role_info['role']
            entity_id = role_info['entity_id']
            role_emb = self.role_proj(self._get_role_embedding(role))
            entity_emb = self._get_entity_embedding(entity_id)
            role_sum = role_sum + role_emb * entity_emb
        return self.config.verb_weight * verb_emb + (1 - self.config.verb_weight) * role_sum

    def _predict_filler(self, context_vec: torch.Tensor, role: str) -> torch.Tensor:
        role_emb = self._get_role_embedding(role)
        combined = torch.cat([context_vec, role_emb], dim=-1)
        return self.predictor(combined)   # (hidden_dim,)

    def compute_loss(self, event_id: int, role_slot_idx: int) -> torch.Tensor:
        event_data = self.dataset['events'][event_id]
        masked_role = event_data['roles'][role_slot_idx]
        role_name = masked_role['role']
        target_entity_id = masked_role['entity_id']
        verb = event_data['event_type']

        context = self._compose_context(event_data, exclude_role_slot=role_slot_idx)
        if self.training:
            noise = torch.randn_like(context) * self.config.noise_std
            context = context + noise

        pred = self._predict_filler(context, role_name)   # (hidden_dim,)
        pos_emb = self._get_entity_embedding(target_entity_id)

        # Negatives: mix of hard and random
        hard_pool = self.role_entity_map.get((verb, role_name), [])
        hard_neg_ids = [eid for eid in hard_pool if eid != target_entity_id]
        random_pool = [eid for eid in self.all_entity_ids if eid != target_entity_id and eid not in hard_neg_ids]

        num_hard = min(len(hard_neg_ids), self.config.num_negatives // 2)
        num_random = self.config.num_negatives - num_hard

        if len(hard_neg_ids) > num_hard:
            sampled_hard = random.sample(hard_neg_ids, num_hard)
        else:
            sampled_hard = hard_neg_ids
            num_random = self.config.num_negatives - len(sampled_hard)

        if len(random_pool) > num_random:
            sampled_random = random.sample(random_pool, num_random)
        else:
            sampled_random = random_pool

        neg_ids = sampled_hard + sampled_random
        if len(neg_ids) < 1:
            neg_ids = random.sample([eid for eid in self.all_entity_ids if eid != target_entity_id],
                                    min(self.config.num_negatives, len(self.all_entity_ids)-1))

        neg_embs = torch.stack([self._get_entity_embedding(eid) for eid in neg_ids])  # (K, hidden_dim)

        # Normalize all vectors
        pred_norm = F.normalize(pred, dim=0)
        pos_norm = F.normalize(pos_emb, dim=0)
        neg_norm = F.normalize(neg_embs, dim=1)

        pos_sim = torch.dot(pred_norm, pos_norm)
        neg_sim = torch.mv(neg_norm, pred_norm)   # (K,)

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
        context = self._compose_context(event_data, exclude_role_slot=role_slot_idx)
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


# --------------------------------------------------------------------
# 3. TRAINING LOOP (same as before)
# --------------------------------------------------------------------
def train_graph_peg(model: GraphPEGModel, dataset: Dict[str, Any], epochs=100):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=model.config.lr,
                                 weight_decay=model.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

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

    batch_size = model.config.batch_size
    for epoch in range(epochs):
        random.shuffle(trainable_event_ids)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for i, eid in enumerate(trainable_event_ids):
            slots = maskable_slots[eid]
            role_slot_idx = random.choice(slots)
            loss = model.compute_loss(eid, role_slot_idx)
            loss = loss / batch_size
            loss.backward()

            if (i + 1) % batch_size == 0 or (i + 1) == len(trainable_event_ids):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * batch_size
                n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")
        scheduler.step()

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
# 5. NOVELTY TEST (adapted)
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
            if event_type in model.event_type_to_idx:
                verb_vec = model._get_event_type_embedding(event_type)
                verb_status = "KNOWN"
            else:
                verb_vec = torch.zeros(model.config.hidden_dim, device=model.config.device)
                verb_status = "NEW (using zero)"
            print(f"  Verb: '{event_type}' [{verb_status}]")

            entity_projs = {}
            for r in roles:
                text = r['entity_text']
                raw_emb = encoder.encode([text], convert_to_tensor=True).squeeze(0)
                raw_emb = raw_emb.clone()
                proj_emb = model.entity_proj(raw_emb.to(model.config.device))
                entity_projs[text] = proj_emb

            surprises = {}
            for i, r in enumerate(roles):
                role_sum = torch.zeros(model.config.hidden_dim, device=model.config.device)
                for j, other in enumerate(roles):
                    if i == j:
                        continue
                    role_emb = model.role_proj(model._get_role_embedding(other['role']))
                    entity_emb = entity_projs[other['entity_text']]
                    role_sum = role_sum + role_emb * entity_emb
                context_vec = model.config.verb_weight * verb_vec + (1 - model.config.verb_weight) * role_sum

                pred = model._predict_filler(context_vec, r['role'])
                target = entity_projs[r['entity_text']]
                sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
                surprise = 1 - sim
                surprises[r['role']] = surprise

            return surprises, verb_status

    # --- Test 1 ---
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

    # --- Test 2 ---
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

    # --- Test 3 ---
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


# --------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print(">>> RUNNING graph_peg.py VERSION: v16-linear-infonce <<<")

    if len(sys.argv) < 2:
        print("Usage: python3 graph_peg_v16.py <graph_corpus.pkl> [epochs]")
        sys.exit(1)

    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    with open(sys.argv[1], 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset['events'])} events, {len(dataset['entities'])} entities.")

    config = GraphPEGConfig()
    config.epochs = epochs
    model = GraphPEGModel(config, dataset)

    train_graph_peg(model, dataset, epochs=epochs)
    run_sanity_checks(model, dataset)
    test_novelty(model, dataset)