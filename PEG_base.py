# -*- coding: utf-8 -*-
"""
PEG — Predictive Energy Grounding with Transformer Decoder
Simplified version using the Eldoria story (synthetic, clean ground truth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import math
import nltk
from nltk.corpus import words as nltk_words
from story_corpus import story_corpus
# ============================================================================
# 1. CORPUS – ELDORIA (60 sentences, known concepts)
# ============================================================================

print(f"Loaded Eldoria corpus: {len(story_corpus)} sentences.")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
@dataclass
class PEGConfig:
    d: int = 256                     # latent dimension
    hidden_size: int = 512           # MLP hidden size (Predictor)
    gamma: float = 0.995             # energy decay
    alpha: float = 0.1               # energy boost
    beta: float = 0.05               # lateral inhibition
    theta_high: float = 0.35       # higher = fewer spawns
    theta_novel: float = 0.5       # higher = fewer spawns
    theta_arch: float = 0.15       # higher = more aggressive pruning
    lambda_pred: float = 1.0         # predictive loss weight
    ontology_size: int = 50000       # number of words in ontology
    ablate_energy: bool = False      # if True, disables decay, inhibition, merge, archive

# ============================================================================
# 3. PEG MODEL
# ============================================================================
class ResidualProjection(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d) * 0.9 + torch.randn(d, d) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class PredictiveModule(nn.Module):
    def __init__(self, config: PEGConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.d)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    def forward(self, context):
        return self.fc2(F.gelu(self.fc1(context)))

@dataclass
class Slot:
    vec: torch.Tensor
    energy: float = 0.8
    age: int = 0
    anchor_idx: int = -1
    def __eq__(self, other):
        return isinstance(other, Slot) and id(self) == id(other)

class PEGModel(nn.Module):
    def __init__(self, config: PEGConfig, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # Frozen encoder (MiniLM)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.embed_dim = self.encoder.get_embedding_dimension()
        self.proj_to_d = nn.Linear(self.embed_dim, config.d) if self.embed_dim != config.d else nn.Identity()

        # Frozen ontology (will be replaced with real embeddings)
        self.register_buffer('ontology', F.normalize(torch.randn(config.ontology_size, config.d), dim=-1))

        # Trainable modules
        self.P = PredictiveModule(config)
        self.Psi = ResidualProjection(config.d)
        self.ghost = nn.Parameter(torch.zeros(config.d))
        nn.init.normal_(self.ghost, mean=0, std=0.01)

        self.active_slots: List[Slot] = []
        self.to(device)

    def _anchor(self, z):
        z_norm = F.normalize(z.unsqueeze(0), dim=-1)
        sims = z_norm @ self.ontology.T
        idx = min(sims.argmax(dim=-1).item(), self.ontology.shape[0] - 1)
        return self.ontology[idx], sims[0, idx].item()

    def _spawn_slot(self, z):
        anchor_vec, sim = self._anchor(z)
        if sim > 0.6:
            vec = self.Psi(anchor_vec)
            anchor_idx = (z @ self.ontology.T).argmax().item()
        else:
            ghost_proj = self.Psi(self.ghost)
            vec = ghost_proj + (z - ghost_proj)
            anchor_idx = -1
        if self.active_slots:
            avg_norm = torch.mean(torch.stack([s.vec.norm() for s in self.active_slots]))
            vec = vec / vec.norm() * avg_norm
        return Slot(vec=vec.detach(), energy=0.8, age=0, anchor_idx=anchor_idx)

    def _soft_bind(self, z, slots):
        if not slots:
            return z, []
        slot_vecs = torch.stack([s.vec for s in slots]).to(self.device)
        z_norm = F.normalize(z.unsqueeze(0), dim=-1)
        scores = (z_norm @ F.normalize(slot_vecs, dim=-1).T).squeeze(0)
        weights = F.softmax(scores / 0.1, dim=0)
        top_vals, top_idx = torch.topk(weights, min(2, len(weights)))
        for i, w in zip(top_idx, top_vals):
            if w > 0.1:
                slots[i].vec += 0.01 * w * (z - slots[i].vec)
                slots[i].vec = F.normalize(slots[i].vec, dim=0) * 1.0
        z_explained = torch.sum(weights.unsqueeze(1) * slot_vecs, dim=0)
        return z - z_explained, weights.tolist()

    def _update_energy(self, binding_weights):
        if self.config.ablate_energy:
            # ---- ABLATION: only boost, no decay, no inhibition, no merge, no archive ----
            # Boost slots that were bound
            for weights in binding_weights:
                for slot, w in zip(self.active_slots, weights):
                    if w > 0.1:
                        slot.energy += self.config.alpha * (1 - slot.energy)
                        slot.age += 1
            # Clamp energies
            for slot in self.active_slots:
                slot.energy = max(0.0, min(1.0, slot.energy))
            # No decay, no inhibition, no merge, no archive
            return

        # ---- ORIGINAL ENERGY SYSTEM (unchanged) ----
        # Decay and boost
        for slot in self.active_slots:
            slot.energy *= self.config.gamma
        for weights in binding_weights:
            for slot, w in zip(self.active_slots, weights):
                if w > 0.1:
                    slot.energy += self.config.alpha * (1 - slot.energy)
                    slot.age += 1
        for slot in self.active_slots:
            slot.energy = max(0.0, min(1.0, slot.energy))

        # Vectorised lateral inhibition
        if len(self.active_slots) > 1:
            slot_matrix = torch.stack([s.vec for s in self.active_slots]).to(self.device)
            normed = F.normalize(slot_matrix, dim=-1)
            sim_matrix = normed @ normed.T
            mask = (sim_matrix > 0.8) & ~torch.eye(len(self.active_slots), device=self.device).bool()
            inhibition_factor = (sim_matrix * mask.float()).sum(dim=0)
            for j, slot in enumerate(self.active_slots):
                if inhibition_factor[j] > 0:
                    slot.energy -= self.config.beta * slot.energy * inhibition_factor[j]
                    slot.energy = max(0.0, slot.energy)

        # Vectorised merging (using precomputed sim_matrix)
        if len(self.active_slots) > 1:
            slot_matrix = torch.stack([s.vec for s in self.active_slots]).to(self.device)
            normed = F.normalize(slot_matrix, dim=-1)
            sim_matrix = normed @ normed.T
            to_merge = set()
            for i in range(len(self.active_slots)):
                for j in range(i+1, len(self.active_slots)):
                    if sim_matrix[i, j].item() > 0.95:
                        to_merge.add((i, j))
            for i, j in to_merge:
                s1, s2 = self.active_slots[i], self.active_slots[j]
                merged_vec = (s1.vec * s1.energy + s2.vec * s2.energy) / (s1.energy + s2.energy + 1e-8)
                merged_energy = max(s1.energy, s2.energy)
                self.active_slots[i] = Slot(vec=merged_vec, energy=merged_energy, age=0)
                del self.active_slots[j]

        # Archive low-energy slots
        to_archive = [s for s in self.active_slots if s.energy < self.config.theta_arch and s.age > 50]
        for slot in to_archive:
            self.active_slots.remove(slot)

    def forward(self, sentences: List[str] = None, next_sentences: List[str] = None,
                z_t_raw: torch.Tensor = None, z_t_next_raw: torch.Tensor = None):
        if z_t_raw is not None:
            batch_size = z_t_raw.size(0)
            with torch.no_grad():
                z_t = self.proj_to_d(z_t_raw)
                z_t_next = self.proj_to_d(z_t_next_raw)
        else:
            batch_size = len(sentences)
            with torch.no_grad():
                z_t = self.proj_to_d(torch.tensor(self.encoder.encode(sentences), device=self.device))
                z_t_next = self.proj_to_d(torch.tensor(self.encoder.encode(next_sentences), device=self.device))

        if self.active_slots:
            energies = torch.tensor([s.energy for s in self.active_slots], device=self.device)
            weights = F.softmax(energies, dim=0)
            context = torch.sum(weights.unsqueeze(1) * torch.stack([s.vec for s in self.active_slots]), dim=0)
        else:
            context = torch.zeros(self.config.d, device=self.device)

        z_pred = self.P(context)
        z_pred_norm = F.normalize(z_pred.unsqueeze(0), dim=-1)
        z_t_norm = F.normalize(z_t, dim=-1)
        cosine_sim = (z_pred_norm @ z_t_norm.T).diag()
        surprise = 1 - cosine_sim
        avg_surprise = surprise.mean()

        all_residuals, all_weights = [], []
        for z in z_t:
            z_res, w = self._soft_bind(z, self.active_slots)
            all_residuals.append(z_res)
            all_weights.append(w)

        spawned = False
        for z, z_res, s in zip(z_t, all_residuals, surprise):
            if z_res.norm().item() > self.config.theta_novel and s > self.config.theta_high:
                self.active_slots.append(self._spawn_slot(z))
                spawned = True

        L_pred = F.mse_loss(z_pred.unsqueeze(0).expand(batch_size, -1), z_t_next)
        total_loss = self.config.lambda_pred * L_pred
        self._update_energy(all_weights)

        return {'loss': total_loss, 'pred_loss': L_pred, 'surprise': avg_surprise,
                'n_slots': len(self.active_slots), 'spawned': spawned}

# ============================================================================
# 4. ONTOLOGY LOADING
# ============================================================================
def load_word_ontology(model: PEGModel, word_list_size=50000, batch_size=256):
    nltk.download('words', quiet=True)
    words = nltk_words.words()
    words = list(set([w.lower() for w in words if w.isalpha()]))[:word_list_size]
    print(f"Loaded {len(words)} words for ontology.")
    embeddings = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        emb = model.encoder.encode(batch, convert_to_tensor=True)
        embeddings.append(emb.cpu())
    emb_all = torch.cat(embeddings, dim=0)
    emb_all = F.normalize(emb_all, dim=-1)
    emb_all = model.proj_to_d(emb_all.to(model.device))
    model.ontology = emb_all
    print(f"Ontology shape: {model.ontology.shape}")
    return words

# ============================================================================
# 5. TRAINING PEG
# ============================================================================
def train_peg(model, corpus, epochs=30, lr=1e-3, encode_batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Pre-encoding corpus with frozen SentenceTransformer (one-time cost)...")
    raw_embeddings = []
    with torch.no_grad():
        for i in range(0, len(corpus), encode_batch_size):
            batch = corpus[i:i+encode_batch_size]
            emb = model.encoder.encode(batch, convert_to_tensor=True, device=model.device)
            raw_embeddings.append(emb)
    raw_embeddings = torch.cat(raw_embeddings, dim=0)
    z_curr_all = raw_embeddings[:-1]
    z_next_all = raw_embeddings[1:]
    print(f"Pre-encoded {raw_embeddings.size(0)} sentences.")

    # seed first slot
    with torch.no_grad():
        z0 = model.proj_to_d(raw_embeddings[0].unsqueeze(0)).squeeze(0)
        model.active_slots.append(model._spawn_slot(z0))

    n_steps = len(corpus) - 1
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(n_steps):
            loss_dict = model(z_t_raw=z_curr_all[i:i+1], z_t_next_raw=z_next_all[i:i+1])
            loss = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / n_steps
        print(f"PEG Epoch {epoch:02d} | loss={avg_loss:.4f} | slots={len(model.active_slots)}")
    print("PEG training complete.")

# ============================================================================
# 6. SLOT AUDIT
# ============================================================================
def audit_slots(model, corpus, device, word_list, top_k=10):
    print(f"\nAuditing {len(model.active_slots)} slots...")
    with torch.no_grad():
        raw_emb = []
        for i in range(0, len(corpus), 256):
            batch = corpus[i:i+256]
            raw_emb.append(model.encoder.encode(batch, convert_to_tensor=True, device=device))
        raw_emb = torch.cat(raw_emb, dim=0)
        z_all = model.proj_to_d(raw_emb)

    slot_vecs = torch.stack([s.vec for s in model.active_slots]).to(device)
    sims = F.cosine_similarity(z_all.unsqueeze(1), slot_vecs.unsqueeze(0), dim=-1)
    assigned = sims.argmax(dim=-1)

    for i, slot in enumerate(model.active_slots):
        # nearest ontology label
        with torch.no_grad():
            slot_norm = F.normalize(slot.vec.unsqueeze(0), dim=-1)
            sim_ont = slot_norm @ F.normalize(model.ontology, dim=-1).T
            label = word_list[sim_ont.argmax().item()]

        # top sentences for this slot
        sim_slot = F.cosine_similarity(z_all, slot.vec.unsqueeze(0), dim=-1)
        top_idx = sim_slot.topk(min(top_k, len(corpus))).indices.tolist()
        top_sents = [corpus[i] for i in top_idx]

        # count assigned
        count = (assigned == i).sum().item()

        print(f"\n{'='*60}")
        print(f"SLOT {i:02d} | label: '{label}' | energy={slot.energy:.3f} | assigned={count}/{len(corpus)}")
        print("Top sentences:")
        for j, sent in enumerate(top_sents[:5]):
            print(f"  {j+1}. {sent[:100]}...")
        # sample of assigned sentences
        assigned_indices = (assigned == i).nonzero(as_tuple=True)[0].tolist()
        if assigned_indices:
            sample = [corpus[idx] for idx in assigned_indices[:3]]
            print("Sample assigned:")
            for sent in sample:
                print(f"  - {sent[:100]}...")

# ============================================================================
# 7. DEMO
# ============================================================================
# ============================================================================
# 7. DEMO
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    model = PEGModel(config, device=str(device))

    word_list = load_word_ontology(model, word_list_size=50000)

    # ---------- Choose which corpus to run ----------
    RUN_PRONOM_TEST = True   # Set to False to run the original Eldoria story

    if RUN_PRONOM_TEST:
        # Pronoun-replaced corpus (names → roles)
        corpus = [
            "The boy was a young adventurer from the village of Eldoria.",
            "The girl was his best friend and the smartest person he knew.",
            "The friend was their clumsy but loyal companion.",
            "One day, the boy found an old map in his grandfather's attic.",
            "The map showed the way to the Lost Treasure of Eldoria.",
            "The boy showed the map to the girl and the friend.",
            "The girl studied the map carefully.",
            "She noticed a hidden path through the Whispering Forest.",
            "The friend accidentally tore the map while examining it.",
            "Everyone laughed, but they were still excited.",
            "They packed their bags with food and water.",
            "They set out at sunrise the next morning.",
            "The Whispering Forest was dark and eerie.",
            "Strange sounds echoed through the trees.",
            "The girl used her knowledge of stars to guide them.",
            "The boy hacked through the thick bushes with a knife.",
            "The friend tripped over a root and fell into a muddy puddle.",
            "They finally reached the center of the forest.",
            "In the center stood an ancient stone door.",
            "The door was covered in strange symbols.",
            "The girl realized the symbols were a riddle.",
            "The boy solved the riddle by saying the password aloud.",
            "The stone door creaked open.",
            "Behind the door was a dark cave.",
            "The cave was cold and damp.",
            "They lit a torch to see inside.",
            "The torchlight revealed a narrow tunnel.",
            "They walked through the tunnel for hours.",
            "The friend kept complaining about his wet shoes.",
            "The girl told him to be quiet and listen for danger.",
            "They heard the sound of running water.",
            "They emerged from the tunnel into a massive underground cavern.",
            "Inside the cavern was a raging underground river.",
            "There was no bridge to cross the river.",
            "The boy spotted a broken rope bridge on the other side.",
            "They needed to get across to reach the treasure.",
            "The girl suggested they build a raft from fallen wood.",
            "They worked together to build a sturdy raft.",
            "The raft barely held together as they crossed.",
            "The friend nearly fell into the river twice.",
            "They made it safely to the other side.",
            "There, they found a golden chest.",
            "The chest was locked with a heavy iron lock.",
            "The girl tried to pick the lock with a hairpin.",
            "She managed to open the lock after several attempts.",
            "Inside the chest, there was no gold.",
            "Instead, there was a single, glowing crystal.",
            "The crystal pulsed with a warm, magical light.",
            "The girl knew immediately that the crystal was priceless.",
            "The boy placed the crystal carefully in his backpack.",
            "They decided to head back to Eldoria.",
            "On their way back, they used the raft to cross the river again.",
            "They passed through the tunnel and the stone door.",
            "The Whispering Forest seemed less scary on the way back.",
            "They arrived in Eldoria as heroes.",
            "The village elder congratulated them on their success.",
            "The boy, the girl, and the friend were proud of their adventure.",
            "They placed the glowing crystal in the village square.",
            "The crystal brought good luck and prosperity to Eldoria.",
            "The boy, the girl, and the friend remained best friends forever."
        ]
        print(f"Running PRONOUN REPLACEMENT TEST: {len(corpus)} sentences.")
    else:
        # Original Eldoria story (already imported from story_corpus)
        corpus = story_corpus
        print(f"Running ORIGINAL ELDORIA STORY: {len(corpus)} sentences.")

    print(f"Training PEG on {len(corpus)} sentences...")
    train_peg(model, corpus, epochs=30)

    audit_slots(model, corpus, device, word_list, top_k=10)

    # Optional: Slot evolution timeline
    # If you want to plot, uncomment the following lines:
    # import matplotlib.pyplot as plt
    # with torch.no_grad():
    #     raw_emb = []
    #     for i in range(0, len(corpus), 256):
    #         batch = corpus[i:i+256]
    #         raw_emb.append(model.encoder.encode(batch, convert_to_tensor=True, device=device))
    #     raw_emb = torch.cat(raw_emb, dim=0)
    #     z_all = model.proj_to_d(raw_emb)
    #     slot_vecs = torch.stack([s.vec for s in model.active_slots]).to(device)
    #     sims = F.cosine_similarity(z_all.unsqueeze(1), slot_vecs.unsqueeze(0), dim=-1)
    #     assigned = sims.argmax(dim=-1)
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(assigned.cpu().numpy(), marker='o', linestyle='-', markersize=4)
    #     plt.xlabel("Sentence Index")
    #     plt.ylabel("Slot Index")
    #     plt.title("Slot Evolution over the Story")
    #     plt.grid(alpha=0.3)
    #     plt.show()


# ============================================================================
# ENERGY ABLATION EXPERIMENT
# ============================================================================
def run_ablation(corpus, corpus_name, ablate):
    print(f"\n{'='*60}")
    print(f"Running ablation on: {corpus_name} | ablate_energy={ablate}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    config.ablate_energy = ablate   # set the flag
    model = PEGModel(config, device=str(device))
    word_list = load_word_ontology(model, word_list_size=50000)

    train_peg(model, corpus, epochs=30)
    audit_slots(model, corpus, device, word_list, top_k=10)
    return model


# Define the two corpora (use the original Eldoria and the pronoun version)
original_corpus = story_corpus   # already defined
pronoun_corpus = [
    # (paste the pronoun corpus from the previous code, or use a variable if you defined it)
]
# If you haven't defined pronoun_corpus, you can reuse the list from earlier.


# Run ablations
print("\n--- ABLATION: Original Eldoria with energy system ON ---")
model_orig_on = run_ablation(original_corpus, "Original Eldoria", ablate=False)

print("\n--- ABLATION: Original Eldoria with energy system OFF ---")
model_orig_off = run_ablation(original_corpus, "Original Eldoria", ablate=True)

print("\n--- ABLATION: Pronoun Corpus with energy system ON ---")
model_pron_on = run_ablation(pronoun_corpus, "Pronoun Corpus", ablate=False)

print("\n--- ABLATION: Pronoun Corpus with energy system OFF ---")
model_pron_off = run_ablation(pronoun_corpus, "Pronoun Corpus", ablate=True)