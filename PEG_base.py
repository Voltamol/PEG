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
# 1. CORPORA
# ============================================================================
# Original Eldoria story (with proper names)
original_corpus = story_corpus
print(f"Loaded Eldoria corpus: {len(original_corpus)} sentences.")

# Pronoun-replaced corpus (names → roles)
pronoun_corpus = [
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
print(f"Loaded pronoun corpus: {len(pronoun_corpus)} sentences.")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
@dataclass
class PEGConfig:
    d: int = 256
    hidden_size: int = 512
    gamma: float = 0.995
    alpha: float = 0.1
    beta: float = 0.05
    theta_high: float = 0.35
    theta_novel: float = 0.5
    theta_arch: float = 0.15
    lambda_pred: float = 1.0
    ontology_size: int = 50000
    ablate_energy: bool = False

# ============================================================================
# 3. PEG MODEL CLASSES
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

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.embed_dim = self.encoder.get_embedding_dimension()
        self.proj_to_d = nn.Linear(self.embed_dim, config.d) if self.embed_dim != config.d else nn.Identity()

        self.register_buffer('ontology', F.normalize(torch.randn(config.ontology_size, config.d), dim=-1))

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
            # ABLATION: only boost, no decay, no inhibition, no merge, no archive
            for weights in binding_weights:
                for slot, w in zip(self.active_slots, weights):
                    if w > 0.1:
                        slot.energy += self.config.alpha * (1 - slot.energy)
                        slot.age += 1
            for slot in self.active_slots:
                slot.energy = max(0.0, min(1.0, slot.energy))
            return

        # ---- ORIGINAL ENERGY SYSTEM ----
        for slot in self.active_slots:
            slot.energy *= self.config.gamma
        for weights in binding_weights:
            for slot, w in zip(self.active_slots, weights):
                if w > 0.1:
                    slot.energy += self.config.alpha * (1 - slot.energy)
                    slot.age += 1
        for slot in self.active_slots:
            slot.energy = max(0.0, min(1.0, slot.energy))

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
        with torch.no_grad():
            slot_norm = F.normalize(slot.vec.unsqueeze(0), dim=-1)
            sim_ont = slot_norm @ F.normalize(model.ontology, dim=-1).T
            label = word_list[sim_ont.argmax().item()]

        sim_slot = F.cosine_similarity(z_all, slot.vec.unsqueeze(0), dim=-1)
        top_idx = sim_slot.topk(min(top_k, len(corpus))).indices.tolist()
        top_sents = [corpus[i] for i in top_idx]

        count = (assigned == i).sum().item()

        print(f"\n{'='*60}")
        print(f"SLOT {i:02d} | label: '{label}' | energy={slot.energy:.3f} | assigned={count}/{len(corpus)}")
        print("Top sentences:")
        for j, sent in enumerate(top_sents[:5]):
            print(f"  {j+1}. {sent[:100]}...")
        assigned_indices = (assigned == i).nonzero(as_tuple=True)[0].tolist()
        if assigned_indices:
            sample = [corpus[idx] for idx in assigned_indices[:3]]
            print("Sample assigned:")
            for sent in sample:
                print(f"  - {sent[:100]}...")

# ============================================================================
# 7. RUN EXPERIMENT
# ============================================================================
def run_experiment(corpus, corpus_name, ablate_energy=False, epochs=30):
    print(f"\n{'='*60}")
    print(f"Running experiment: {corpus_name} | ablate_energy={ablate_energy}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    config.ablate_energy = ablate_energy
    model = PEGModel(config, device=str(device))
    word_list = load_word_ontology(model, word_list_size=50000)

    train_peg(model, corpus, epochs=epochs)
    audit_slots(model, corpus, device, word_list, top_k=10)
    return model, word_list

# ============================================================================
# STRUCTURED TOPIC DISCOVERY TEST (AI / Sports / Finance / Politics)
# ============================================================================

def generate_mixed_headlines():
    """4 distinct domains, 20 headlines each, shuffled."""
    import random
    random.seed(42)  # reproducible shuffle

    ai_headlines = [
        "OpenAI releases GPT-5 with reasoning capabilities",
        "Google DeepMind develops new protein folding model",
        "Anthropic's Claude surpasses human coding benchmarks",
        "Nvidia announces new AI chip for data centers",
        "Meta open-sources Llama 4 language model",
        "AI startup raises $1 billion for autonomous agents",
        "US government proposes AI safety regulations",
        "China approves first generative AI model for public use",
        "AI detects cancer with 95% accuracy in new study",
        "Microsoft integrates Copilot into Windows kernel",
        "Apple acquires AI music generation startup",
        "EU passes landmark AI liability law",
        "Robotics company deploys AI warehouse workers",
        "OpenAI launches text-to-video generation tool",
        "AI weather forecasting beats traditional models",
        "Google introduces AI-powered search with citations",
        "AMD unveils AI accelerator chip for laptops",
        "AI generates full-length movie script in 24 hours",
        "US military tests AI for drone navigation",
        "AI language model passes medical licensing exam"
    ]

    sports_headlines = [
        "Lionel Messi scores hat-trick in Inter Miami win",
        "LeBron James signs extension with Lakers through 2027",
        "Manchester City wins Premier League title on final day",
        "Serena Williams announces retirement from tennis",
        "Super Bowl LVIII draws record viewership",
        "Real Madrid defeats Barcelona in El Clasico thriller",
        "Tiger Woods withdraws from Masters due to injury",
        "Olympic committee adds breakdancing to 2028 games",
        "NBA finals go to Game 7 for first time in decade",
        "Formula 1 driver wins dramatic rain-soaked Grand Prix",
        "College football playoff expands to 16 teams",
        "Wimbledon champion stunned in first-round upset",
        "MLB proposes pitch clock changes for 2025 season",
        "UFC heavyweight title fight ends in controversial decision",
        "Women's World Cup final breaks attendance records",
        "Boston Celtics acquire All-Star in blockbuster trade",
        "Marathon world record shattered in Berlin",
        "NHL postpones game due to severe weather",
        "Tour de France champion disqualified for doping",
        "Cricket World Cup final goes to super over"
    ]

    finance_headlines = [
        "Fed signals rate cut as inflation cools to 2.8%",
        "Goldman Sachs beats earnings estimates on trading surge",
        "Tesla stock jumps 15% on record vehicle deliveries",
        "Bitcoin crosses $70,000 amid institutional buying",
        "JPMorgan reports strong quarterly profits, lowers loan loss reserves",
        "S&P 500 hits all-time high on tech rally",
        "Oil prices drop to 6-month low on demand fears",
        "Apple faces $2 billion antitrust fine in EU",
        "US treasury yields climb as jobs data beats forecasts",
        "Visa acquires fintech startup for $5 billion",
        "Alibaba shares soar on China economic stimulus news",
        "Hedge fund returns surge after market volatility",
        "SEC approves first spot bitcoin ETFs",
        "Porsche IPO valuation falls short of expectations",
        "Warren Buffett increases stake in Apple, sells Bank of America",
        "China's economy grows 5.2% in Q4, below target",
        "FedEx announces major restructuring, cuts 10,000 jobs",
        "BlackRock launches new AI-focused fund",
        "Eurozone inflation rises unexpectedly to 2.6%",
        "UBS finalizes merger with Credit Suisse, cuts 3,000 roles"
    ]

    politics_headlines = [
        "US Congress passes bipartisan infrastructure bill",
        "Biden administration announces new climate executive order",
        "UK Parliament votes to recognize Palestine as a state",
        "French president calls for snap election after no-confidence vote",
        "German coalition collapses in dispute over nuclear phase-out",
        "China's military conducts largest-ever naval exercise",
        "India's opposition wins state elections, boosts Modi pressure",
        "Turkey blocks NATO expansion over Sweden membership dispute",
        "Brazil launches criminal probe into former president",
        "UN Security Council passes resolution on Gaza ceasefire",
        "Russia offers sanctions relief in exchange for Ukraine neutrality",
        "Japan's prime minister faces corruption allegations",
        "Mexico elects first female president in landslide victory",
        "Saudi Arabia announces major investment in renewable energy",
        "South Africa appeals to Hague court on Israel genocide case",
        "Australia commits to nuclear submarine fleet by 2040",
        "Iran and US resume nuclear talks in Vienna",
        "Canadian parliament approves carbon tax hike",
        "Nigeria's president declares state of emergency on food security",
        "Spain and Netherlands agree to boost European defense spending"
    ]

    all = ai_headlines + sports_headlines + finance_headlines + politics_headlines
    random.shuffle(all)
    return all

# ============================================================================
# RUN THE TEST
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    
    # Generate mixed headlines
    headlines = generate_mixed_headlines()
    print(f"Generated {len(headlines)} headlines.")

    # Train PEG with tuned hyperparameters (Energy ON)
    model = PEGModel(config, device=str(device))
    word_list = load_word_ontology(model, word_list_size=50000)
    
    print("Training PEG on mixed headlines...")
    train_peg(model, headlines, epochs=30)

    # Audit the slots
    audit_slots(model, headlines, device, word_list, top_k=10)

    # Optional: Check slot purity manually
    # For each slot, look at the sample assigned sentences and note which domain they belong to.
    # If Slot 0 has 19/20 AI headlines -> pure! 
    # If Slot 0 has 8 AI, 6 Finance, 6 Politics -> mixed!