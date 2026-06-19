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
    theta_high: float = 0.35       # higher = fewer spawns
    theta_novel: float = 0.5       # higher = fewer spawns
    theta_arch: float = 0.15       # higher = more aggressive pruning
    lambda_pred: float = 1.0
    ontology_size: int = 50000
    ablate_energy: bool = False    # if True, disables decay, inhibition, merge, archive

# ============================================================================
# 3. PEG MODEL (same as before, with _update_energy that checks ablate_energy)
# ============================================================================
# ... (copy the entire PEGModel class from your current file, it's already correct)
# I'll skip the full class here for brevity, but it's identical to what you have.
# The key is that _update_energy respects config.ablate_energy.

# ============================================================================
# 4. ONTOLOGY LOADING, TRAINING, AUDIT (same as before)
# ============================================================================
# ... (copy the load_word_ontology, train_peg, audit_slots functions)
# They are unchanged.

# ============================================================================
# 5. DEMO & ABLATION EXPERIMENTS
# ============================================================================
def run_experiment(corpus, corpus_name, ablate_energy=False, epochs=30):
    """Train PEG on a given corpus with optional energy ablation."""
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

if __name__ == "__main__":
    # ---------- Choose which corpus to run in the demo ----------
    RUN_PRONOM_TEST = True   # Set to False to run the original Eldoria story

    if RUN_PRONOM_TEST:
        demo_corpus = pronoun_corpus
        demo_name = "Pronoun Corpus"
    else:
        demo_corpus = original_corpus
        demo_name = "Original Eldoria"

    print(f"\n--- DEMO: Training on {demo_name} ---")
    model_demo, word_list_demo = run_experiment(demo_corpus, demo_name, ablate_energy=False, epochs=30)

    # ---------- Run the four ablation experiments ----------
    print("\n\n" + "="*80)
    print("STARTING ABLATION EXPERIMENTS")
    print("="*80)

    # 1. Original Eldoria, energy ON (already done above? We'll re-run for consistency)
    run_experiment(original_corpus, "Original Eldoria", ablate_energy=False, epochs=30)

    # 2. Original Eldoria, energy OFF
    run_experiment(original_corpus, "Original Eldoria", ablate_energy=True, epochs=30)

    # 3. Pronoun Corpus, energy ON
    run_experiment(pronoun_corpus, "Pronoun Corpus", ablate_energy=False, epochs=30)

    # 4. Pronoun Corpus, energy OFF
    run_experiment(pronoun_corpus, "Pronoun Corpus", ablate_energy=True, epochs=30)

    print("\nAll experiments complete.")