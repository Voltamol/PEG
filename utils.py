# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import nltk
from nltk.corpus import words as nltk_words
import math
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# Positional Encoding (used by decoders)
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------
class SlotTextDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (slot_vec, tokens) or (slot_vec, tokens, domain_id)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------------------------------------------------
# Ontology loading
# ----------------------------------------------------------------------
def load_word_ontology(model, word_list_size=50000, batch_size=256):
    """Build ontology matrix from nltk words using the model's encoder."""
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


# ----------------------------------------------------------------------
# Nearest word utility
# ----------------------------------------------------------------------
def get_nearest_word(slot_vec, model, word_list):
    """Return the nearest ontology word for a given slot vector."""
    with torch.no_grad():
        slot_norm = F.normalize(slot_vec.unsqueeze(0), dim=-1)
        sims = slot_norm @ F.normalize(model.ontology, dim=-1).T
        idx = sims.argmax().item()
        return word_list[idx]


# ----------------------------------------------------------------------
# Build dataset (slot → tokens) with optional domain ID
# ----------------------------------------------------------------------
def build_slot_dataset(model, corpus, tokenizer, device, domain_id=None):
    """
    For each sentence, find the closest slot and store (slot_vec, tokens, domain_id).
    If domain_id is not None, it's added as a third element.
    """
    dataset = []
    for sent in corpus:
        with torch.no_grad():
            z = model.proj_to_d(torch.tensor(model.encoder.encode([sent]), device=device)).squeeze(0)
            if not model.active_slots:
                continue
            slot_vecs = torch.stack([s.vec for s in model.active_slots]).to(device)
            sims = F.cosine_similarity(z.unsqueeze(0), slot_vecs, dim=1)
            best_idx = sims.argmax().item()
            slot_vec = model.active_slots[best_idx].vec
        tokens = tokenizer.encode(sent, add_special_tokens=True)
        if domain_id is not None:
            dataset.append((slot_vec.cpu(), tokens, domain_id))
        else:
            dataset.append((slot_vec.cpu(), tokens))
    return dataset


# ----------------------------------------------------------------------
# Dataloader builder with flexible collation
# ----------------------------------------------------------------------
def get_dataloaders(dataset, batch_size=16, val_split=0.2, include_domain=False):
    split = int((1 - val_split) * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    def collate_fn(batch):
        if include_domain:
            slot_vecs = torch.stack([b[0] for b in batch])
            tokens = [b[1] for b in batch]
            domains = torch.tensor([b[2] for b in batch], dtype=torch.long)
            tokens_pad = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
            return slot_vecs, tokens_pad, domains
        else:
            slot_vecs = torch.stack([b[0] for b in batch])
            tokens_pad = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
            return slot_vecs, tokens_pad

    train_loader = DataLoader(SlotTextDataset(train_data), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SlotTextDataset(val_data), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Philosophy corpus (60 abstract sentences)
# ----------------------------------------------------------------------
PHILOSOPHY_CORPUS = [
    "The essence of existence precedes the nature of being.",
    "Causality is the necessary connection of phenomena.",
    "Truth is the correspondence between thought and reality.",
    "The categorical imperative commands us to act universally.",
    "Knowledge is justified true belief.",
    "Reality is fundamentally composed of substances.",
    "The self is the subject of all experience.",
    "Freedom is the condition for moral responsibility.",
    "Beauty is the object of pure aesthetic judgment.",
    "The sublime overwhelms our cognitive faculties.",
    "Time is the form of inner sense.",
    "Space is the form of outer sense.",
    "The world is my representation.",
    "The will to power is the fundamental drive of life.",
    "Existentialism emphasizes individual freedom and choice.",
    "The absurd arises from the conflict between reason and the world.",
    "Authenticity requires living in accordance with one's own values.",
    "The other is the condition for self-consciousness.",
    "Language is the house of being.",
    "The unconscious governs much of human behavior.",
    "The death of God signifies the end of metaphysical foundations.",
    "Nihilism denies any objective meaning.",
    "The eternal return compels us to affirm life.",
    "The overman is the goal of human evolution.",
    "The herd morality is a slave morality.",
    "The will to truth is a will to power.",
    "The genealogy of morals reveals the origin of values.",
    "The prison of language limits our thought.",
    "The text is a field of multiple interpretations.",
    "The author is dead; the reader is born.",
    "The world is a will to power and nothing besides.",
    "The God of the philosophers is not the God of faith.",
    "The soul is the form of the body.",
    "The intellect is the active principle of the mind.",
    "The senses provide the raw data for knowledge.",
    "The imagination synthesizes the manifold of intuition.",
    "The categories are the a priori conditions of understanding.",
    "The transcendental ego is the ground of all experience.",
    "The thing-in-itself is unknowable.",
    "The phenomenal world is the world of appearances.",
    "The noumenal world is the world of things as they are.",
    "The moral law is given by reason.",
    "The good will is the only thing good without qualification.",
    "The kingdom of ends is the ideal of moral community.",
    "The autonomy of the will is the basis of morality.",
    "The heteronomy of the will is the source of immorality.",
    "The sublime is beyond the beautiful.",
    "The aesthetic judgment is subjective yet universal.",
    "The teleological judgment presupposes purpose.",
    "The dialectic reveals the contradictions of pure reason.",
    "The antinomies of reason arise from transcendental illusion.",
    "The critique of pure reason limits knowledge to experience.",
    "The critique of practical reason establishes moral freedom.",
    "The critique of judgment bridges the two previous critiques.",
    "The enlightenment is the emergence from self-incurred immaturity.",
    "The public use of reason is essential for progress.",
    "The private use of reason is subordinate to the public.",
    "The perpetual peace is the ideal of international relations.",
    "The cosmopolitanism is the moral ideal of world citizenship.",
    "The human being is the end in itself."
]