# path1_domain_token/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from PEG_base import PEGConfig, PEGModel, train_peg
from utils import load_word_ontology, build_slot_dataset, get_dataloaders, PHILOSOPHY_CORPUS, get_nearest_word
from story_corpus import story_corpus

# Import the domain token decoder (we'll define it here or in a separate file)
# For simplicity, we'll include the class definition in this script.

class PositionalEncoding(nn.Module):
    # same as utils – we'll import from utils, but to avoid circular import, we define it here.
    # Actually we can import from utils.
    pass

# We'll just import from utils
from utils import PositionalEncoding

class DomainTokenDecoder(nn.Module):
    def __init__(self, slot_dim=256, d_model=256, nhead=4, num_layers=2,
                 dim_feedforward=512, vocab_size=30522, max_len=50, num_domains=2):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_domains = num_domains
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.domain_emb = nn.Embedding(num_domains, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, slot_vec, domain_id, target_tokens=None, teacher_forcing_ratio=0.5, max_len=None):
        batch_size = slot_vec.size(0)
        max_len = max_len or self.max_len
        slot_emb = self.slot_proj(slot_vec).unsqueeze(1)      # (batch, 1, d)
        dom_emb = self.domain_emb(domain_id).unsqueeze(1)      # (batch, 1, d)
        memory = torch.cat([slot_emb, dom_emb], dim=1)         # (batch, 2, d)

        if target_tokens is not None:
            tgt_input = target_tokens[:, :-1]
            tgt_emb = self.pos_encoder(self.embedding(tgt_input))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1), device=slot_vec.device)
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   target_tokens[:, 1:].reshape(-1),
                                   ignore_index=0)
            return loss
        else:
            generated = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=slot_vec.device)
            for _ in range(max_len - 1):
                tgt_emb = self.pos_encoder(self.embedding(generated))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=slot_vec.device)
                output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(output[:, -1, :])
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == tokenizer.sep_token_id).all():
                    break
            return generated


def train_domain_token_decoder(decoder, train_loader, val_loader, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    for epoch in range(epochs):
        decoder.train()
        total_loss = 0
        for slot_vecs, target_tokens, domain_ids in train_loader:
            slot_vecs = slot_vecs.to(decoder.fc_out.weight.device)
            target_tokens = target_tokens.to(decoder.fc_out.weight.device)
            domain_ids = domain_ids.to(decoder.fc_out.weight.device)
            loss = decoder(slot_vecs, domain_ids, target_tokens, teacher_forcing_ratio=0.5)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for slot_vecs, target_tokens, domain_ids in val_loader:
                slot_vecs = slot_vecs.to(decoder.fc_out.weight.device)
                target_tokens = target_tokens.to(decoder.fc_out.weight.device)
                domain_ids = domain_ids.to(decoder.fc_out.weight.device)
                loss = decoder(slot_vecs, domain_ids, target_tokens, teacher_forcing_ratio=0.0)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        print(f"DomainToken Epoch {epoch:02d} | train loss={avg_train:.4f} | val loss={avg_val:.4f}")
    print("DomainToken decoder training complete.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    model = PEGModel(config, device=str(device))

    # Load ontology
    word_list = load_word_ontology(model, word_list_size=50000)

    # Two corpora
    philo_corpus = PHILOSOPHY_CORPUS

    # Combine with domain labels: 0 = story, 1 = philosophy
    combined_corpus = story_corpus + philo_corpus
    domain_labels = [0]*len(story_corpus) + [1]*len(philo_corpus)

    # Train PEG on the combined corpus (so slots learn both domains)
    print("Training PEG on combined corpus...")
    train_peg(model, combined_corpus, epochs=30)

    # Build dataset with domain IDs
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.pad_token = tokenizer.eos_token

    dataset_with_domain = []
    for sent, d_id in zip(combined_corpus, domain_labels):
        with torch.no_grad():
            z = model.proj_to_d(torch.tensor(model.encoder.encode([sent]), device=device)).squeeze(0)
            if not model.active_slots:
                continue
            slot_vecs = torch.stack([s.vec for s in model.active_slots]).to(device)
            sims = F.cosine_similarity(z.unsqueeze(0), slot_vecs, dim=1)
            best_idx = sims.argmax().item()
            slot_vec = model.active_slots[best_idx].vec
        tokens = tokenizer.encode(sent, add_special_tokens=True)
        dataset_with_domain.append((slot_vec.cpu(), tokens, d_id))

    train_loader, val_loader = get_dataloaders(dataset_with_domain, batch_size=16, include_domain=True)

    # Train decoder
    decoder = DomainTokenDecoder(
        slot_dim=config.d,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        vocab_size=tokenizer.vocab_size,
        max_len=50,
        num_domains=2
    ).to(device)

    train_domain_token_decoder(decoder, train_loader, val_loader, epochs=30)

    # Test generation from a slot whose label contains 'raft'
    raft_slot = None
    for slot in model.active_slots:
        label = get_nearest_word(slot.vec, model, word_list)
        if 'raft' in label.lower():
            raft_slot = slot
            print(f"Found slot labelled '{label}' (contains 'raft')")
            break

    if raft_slot is None:
        raft_slot = model.active_slots[0] if model.active_slots else None
        if raft_slot:
            print(f"No 'raft' slot found, using first slot with label '{get_nearest_word(raft_slot.vec, model, word_list)}'")

    if raft_slot:
        decoder.eval()
        with torch.no_grad():
            # Generate with story domain (0) and philosophy domain (1) to compare
            for dom, dom_name in [(0, "story"), (1, "philosophy")]:
                tokens = decoder(raft_slot.vec.unsqueeze(0).to(device),
                                 torch.tensor([dom], device=device),
                                 target_tokens=None, max_len=30)
                print(f"Generated from 'raft' slot with domain '{dom_name}':")
                print(tokenizer.decode(tokens[0].cpu().numpy(), skip_special_tokens=True))
    else:
        print("No raft slot found.")