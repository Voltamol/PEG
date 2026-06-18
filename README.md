# PEG — Predictive Energy Grounding

**Unsupervised, memory‑based AI that learns on the fly.**

PEG is a lightweight, biologically‑inspired architecture that builds a dynamic memory of concepts (slots) from raw text. Instead of processing every token against every other token (`O(N²)`), it maintains a small pool of energy‑based slots that live, die, and evolve based on *predictive surprise*.

## The Big Idea

- **Predictive**: The system predicts the next sentence embedding. If the prediction is wrong (high "surprise"), it pays attention.
- **Energy**: Every memory slot has an energy level. High‑energy slots survive; low‑energy slots decay and are archived.
- **Grounding**: Slots are anchored to real‑world concepts using a frozen word‑embedding ontology, making them interpretable.

We started with a question: *"Do we really need to pay attention to every word, or just the entities?"* PEG is our answer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PEG_base.py                           │
│  (Foundation – Single‑domain slot learning & generation)   │
│                                                             │
│  • Predictor (P)  • Energy (E)  • Grounding (G)           │
│  • Transformer decoder with cross‑attention                │
│  • ~1.5M trainable parameters                              │
└─────────────────────────────────────────────────────────────┘
							  │
		  ┌───────────────────┼───────────────────┐
		  ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Path 1        │ │   Path 2        │ │   Path 3        │
│  Domain Token   │ │ Separate Decoders│ │Dynamic Switching│
│                 │ │                 │ │                 │
│ • Learnable     │ │ • One decoder   │ │ • Auto‑detects  │
│   domain emb    │ │   per domain    │ │   domain from   │
│ • One decoder   │ │ • No task       │ │   ontology      │
│ • Minimal       │ │   interference   │ │ • Zero user     │
│   parameter     │ │   • Highest     │ │   input         │
│   increase      │ │   quality       │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## The Three Pathways

### Path 1 — Domain Token (The Efficient Hypothesis)
Add a tiny, learnable **Domain Embedding** to the decoder's memory sequence. One decoder handles both domains, conditioned on a context token.

- **Pros**: Minimal parameter increase (~2k).
- **The Dream**: Mimics human brain efficiency—new pathway, not a bigger brain.
- **Folder**: `path1_domain_token/`

### Path 2 — Separate Decoders (The Ensemble)
Maintain one shared PEG slot pool, but train **domain‑specific decoders** (Decoder A for Stories, Decoder B for Philosophy).

- **Pros**: Zero interference between domains. Highest potential quality.
- **Cons**: Slightly more memory (still < 50 MB total).
- **Folder**: `path2_separate_decoders/`

### Path 3 — Dynamic Switching (The Autonomous System)
Use PEG's grounding ontology to **automatically detect** whether a slot is concrete (e.g., `'raft'`) or abstract (e.g., `'truth'`), and route it to the appropriate decoder head at inference time.

- **Pros**: Zero user input required for domain selection.
- **Folder**: `path3_dynamic_switch/`

---

## Current Status (Foundation) ✅

The `PEG_base.py` has been successfully validated on a single domain (Eldoria story).
- **Result**: PEG discovered meaningful slots (`'raft'`, `'leo'`, `'mia'`, `'crystal'`). A lightweight Transformer decoder generates coherent sentences from these slots.
- **Impact**: Proves that unsupervised, on‑device memory learning is possible without billions of parameters.

**Total trainable parameters:** ~1.5M (runs on a CPU).

---

## The Bigger Question

*"How do we scale PEG to multiple domains without growing the brain?"*

These three pathways are our systematic attempt to answer that. We'll compare them on:
- **Per‑domain validation loss**
- **Cross‑domain perplexity** (does it produce nonsense when given the wrong context?)
- **Slot stability** (does abstract text corrupt concrete slots?)

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/peg-ai.git
cd peg-ai

# Install dependencies
pip install torch sentence-transformers transformers nltk

# Run the foundation (PEG_base.py)
python PEG_base.py

# Run each pathway
python path1_domain_token/train.py
python path2_separate_decoders/train.py
python path3_dynamic_switch/train.py
```

---

## Repository Structure

```
PEG/
├── PEG_base.py              # Foundation – single‑domain PEG
├── utils.py                 # Shared utilities
├── README.md
├── path1_domain_token/      # Domain Token (conditional decoder)
│   └── train.py
├── path2_separate_decoders/ # Separate decoders per domain
│   └── train.py
└── path3_dynamic_switch/    # Ontology‑guided auto‑switching
	└── train.py
```

---

## The Philosophy

> *"The best way to predict the future is to invent it."* — Alan Kay

This project is proof that one person with a "crazy idea" can build something that works—without a lab, without a budget, and without a billion‑parameter model.

**AI should be accessible. PEG is our contribution to that future.**

---

## License

MIT — Open, accessible, and free for everyone.

*Hats off to the late-night debug sessions, the `IndexError` hunts, and the willingness to tolerate "crazy" ideas.* 🍻
