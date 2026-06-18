# PEG — Predictive Energy Grounding

**Unsupervised, memory-based AI that learns on the fly.**

PEG is a lightweight, biologically-inspired architecture that builds a dynamic memory of concepts (slots) from raw text. Instead of processing every token against every other token (\(O(N^2)\)), it maintains a small pool of energy-based slots that live, die, and evolve based on *predictive surprise*.

## The Big Idea

- **Predictive**: The system predicts the next sentence embedding. If the prediction is wrong (high "surprise"), it pays attention.
- **Energy**: Every memory slot has an energy level. High-energy slots survive; low-energy slots decay and are archived.
- **Grounding**: Slots are anchored to real-world concepts using a frozen word-embedding ontology, making them interpretable.

We started with a philosophical question: *"Do we really need to pay attention to every word, or just the entities?"* PEG is our answer.

## Current Status (Path 1 — Foundation) ✅

We have successfully built and validated a **single-domain** version of PEG.
- **Corpus**: A 60-sentence adventure story (Eldoria).
- **Result**: PEG discovered meaningful slots (`'raft'`, `'leo'`, `'mia'`, `'crystal'`). A lightweight Transformer decoder generates coherent sentences from these slots.
- **Impact**: Proves that unsupervised, on-device memory learning is possible without billions of parameters.

**Total trainable parameters:** ~1.5M (runs on a CPU).

## The Next Challenge: Multi-Domain Adaptability

Humans don't grow bigger brains when they learn new things—they build new *neural pathways*. 

We want PEG to handle **multiple domains** (e.g., Stories and Philosophy) without:
- Catastrophic forgetting.
- Exploding the parameter count.
- Requiring a supercomputer.

We are testing **three different pathways** to achieve this:

### Path 2 — Domain Token (The Efficient Hypothesis)
Add a tiny, learnable **Domain Embedding** to the decoder's input. One decoder, conditioned on the context.
- **Pros**: Minimal parameter increase (~2k).
- **The Dream**: Mimics human brain efficiency—new pathway, not a bigger brain.

### Path 3 — Separate Decoders (The Ensemble)
Maintain one shared PEG slot pool, but train **domain-specific decoders** (Decoder A for Stories, Decoder B for Philosophy).
- **Pros**: Zero interference between domains. Highest potential quality.
- **Cons**: Slightly more memory (still < 50 MB total).

### Path 4 — Dynamic Switching (The Autonomous System)
Use PEG's grounding ontology to **automatically detect** whether a slot is concrete (e.g., 'raft') or abstract (e.g., 'truth'), and route it to the appropriate decoder head at inference time.
- **Pros**: Zero user input required for domain selection.

## Project Goals

1.  **Find the most parameter-efficient way** to extend PEG to multiple domains.
2.  **Keep AI accessible**: The entire system must run locally on a laptop or phone.
3.  **Challenge the status quo**: Prove that you don't need 7-billion-parameter models to build a general-purpose intelligence system.

## Repository Structure

- `peg_core.py` — The main PEG model (slot learning and prediction).
- `decoder.py` — Transformer decoder with cross-attention.
- `train.py` — Scripts for training and evaluating the four pathways.
- `data/` — Sample datasets (Eldoria stories, philosophy aphorisms).

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/peg-ai.git
cd peg-ai

# Install dependencies
pip install torch sentence-transformers transformers nltk

# Run the baseline (Path 1)
python train.py --mode baseline

# Test multi-domain pathways
python train.py --mode domain_token    # Path 2
python train.py --mode separate_decoders  # Path 3
python train.py --mode dynamic_switch  # Path 4