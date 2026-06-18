# path3_dynamic_switch/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from PEG_base import PEGConfig, PEGModel, train_peg, train_decoder
from utils import load_word_ontology, get_nearest_word, PHILOSOPHY_CORPUS
from PEG_base import TransformerSlotDecoder
from story_corpus import story_corpus

# We'll reuse the separate decoders training logic (Path 2) but then add switching logic.
# So we can import from path2_separate_decoders.train_separate_decoders? To keep it simple, we replicate.

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PEGConfig()
    model = PEGModel(config, device=str(device))

    word_list = load_word_ontology(model, word_list_size=50000)

    philo_corpus = PHILOSOPHY_CORPUS

    # Train PEG on combined corpus
    print("Training PEG on combined corpus...")
    train_peg(model, story_corpus + philo_corpus, epochs=30)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.pad_token = tokenizer.eos_token

    # Build datasets separately (as in Path 2)
    from path2_separate_decoders.train import train_separate_decoders
    decoder_story, decoder_philo = train_separate_decoders(
        model, story_corpus, philo_corpus, tokenizer, device, epochs=30
    )

    # Define a set of abstract words (to switch to philosophy decoder)
    abstract_words = {'truth', 'essence', 'causality', 'metaphysics', 'existence',
                      'reason', 'being', 'knowledge', 'reality', 'freedom',
                      'beauty', 'sublime', 'time', 'space', 'world',
                      'will', 'power', 'existentialism', 'authenticity',
                      'consciousness', 'language', 'unconscious', 'nihilism',
                      'overman', 'morality', 'genealogy', 'interpretation',
                      'author', 'reader', 'soul', 'intellect', 'senses',
                      'imagination', 'categories', 'transcendental', 'phenomenal',
                      'noumenal', 'moral', 'autonomy', 'heteronomy', 'aesthetic',
                      'teleological', 'dialectic', 'antinomy', 'enlightenment',
                      'cosmopolitanism'}

    # Test generation with dynamic switching
    raft_slot = None
    for slot in model.active_slots:
        label = get_nearest_word(slot.vec, model, word_list)
        if label == 'raft':
            raft_slot = slot
            break

    if raft_slot:
        decoder_story.eval()
        decoder_philo.eval()
        with torch.no_grad():
            slot_vec = raft_slot.vec.unsqueeze(0).to(device)
            label = get_nearest_word(raft_slot.vec, model, word_list)
            if label in abstract_words:
                tokens = decoder_philo(slot_vec, target_tokens=None, max_len=30)
                print(f"Slot label '{label}' is abstract → using philosophy decoder.")
            else:
                tokens = decoder_story(slot_vec, target_tokens=None, max_len=30)
                print(f"Slot label '{label}' is concrete → using story decoder.")
            print("Generated:")
            print(tokenizer.decode(tokens[0].cpu().numpy(), skip_special_tokens=True))
    else:
        print("No raft slot found.")