#!/usr/bin/env python3
"""
analyze_log.py — Load a Graph‑PEG pickle and analyse the feature_log.

Usage:
    python analyze_log.py --input <pickle_file>

Example:
    python analyze_log.py --input alice_graph.pkl
    python analyze_log.py --input inline_graph.pkl
"""

import argparse
import pickle
import sys
from collections import Counter

import pandas as pd

# --------------------------------------------------------------------
# 1. LOAD THE PICKLE
# --------------------------------------------------------------------
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# --------------------------------------------------------------------
# 2. EXTRACT FEATURE LOG
# --------------------------------------------------------------------
def get_feature_log(data):
    feature_log = data.get('feature_log', [])
    if not feature_log:
        print("[WARNING] feature_log is empty or missing.")
    return feature_log

# --------------------------------------------------------------------
# 3. SUMMARISE
# --------------------------------------------------------------------
def summarise(df):
    print("\n" + "=" * 60)
    print("FEATURE LOG SUMMARY")
    print("=" * 60)

    # Basic counts
    print(f"\nTotal feature vectors: {len(df):,}")
    print(f"Unique verbs:          {df['verb'].nunique():,}")
    print(f"Unique roles:          {df['role'].nunique():,}")
    print(f"Unique sources:        {df['source'].nunique():,}")

    # Role distribution
    print("\n--- Role distribution ---")
    role_counts = df['role'].value_counts()
    for role, count in role_counts.items():
        print(f"  {role:12s}: {count:6,} ({count/len(df)*100:5.1f}%)")

    # Source distribution (top 10)
    print("\n--- Source distribution (top 10) ---")
    src_counts = df['source'].value_counts().head(10)
    for src, count in src_counts.items():
        print(f"  {src:20s}: {count:6,}")

    # Top verb-role pairs
    print("\n--- Top 10 verb → role pairs ---")
    vr_pairs = df.groupby(['verb', 'role']).size().sort_values(ascending=False).head(10)
    for (verb, role), count in vr_pairs.items():
        print(f"  {verb:12s} → {role:12s}: {count:6,}")

    # Filler POS per role
    print("\n--- Filler POS per role (top 2 per role) ---")
    for role in df['role'].unique():
        pos_counts = df[df['role'] == role]['filler_pos'].value_counts().head(2)
        pos_str = ", ".join([f"{p} ({c})" for p, c in pos_counts.items()])
        print(f"  {role:12s}: {pos_str}")

    # Conjuncts vs non-conjuncts
    if 'is_conjunct' in df.columns:
        print("\n--- Conjunct vs direct ---")
        conj_counts = df['is_conjunct'].value_counts()
        print(f"  Direct (False): {conj_counts.get(False, 0):,}")
        print(f"  Conjunct (True): {conj_counts.get(True, 0):,}")

    # Pronoun vs non-pronoun
    if 'is_pronoun' in df.columns:
        print("\n--- Pronoun vs non-pronoun ---")
        pron_counts = df['is_pronoun'].value_counts()
        print(f"  Non-pronoun (False): {pron_counts.get(False, 0):,}")
        print(f"  Pronoun (True):      {pron_counts.get(True, 0):,}")

    # Confidence distribution
    if 'confidence' in df.columns:
        print("\n--- Confidence statistics ---")
        print(f"  Mean:   {df['confidence'].mean():.3f}")
        print(f"  Std:    {df['confidence'].std():.3f}")
        print(f"  Min:    {df['confidence'].min():.3f}")
        print(f"  Max:    {df['confidence'].max():.3f}")
        print(f"  Median: {df['confidence'].median():.3f}")

    print("\n" + "=" * 60)

# --------------------------------------------------------------------
# 4. SAVE TO CSV
# --------------------------------------------------------------------
def save_csv(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"\nFeature log saved to: {output_file}")

# --------------------------------------------------------------------
# 5. MAIN
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyse feature_log from Graph‑PEG pickle.")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the pickle file (e.g., alice_graph.pkl)')
    parser.add_argument('--csv', '-c', type=str, default=None,
                        help='Optional: save feature log to CSV file')
    args = parser.parse_args()

    # Load
    try:
        data = load_pickle(args.input)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading pickle: {e}")
        sys.exit(1)

    # Extract log
    feature_log = get_feature_log(data)
    if not feature_log:
        print("No features to analyse.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(feature_log)

    # Summarise
    summarise(df)

    # Optional CSV export
    if args.csv:
        save_csv(df, args.csv)
    else:
        # Suggest a default CSV name
        default_csv = args.input.rsplit('.', 1)[0] + '_feature_log.csv'
        print(f"\nTo save to CSV, run with --csv {default_csv}")

if __name__ == "__main__":
    main()