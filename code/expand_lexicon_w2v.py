"""
Lexicon Expansion via Word2Vec

Expands the Gaucher et al. (2011) masculine/feminine word stems using
Word2Vec (word2vec-google-news-300) embeddings. Uses centroid expansion,
adjective-first resolution, cross-list filtering, and junk token removal.
Final additions: 3 masculine, 9 feminine terms after manual curation.

Author: Manya Reddy
Date:   2026-04-15
CS 72, Dartmouth College
"""

import gensim.downloader as api
import json
import os
import numpy as np

# ──────────────────────────────────────────────
# STEP 1: Define the Gaucher et al. (2011) word lists
# ──────────────────────────────────────────────

# Entries with * are stems (match any word starting with that prefix).
# Entries without * are exact words.
# Source: Gaucher, Friesen & Kay (2011), Table 1.

MASCULINE_ENTRIES = [
    "active", "adventurous", "aggress*", "ambition", "ambitious", "analy*",
    "assert*", "athlet*", "autonom*", "boast*", "challenge*",
    "compet*", "confident", "courag*", "decide", "decisive",
    "decision*", "determin*", "dominant", "dominance", "force*",
    "greedy", "headstrong", "hierarch*", "hostil*", "impulsive",
    "independent", "independence", "individual*", "intellect*", "lead*", "logic",
    "masculine", "objective", "opinion", "outspoken", "persist",
    "principle*", "reckless", "stubborn", "superior",
    "self-confident", "self-confidence", "self-sufficience", "self-sufficient", "self-reliance", "self-reliant",
]

FEMININE_ENTRIES = [
    "affectionate", "child*", "cheer*", "commit*", "communal",
    "compassion*", "connect*", "considerate", "cooperat*", "depend*",
    "emotional", "empath*", "feminine", "flatterable", "gentle",
    "honest", "interpersonal", "interdependence", "interdependent", "interpersonal",
    "kind", "kinship", "loyal*", "modesty", "nag", "nurtur*",
    "pleasant*", "polite", "quiet*", "respon*", "sensitiv*",
    "submissive", "support*", "sympath*", "tender*", "together*",
    "trust*", "understand*", "warm*", "whin*", "yield*",
]

# ──────────────────────────────────────────────
# STEP 2: Load the pretrained Word2Vec model
# ──────────────────────────────────────────────
print("Loading Word2Vec Google News model (this may take a minute)...")
model = api.load("word2vec-google-news-300")
print(f"Model loaded. Vocabulary size: {len(model.key_to_index):,} words\n")

# ──────────────────────────────────────────────
# STEP 3: Configuration
# ──────────────────────────────────────────────
THRESHOLDS = [0.75, 0.70, 0.60]
TOP_N = 30
CENTROID_TOP_N = 50

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────
# STEP 4: Filtering functions
# ──────────────────────────────────────────────
def parse_entries(entries):
    """Split entries into stem prefixes (those with *) and exact words."""
    stems = []
    exact = []
    for entry in entries:
        if entry.endswith("*"):
            stems.append(entry[:-1].lower())  # strip * and lowercase
        else:
            exact.append(entry.lower())
    return stems, exact


def is_junk_token(word):
    """Filter out proper nouns, multi-word tokens, and tokens with digits."""
    if "_" in word:
        return True
    if word[0].isupper():
        return True
    if any(c.isdigit() for c in word):
        return True
    return False


def is_already_covered(word, stems, exact_words):
    """Check if a word is already covered by the Gaucher entries.

    A word is covered if:
      - It exactly matches an entry (no asterisk), OR
      - It starts with any stem prefix (entry with asterisk)
    """
    w = word.lower()
    if w in exact_words:
        return True
    for stem in stems:
        if w.startswith(stem):
            return True
    return False


# ──────────────────────────────────────────────
# STEP 5: Per-word expansion
# ──────────────────────────────────────────────
def expand_words(entries, label):
    """Query Word2Vec for similar words, filter, and print results."""
    stems, exact_words = parse_entries(entries)

    # Build a list of query words to look up in the model.
    # For stems, we need a representative full word — use the stem itself
    # if it's in the vocabulary, otherwise try common suffixes.
    query_words = []
    for entry in entries:
        raw = entry.rstrip("*").lower()
        # Try the raw form first, then common extensions
        # Prioritize adjective forms, then other parts of speech
        candidates = [
            raw + "ive",    # assertive, competitive, aggressive
            raw + "ous",    # adventurous, courageous, ambitious
            raw + "al",     # emotional, communal, analytical
            raw + "ful",    # cheerful, boastful
            raw + "ant",    # dominant, pleasant
            raw + "ent",    # confident, independent
            raw + "able",   # dependable, flatterable
            raw + "ible",   # responsible
            raw + "ic",     # athletic
            raw,            # raw stem / exact word
            raw + "e",      # base verb: dominate, force
            raw + "ing",    # gerund: nurturing, understanding
            raw + "ion",    # noun: decision, compassion
            raw + "ity",    # noun: hostility
            raw + "ness",   # noun: kindness
        ]
        found = False
        for c in candidates:
            if c in model.key_to_index:         # checking if word c is in the model's vocab dictionary 
                query_words.append((entry, c))
                found = True
                break
        if not found:
            query_words.append((entry, None))

    results = {t: {} for t in THRESHOLDS}
    skipped = []

    for entry, query in query_words:
        if query is None:
            skipped.append(entry)
            continue

        neighbors = model.most_similar(query, topn=TOP_N)

        for t in THRESHOLDS:
            hits = []
            for w, score in neighbors:
                if score < t:
                    continue
                if is_junk_token(w):
                    continue
                if is_already_covered(w, stems, exact_words):
                    continue
                hits.append((w, round(score, 3)))
            if hits:
                results[t][entry] = hits

    # ── Print results ─────────────────────────
    print(f"{'=' * 60}")
    print(f"  {label} — Per-Word Expansion Results")
    print(f"{'=' * 60}")

    if skipped:
        print(f"\n  Skipped (not in model vocabulary): {', '.join(skipped)}")

    for t in THRESHOLDS:
        print(f"\n  --- Threshold: {t} ---")
        if not results[t]:
            print("  No expansions found at this threshold.")
        for entry, neighbors in sorted(results[t].items()):
            neighbor_str = ", ".join(f"{w} ({s})" for w, s in neighbors)
            print(f"  {entry:20s} -> {neighbor_str}")

    # Collect unique new words per threshold
    expanded = {}
    for t in THRESHOLDS:
        words_at_t = set()
        for neighbors in results[t].values():
            for w, _ in neighbors:
                words_at_t.add(w.lower())
        expanded[t] = words_at_t

    print(f"\n  Summary:")
    for t in THRESHOLDS:
        print(f"    New words at {t}+: {len(expanded[t])}")
    print()

    return expanded


# ──────────────────────────────────────────────
# STEP 6: Centroid-based expansion
# ──────────────────────────────────────────────
def centroid_expand(entries, label):
    """Compute the average vector of the entry list and find nearest neighbors."""
    stems, exact_words = parse_entries(entries)

    # Build centroid vector — resolve each entry to a model word
    vectors = []
    used = []
    skipped = []
    for entry in entries:
        raw = entry.rstrip("*").lower()
        # Prioritize adjective forms, then other parts of speech
        candidates = [
            raw + "ive",    # assertive, competitive, aggressive
            raw + "ous",    # adventurous, courageous, ambitious
            raw + "al",     # emotional, communal, analytical
            raw + "ful",    # cheerful, boastful
            raw + "ant",    # dominant, pleasant
            raw + "ent",    # confident, independent
            raw + "able",   # dependable, flatterable
            raw + "ible",   # responsible
            raw + "ic",     # athletic
            raw,            # raw stem / exact word
            raw + "e",      # base verb: dominate, force
            raw + "ing",    # gerund: nurturing, understanding
            raw + "ion",    # noun: decision, compassion
            raw + "ity",    # noun: hostility
            raw + "ness",   # noun: kindness
        ]
        found = False
        for c in candidates:
            if c in model.key_to_index:
                vectors.append(model[c])
                used.append(f"{entry} -> {c}")
                found = True
                break
        if not found:
            skipped.append(entry)

    centroid = np.mean(vectors, axis=0)

    # Query nearest neighbors to the centroid
    neighbors = model.similar_by_vector(centroid, topn=CENTROID_TOP_N)

    print(f"{'=' * 60}")
    print(f"  {label} — Centroid Expansion Results")
    print(f"{'=' * 60}")
    print(f"  Centroid built from {len(used)}/{len(entries)} entries")
    if skipped:
        print(f"  Skipped (not in model vocabulary): {', '.join(skipped)}")

    # Filter and collect at each threshold
    expanded = {}
    for t in THRESHOLDS:
        hits = []
        for w, score in neighbors:
            if score < t:
                continue
            if is_junk_token(w):
                continue
            if is_already_covered(w, stems, exact_words):
                continue
            hits.append((w, round(score, 3)))

        expanded[t] = {w.lower() for w, _ in hits}

        print(f"\n  --- Threshold: {t} ---")
        if not hits:
            print("  No expansions found at this threshold.")
        for w, s in hits:
            print(f"    {w:25s} ({s})")

    print(f"\n  Summary:")
    for t in THRESHOLDS:
        print(f"    New words at {t}+: {len(expanded[t])}")
    print()

    return expanded


# ──────────────────────────────────────────────
# STEP 7: Run both expansion methods
# ──────────────────────────────────────────────
print("=" * 60)
print("  PART A: PER-WORD EXPANSION")
print("=" * 60)
print()

print("Expanding masculine entries...")
masc_perword = expand_words(MASCULINE_ENTRIES, "MASCULINE")

print("Expanding feminine entries...")
fem_perword = expand_words(FEMININE_ENTRIES, "FEMININE")

print()
print("=" * 60)
print("  PART B: CENTROID EXPANSION")
print("=" * 60)
print()

print("Computing masculine centroid...")
masc_centroid = centroid_expand(MASCULINE_ENTRIES, "MASCULINE")

print("Computing feminine centroid...")
fem_centroid = centroid_expand(FEMININE_ENTRIES, "FEMININE")

# ──────────────────────────────────────────────
# STEP 8: Cross-list filter
# ──────────────────────────────────────────────
# Remove words that appear in both masculine and feminine expansions
print()
print("=" * 60)
print("  CROSS-LIST FILTERING")
print("=" * 60)

for t in THRESHOLDS:
    masc_all = masc_perword[t] | masc_centroid[t]
    fem_all = fem_perword[t] | fem_centroid[t]
    overlap = masc_all & fem_all

    if overlap:
        print(f"\n  Threshold {t} — removing {len(overlap)} overlapping words: {', '.join(sorted(overlap))}")
        masc_perword[t] -= overlap
        masc_centroid[t] -= overlap
        fem_perword[t] -= overlap
        fem_centroid[t] -= overlap
    else:
        print(f"\n  Threshold {t} — no overlap found")

# ──────────────────────────────────────────────
# STEP 9: Print final summary
# ──────────────────────────────────────────────
print()
print(f"{'=' * 60}")
print(f"  FINAL SUMMARY (after all filtering)")
print(f"{'=' * 60}")
print(f"  Original masculine entries:  {len(MASCULINE_ENTRIES)}")
print(f"  Original feminine entries:   {len(FEMININE_ENTRIES)}")
print()
for t in THRESHOLDS:
    masc_total = masc_perword[t] | masc_centroid[t]
    fem_total = fem_perword[t] | fem_centroid[t]
    print(f"  At {t} threshold:")
    print(f"    Masculine — per-word: {len(masc_perword[t])}, centroid: {len(masc_centroid[t])}, combined: {len(masc_total)}")
    print(f"    Feminine  — per-word: {len(fem_perword[t])}, centroid: {len(fem_centroid[t])}, combined: {len(fem_total)}")
    print()

# ──────────────────────────────────────────────
# STEP 10: Save expanded lexicons to JSON
# ──────────────────────────────────────────────
output = {
    "thresholds": THRESHOLDS,
    "filters_applied": [
        "prefix_dedup (skip candidates matching any Gaucher stem prefix or exact word)",
        "junk_token (skip underscores, proper nouns, digits)",
        "cross_list (remove words appearing in both masculine and feminine expansions)",
    ],
    "masculine": {
        "original_entries": MASCULINE_ENTRIES,
    },
    "feminine": {
        "original_entries": FEMININE_ENTRIES,
    },
}

for t in THRESHOLDS:
    output["masculine"][f"perword_{t}"] = sorted(masc_perword[t])
    output["masculine"][f"centroid_{t}"] = sorted(masc_centroid[t])
    output["feminine"][f"perword_{t}"] = sorted(fem_perword[t])
    output["feminine"][f"centroid_{t}"] = sorted(fem_centroid[t])

output_path = os.path.join(OUTPUT_DIR, "expanded_lexicon_w2v.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved expanded lexicon to: {output_path}")
