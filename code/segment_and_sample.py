"""
Sampling and Sentence Segmentation

Creates two samples of 300 postings each (75/industry): a random sample and
an enriched sample (top postings by gendered keyword density). Segments each
posting into sentences via NLTK sent_tokenize and flags sentences containing
Gaucher et al. (2011) stems or expanded lexicon words.

Author: Manya Reddy
Date:   2026-04-20
CS 72, Dartmouth College
"""

import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_postings.csv")
OUTPUT_RANDOM = os.path.join(BASE_DIR, "sample_random.csv")
OUTPUT_BIASED = os.path.join(BASE_DIR, "sample_biased.csv")

SAMPLE_PER_INDUSTRY = 75
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# STEP 1: Build the gendered word set
# ──────────────────────────────────────────────
# Original Gaucher et al. (2011) word lists as stems.
# Entries with * match any word starting with that prefix.
# Entries without * are exact-match only.

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

# Curated Word2Vec expansions (human-reviewed from cluster results)
MASCULINE_EXPANDED = ["adventuresome", "strong-willed", "manly"]
FEMININE_EXPANDED = [
    "loving", "endearing", "chatty", "perky", "jovial",
    "kindness", "caring", "subservient", "reliable",
]


def build_word_matcher(entries, expanded):
    """Build prefix and exact-match sets from Gaucher entries + curated expansions."""
    stems = []
    exact = set()
    for entry in entries:
        if entry.endswith("*"):
            stems.append(entry[:-1].lower())
        else:
            exact.add(entry.lower())
    for word in expanded:
        exact.add(word.lower())
    return stems, exact


def matches_gendered(word, stems, exact):
    """Check if a word matches any gendered entry."""
    w = word.lower()
    if w in exact:
        return True
    for stem in stems:
        if w.startswith(stem):
            return True
    return False


masc_stems, masc_exact = build_word_matcher(MASCULINE_ENTRIES, MASCULINE_EXPANDED)
fem_stems, fem_exact = build_word_matcher(FEMININE_ENTRIES, FEMININE_EXPANDED)

print(f"Masculine: {len(masc_stems)} stems + {len(masc_exact)} exact words")
print(f"Feminine:  {len(fem_stems)} stems + {len(fem_exact)} exact words")

# ──────────────────────────────────────────────
# STEP 2: Load cleaned postings
# ──────────────────────────────────────────────
print("Loading cleaned postings...")
df = pd.read_csv(CLEANED_PATH)
print(f"  Total postings: {len(df):,}")
print(f"  Industry breakdown:")
for group, count in df["industry_group"].value_counts().sort_index().items():
    print(f"    {group}: {count:,}")

# ──────────────────────────────────────────────
# STEP 3: Score each posting by gendered word hits
# ──────────────────────────────────────────────
def count_gendered_hits(text):
    """Count how many gendered stems/words match in the text."""
    # Tokenize into words for prefix matching
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
    count = 0
    all_stems = masc_stems + fem_stems
    all_exact = masc_exact | fem_exact
    matched = set()
    for w in words:
        if w in matched:
            continue
        if w in all_exact:
            matched.add(w)
            count += 1
            continue
        for stem in all_stems:
            if w.startswith(stem) and stem not in matched:
                matched.add(stem)
                count += 1
                break
    return count


print("Scoring postings by gendered word hits...")
df["gendered_hits"] = df["description_clean"].apply(count_gendered_hits)

# ──────────────────────────────────────────────
# STEP 4: Sample 75 per industry — two strategies
# ──────────────────────────────────────────────
# Random sample
random_ids = (
    df.groupby("industry_group", group_keys=False)
    .apply(lambda g: g.sample(n=min(SAMPLE_PER_INDUSTRY, len(g)), random_state=RANDOM_SEED))
    .index
)
random_sample = df.loc[random_ids].reset_index(drop=True)

# Biased sample: take the 75 with the most gendered word hits per industry
biased_ids = (
    df.groupby("industry_group", group_keys=False)
    .apply(lambda g: g.nlargest(min(SAMPLE_PER_INDUSTRY, len(g)), "gendered_hits"))
    .index
)
biased_sample = df.loc[biased_ids].reset_index(drop=True)

print(f"\nRandom sample: {len(random_sample)} postings")
print(f"  Avg gendered hits: {random_sample['gendered_hits'].mean():.1f}")
print(f"Biased sample: {len(biased_sample)} postings")
print(f"  Avg gendered hits: {biased_sample['gendered_hits'].mean():.1f}")

# ──────────────────────────────────────────────
# STEP 5: Segment into sentences
# ──────────────────────────────────────────────
def find_gendered_words(sentence):
    """Find all gendered words in a sentence, returning matched words and their direction."""
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", sentence.lower())
    masc_found = []
    fem_found = []
    seen = set()

    for w in words:
        if w in seen:
            continue
        # Check masculine
        if w in masc_exact:
            masc_found.append(w)
            seen.add(w)
            continue
        masc_match = False
        for stem in masc_stems:
            if w.startswith(stem):
                masc_found.append(w)
                seen.add(w)
                masc_match = True
                break
        if masc_match:
            continue
        # Check feminine
        if w in fem_exact:
            fem_found.append(w)
            seen.add(w)
            continue
        for stem in fem_stems:
            if w.startswith(stem):
                fem_found.append(w)
                seen.add(w)
                break

    return masc_found, fem_found


def segment_postings(sample_df, label):
    """Split each posting into sentences and return a flat DataFrame."""
    rows = []
    for _, posting in sample_df.iterrows():
        sentences = sent_tokenize(posting["description_clean"])
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            masc_found, fem_found = find_gendered_words(sentence)
            all_found = masc_found + fem_found
            rows.append({
                "job_id": posting["job_id"],
                "title": posting["title"],
                "industry_group": posting["industry_group"],
                "sentence_index": i,
                "sentence_text": sentence,
                "flagged": len(all_found) > 0,
                "masculine_words": ", ".join(masc_found) if masc_found else "",
                "feminine_words": ", ".join(fem_found) if fem_found else "",
                "all_flagged_words": ", ".join(all_found) if all_found else "",
            })
    result = pd.DataFrame(rows)
    flagged_count = result["flagged"].sum()
    print(f"\n{label}:")
    print(f"  Postings: {sample_df['job_id'].nunique()}")
    print(f"  Sentences: {len(result):,}")
    print(f"  Flagged sentences: {flagged_count:,} ({flagged_count/len(result)*100:.1f}%)")
    print(f"  Avg sentences per posting: {len(result) / sample_df['job_id'].nunique():.1f}")
    return result


print("\nSegmenting sentences...")
random_sentences = segment_postings(random_sample, "Random sample")
biased_sentences = segment_postings(biased_sample, "Biased sample")

# ──────────────────────────────────────────────
# STEP 6: Save
# ──────────────────────────────────────────────
random_sentences.to_csv(OUTPUT_RANDOM, index=False)
biased_sentences.to_csv(OUTPUT_BIASED, index=False)

print(f"\nSaved random sample to: {OUTPUT_RANDOM}")
print(f"Saved biased sample to: {OUTPUT_BIASED}")
