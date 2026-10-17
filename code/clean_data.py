"""
Data Cleaning for LinkedIn Job Postings

Filters ~124,000 raw LinkedIn postings to 4 target industries (tech, finance,
healthcare, education), strips HTML artifacts, removes EEO/boilerplate
disclaimers, and enforces a 150-word minimum. Outputs cleaned_postings.csv.

Author: Manya Reddy
Date:   2026-05-01
CS 72, Dartmouth College
"""

import pandas as pd
import re
import os

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTINGS_PATH = os.path.join(BASE_DIR, "postings.csv")
JOB_INDUSTRIES_PATH = os.path.join(BASE_DIR, "jobs", "job_industries.csv")
INDUSTRIES_PATH = os.path.join(BASE_DIR, "mappings", "industries.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "cleaned_postings.csv")

MIN_WORD_COUNT = 150

# ──────────────────────────────────────────────
# INDUSTRY DEFINITIONS
# ──────────────────────────────────────────────
INDUSTRY_MAP = {
    "tech": [
        4, 6, 96, 3, 5, 7, 8, 118, 109, 1594, 1855, 2458,
        3101, 3102, 3103, 3105, 3106, 3107, 3130, 3231, 3234, 3235,
    ],
    "finance": [
        41, 43, 45, 46, 47, 106, 129, 1673, 1696, 1713, 1720,
        # Uncomment to include insurance / real estate:
        # 42, 44,
    ],
    "healthcare": [
        13, 14, 15, 17, 139, 2045, 2054, 2069, 2074, 2081, 2091, 2358, 3217,
    ],
    "education": [
        67, 68, 69, 132, 1999, 2018, 3200, 3208,
    ],
}

# Reverse lookup: industry_id -> group label
ID_TO_GROUP = {}
for group, ids in INDUSTRY_MAP.items():
    for _id in ids:
        ID_TO_GROUP[_id] = group

# ──────────────────────────────────────────────
# BOILERPLATE PATTERNS (EEO disclaimers, etc.)
# ──────────────────────────────────────────────
BOILERPLATE_PATTERNS = [
    r"equal\s+opportunity\s+employer",
    r"we\s+are\s+an?\s+e\.?o\.?e\.?",
    r"all\s+qualified\s+applicants\s+will\s+receive\s+consideration",
    r"without\s+regard\s+to\s+race,?\s+color,?\s+religion",
    r"we\s+do\s+not\s+discriminate",
    r"affirmative\s+action",
    r"EEO\s*(is\s+the\s+law|statement|employer)",
    r"reasonable\s+accommodations?\s+(are|will\s+be)\s+made",
    r"if\s+you\s+(are\s+an?\s+individual|need)\s+(with\s+a\s+)?disabilit",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


def clean_description(text: str) -> str:
    """Strip HTML, special characters, and boilerplate from a job description."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove boilerplate sentences
    sentences = text.split(".")
    filtered = [s for s in sentences if not BOILERPLATE_RE.search(s)]
    text = ".".join(filtered)
    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # ── Load data ─────────────────────────────
    print("Loading postings...")
    postings = pd.read_csv(POSTINGS_PATH, usecols=["job_id", "title", "description"])
    print(f"  Raw postings: {len(postings):,}")

    print("Loading industry mappings...")
    job_industries = pd.read_csv(JOB_INDUSTRIES_PATH)
    industries = pd.read_csv(INDUSTRIES_PATH)

    # ── Filter to target industries ───────────
    target_ids = set(ID_TO_GROUP.keys())
    job_industries = job_industries[job_industries["industry_id"].isin(target_ids)]
    job_industries["industry_group"] = job_industries["industry_id"].map(ID_TO_GROUP)

    # Add industry name for reference
    job_industries = job_industries.merge(industries, on="industry_id", how="left")

    # Join with postings
    df = postings.merge(job_industries, on="job_id")
    # A posting can map to multiple industries; keep one row per posting,
    # preferring whichever target-group match comes first alphabetically.
    df = df.sort_values("industry_group").drop_duplicates(subset="job_id", keep="first")
    print(f"  Postings in target industries: {len(df):,}")

    # ── Drop missing descriptions ─────────────
    df = df.dropna(subset=["description"])
    print(f"  After dropping nulls: {len(df):,}")

    # ── Clean text ────────────────────────────
    print("Cleaning descriptions...")
    df["description_clean"] = df["description"].apply(clean_description)

    # ── Filter by word count ──────────────────
    df["word_count"] = df["description_clean"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORD_COUNT]
    print(f"  After {MIN_WORD_COUNT}-word minimum: {len(df):,}")

    # ── Final summary ─────────────────────────
    print("\nPostings per industry group:")
    counts = df["industry_group"].value_counts().sort_index()
    for group, count in counts.items():
        print(f"  {group}: {count:,}")

    # ── Save ──────────────────────────────────
    out_cols = [
        "job_id", "title", "industry_id", "industry_name",
        "industry_group", "description_clean", "word_count",
    ]
    df[out_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned data to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df):,}")


if __name__ == "__main__":
    main()
