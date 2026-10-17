"""
Corpus-Wide Industry Analysis

Runs the enriched multiclass BERT model on 10,000 unlabeled postings (2,500
per industry, excluding training/testing data) to produce per-industry gendered
language statistics. Validates that BERT's predictions align with Gaucher et
al. (2011): male-dominated industries use more masculine language, female-
dominated industries use more feminine language.

Outputs:
  - industry_analysis.xlsx: Per-industry breakdown with statistics
  - industry_analysis_sentences.csv: All 213,779 sentence-level predictions
  - industry_analysis_summary.json: Machine-readable summary

Author: Manya Reddy
Date:   2026-05-15
CS 72, Dartmouth College
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import openpyxl
from openpyxl.styles import Font, Alignment
import json

# punkt data already installed at ~/nltk_data — skip download on compute nodes
# (no internet access on SLURM compute nodes)
os.environ.setdefault("NLTK_DATA", os.path.expanduser("~/nltk_data"))

# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=0,
                    help="Number of postings to sample (0 = use all). Default: 0 (all)")
parser.add_argument("--threshold", type=float, default=0.85,
                    help="Confidence threshold for gendered classification. Default: 0.85")
parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for inference. Default: 64")
parser.add_argument("--skip-multiclass", action="store_true",
                    help="Skip multiclass model (faster, binary only)")
parser.add_argument("--multiclass-only", action="store_true",
                    help="Run only multiclass model (masculine/feminine/neutral)")
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths — check server vs local structure
_binary_local = os.path.join(BASE_DIR, "bert_gendered", "output", "biased_binary", "checkpoint-best")
_binary_server = os.path.join(BASE_DIR, "output", "biased_binary", "checkpoint-best")
BINARY_MODEL_DIR = _binary_local if os.path.isdir(_binary_local) else _binary_server

_multi_local = os.path.join(BASE_DIR, "bert_gendered", "output", "biased_multiclass", "checkpoint-best")
_multi_server = os.path.join(BASE_DIR, "output", "biased_multiclass", "checkpoint-best")
MULTI_MODEL_DIR = _multi_local if os.path.isdir(_multi_local) else _multi_server

CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_postings.csv")

# Confidence threshold from args
CONFIDENCE_THRESHOLD = args.threshold


# ──────────────────────────────────────────────
# Sentence Segmentation (same as segment_and_sample.py)
# ──────────────────────────────────────────────
def segment_posting(text):
    """Split a posting into sentences, filtering very short ones."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


# ──────────────────────────────────────────────
# Get IDs to Exclude (training + test postings)
# ──────────────────────────────────────────���───
def get_excluded_ids():
    """Load the job IDs used in training/testing from annotation workbooks."""
    excluded = set()

    for filename in ["annotation_random.xlsx", "annotation_biased.xlsx"]:
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {filename} not found, cannot exclude its IDs")
            continue

        wb = openpyxl.load_workbook(path, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for row in ws.iter_rows(min_row=2, values_only=True):
                cell_a = row[0]
                if cell_a is None or str(cell_a).strip() == "":
                    continue
                if str(cell_a).startswith("--- Posting:"):
                    continue
                job_id = row[0]
                if isinstance(job_id, float):
                    job_id = str(int(job_id))
                else:
                    job_id = str(job_id)
                excluded.add(job_id)

    return excluded


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, tokenizer, texts, batch_size=64):
    """Run BERT inference and return all class probabilities."""
    model = model.to(DEVICE)
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        if (i // batch_size) % 50 == 0:
            done = min(i + batch_size, len(texts))
            print(f"    {done:,}/{len(texts):,} sentences processed...", end="\r")

    print(f"    {len(texts):,}/{len(texts):,} sentences processed.      ")
    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CORPUS-WIDE INDUSTRY ANALYSIS")
    print("=" * 60)
    print(f"\n  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load cleaned postings ──
    print("\n  Loading cleaned postings...")
    df = pd.read_csv(CLEANED_PATH)
    print(f"  Total postings in corpus: {len(df):,}")

    # ── Exclude training/test IDs ──
    print("\n  Identifying postings to exclude (used in training/testing)...")
    excluded_ids = get_excluded_ids()
    print(f"  Excluding {len(excluded_ids)} posting IDs")

    df["job_id_str"] = df["job_id"].astype(str)
    df = df[~df["job_id_str"].isin(excluded_ids)].copy()
    print(f"  Remaining postings for analysis: {len(df):,}")

    # ── Subsample if requested ──
    if args.subsample > 0 and args.subsample < len(df):
        per_industry = args.subsample // 4  # Equal samples per industry
        print(f"\n  Subsampling {per_industry:,} postings per industry ({per_industry * 4:,} total)...")
        df = df.groupby("industry_group", group_keys=False).apply(
            lambda g: g.sample(
                n=min(per_industry, len(g)),
                random_state=42
            )
        ).reset_index(drop=True)
        print(f"  Subsampled to: {len(df):,} postings")

    print(f"  Per industry:")
    for ind in sorted(df["industry_group"].unique()):
        count = (df["industry_group"] == ind).sum()
        print(f"    {ind}: {count:,}")

    # ── Segment into sentences ──
    print("\n  Segmenting postings into sentences...")
    sentence_rows = []
    for idx, row in df.iterrows():
        sentences = segment_posting(row["description_clean"])
        for i, sent in enumerate(sentences):
            sentence_rows.append({
                "job_id": str(row["job_id"]),
                "title": row["title"],
                "industry": row["industry_group"],
                "sentence_num": i,
                "sentence_text": sent,
            })

        if (idx + 1) % 5000 == 0:
            print(f"    Segmented {idx + 1:,}/{len(df):,} postings...", end="\r")

    sentences_df = pd.DataFrame(sentence_rows)
    print(f"    Segmented {len(df):,}/{len(df):,} postings.        ")
    print(f"  Total sentences: {len(sentences_df):,}")
    print(f"  Avg sentences per posting: {len(sentences_df) / len(df):.1f}")

    texts = sentences_df["sentence_text"].tolist()

    # ── Load and run models ──
    if args.multiclass_only:
        # Only multiclass: masculine / feminine / neutral
        print(f"\n  Loading biased_multiclass model from:\n    {MULTI_MODEL_DIR}")
        multi_tokenizer = BertTokenizer.from_pretrained(MULTI_MODEL_DIR)
        multi_model = BertForSequenceClassification.from_pretrained(MULTI_MODEL_DIR)
        print(f"  Multiclass model loaded.")

        print(f"\n  Running multiclass inference...")
        multi_probs = run_inference(multi_model, multi_tokenizer, texts, batch_size=args.batch_size)
        # Multiclass: 0=neutral, 1=masculine, 2=feminine
        sentences_df["masculine_prob"] = multi_probs[:, 1]
        sentences_df["feminine_prob"] = multi_probs[:, 2]
        sentences_df["predicted_class"] = np.argmax(multi_probs, axis=1)
        sentences_df["predicted_label"] = sentences_df["predicted_class"].map(
            {0: "neutral", 1: "masculine", 2: "feminine"}
        )
        # Derive binary gendered from multiclass (anything not neutral = gendered)
        sentences_df["predicted_gendered"] = (sentences_df["predicted_class"] != 0).astype(int)
        sentences_df["gendered_prob"] = 1.0 - multi_probs[:, 0]  # P(not neutral)

    else:
        # Binary model
        if not args.skip_multiclass or not args.multiclass_only:
            print(f"\n  Loading biased_binary model from:\n    {BINARY_MODEL_DIR}")
            binary_tokenizer = BertTokenizer.from_pretrained(BINARY_MODEL_DIR)
            binary_model = BertForSequenceClassification.from_pretrained(BINARY_MODEL_DIR)
            print(f"  Binary model loaded.")

            print(f"\n  Running binary inference (threshold={CONFIDENCE_THRESHOLD})...")
            binary_probs = run_inference(binary_model, binary_tokenizer, texts, batch_size=args.batch_size)
            sentences_df["gendered_prob"] = binary_probs[:, 1]
            sentences_df["predicted_gendered"] = (binary_probs[:, 1] >= CONFIDENCE_THRESHOLD).astype(int)

        # Multiclass model
        if not args.skip_multiclass:
            print(f"\n  Loading biased_multiclass model from:\n    {MULTI_MODEL_DIR}")
            multi_tokenizer = BertTokenizer.from_pretrained(MULTI_MODEL_DIR)
            multi_model = BertForSequenceClassification.from_pretrained(MULTI_MODEL_DIR)
            print(f"  Multiclass model loaded.")

            print(f"\n  Running multiclass inference...")
            multi_probs = run_inference(multi_model, multi_tokenizer, texts, batch_size=args.batch_size)
            sentences_df["masculine_prob"] = multi_probs[:, 1]
            sentences_df["feminine_prob"] = multi_probs[:, 2]
            sentences_df["predicted_class"] = np.argmax(multi_probs, axis=1)
            sentences_df["predicted_label"] = sentences_df["predicted_class"].map(
                {0: "neutral", 1: "masculine", 2: "feminine"}
            )
        else:
            sentences_df["masculine_prob"] = 0.0
            sentences_df["feminine_prob"] = 0.0
            sentences_df["predicted_class"] = sentences_df["predicted_gendered"]
            sentences_df["predicted_label"] = sentences_df["predicted_gendered"].map(
                {0: "neutral", 1: "gendered"}
            )

    # ── Save raw sentence-level predictions ──
    raw_output_path = os.path.join(BASE_DIR, "industry_analysis_sentences.csv")
    print(f"\n  Saving sentence-level predictions to {raw_output_path}...")
    sentences_df.to_csv(raw_output_path, index=False)
    print(f"  Saved: {len(sentences_df):,} rows")

    # ── Aggregate at posting level ──
    print("\n  Aggregating at posting level...")
    posting_stats = sentences_df.groupby("job_id").agg(
        title=("title", "first"),
        industry=("industry", "first"),
        total_sentences=("sentence_text", "count"),
        gendered_count=("predicted_gendered", "sum"),
        masculine_count=("predicted_class", lambda x: (x == 1).sum()),
        feminine_count=("predicted_class", lambda x: (x == 2).sum()),
        mean_gendered_prob=("gendered_prob", "mean"),
        max_gendered_prob=("gendered_prob", "max"),
    ).reset_index()

    posting_stats["gendered_proportion"] = posting_stats["gendered_count"] / posting_stats["total_sentences"]
    posting_stats["masculine_proportion"] = posting_stats["masculine_count"] / posting_stats["total_sentences"]
    posting_stats["feminine_proportion"] = posting_stats["feminine_count"] / posting_stats["total_sentences"]
    posting_stats["has_gendered"] = (posting_stats["gendered_count"] > 0).astype(int)
    posting_stats["has_masculine"] = (posting_stats["masculine_count"] > 0).astype(int)
    posting_stats["has_feminine"] = (posting_stats["feminine_count"] > 0).astype(int)

    # ── Industry-level summary ──
    print("\n  Computing industry-level statistics...")
    industry_stats = posting_stats.groupby("industry").agg(
        total_postings=("job_id", "count"),
        postings_with_gendered=("has_gendered", "sum"),
        postings_with_masculine=("has_masculine", "sum"),
        postings_with_feminine=("has_feminine", "sum"),
        avg_gendered_proportion=("gendered_proportion", "mean"),
        avg_masculine_proportion=("masculine_proportion", "mean"),
        avg_feminine_proportion=("feminine_proportion", "mean"),
        avg_gendered_count=("gendered_count", "mean"),
        avg_masculine_count=("masculine_count", "mean"),
        avg_feminine_count=("feminine_count", "mean"),
        median_gendered_count=("gendered_count", "median"),
    ).reset_index()

    industry_stats["pct_with_gendered"] = (
        industry_stats["postings_with_gendered"] / industry_stats["total_postings"] * 100
    ).round(1)
    industry_stats["pct_with_masculine"] = (
        industry_stats["postings_with_masculine"] / industry_stats["total_postings"] * 100
    ).round(1)
    industry_stats["pct_with_feminine"] = (
        industry_stats["postings_with_feminine"] / industry_stats["total_postings"] * 100
    ).round(1)

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  RESULTS: Per-Industry Gendered Language Prevalence")
    print("=" * 60)
    print(f"\n  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Postings analyzed: {len(posting_stats):,}")
    print(f"  Sentences analyzed: {len(sentences_df):,}")

    print(f"\n  {'Industry':<14} {'Postings':<10} {'% w/ Masc':<12} {'% w/ Fem':<12} {'Avg Masc/post':<16} {'Avg Fem/post'}")
    print(f"  {'-'*75}")
    for _, row in industry_stats.sort_values("pct_with_masculine", ascending=False).iterrows():
        print(f"  {row['industry']:<14} {int(row['total_postings']):<10} "
              f"{row['pct_with_masculine']:.1f}%       {row['pct_with_feminine']:.1f}%       "
              f"{row['avg_masculine_count']:.2f}            {row['avg_feminine_count']:.2f}")

    print(f"\n  Expected pattern (from prior research):")
    print(f"    Tech & Finance    → higher masculine prevalence")
    print(f"    Healthcare & Edu  → higher feminine prevalence")

    # Check if pattern holds
    masc_by_industry = industry_stats.set_index("industry")["pct_with_masculine"]
    fem_by_industry = industry_stats.set_index("industry")["pct_with_feminine"]

    male_dom = (masc_by_industry.get("tech", 0) + masc_by_industry.get("finance", 0)) / 2
    fem_dom = (masc_by_industry.get("healthcare", 0) + masc_by_industry.get("education", 0)) / 2
    print(f"\n  Avg % with masculine sentences:")
    print(f"    Tech + Finance (male-dominated):         {male_dom:.1f}%")
    print(f"    Healthcare + Education (female-dominated): {fem_dom:.1f}%")
    if male_dom > fem_dom:
        print(f"    --> Pattern CONFIRMED: male-dominated industries have more masculine language")
    else:
        print(f"    --> Pattern NOT confirmed")

    fem_female_dom = (fem_by_industry.get("healthcare", 0) + fem_by_industry.get("education", 0)) / 2
    fem_male_dom = (fem_by_industry.get("tech", 0) + fem_by_industry.get("finance", 0)) / 2
    print(f"\n  Avg % with feminine sentences:")
    print(f"    Healthcare + Education (female-dominated): {fem_female_dom:.1f}%")
    print(f"    Tech + Finance (male-dominated):         {fem_male_dom:.1f}%")
    if fem_female_dom > fem_male_dom:
        print(f"    --> Pattern CONFIRMED: female-dominated industries have more feminine language")
    else:
        print(f"    --> Pattern NOT confirmed")

    # ── Export to Excel ──
    output_path = os.path.join(BASE_DIR, "industry_analysis.xlsx")
    print(f"\n  Exporting results to {output_path}...")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Industry summary
        industry_export = industry_stats[[
            "industry", "total_postings",
            "pct_with_masculine", "pct_with_feminine", "pct_with_gendered",
            "avg_masculine_count", "avg_feminine_count", "avg_gendered_count",
            "avg_masculine_proportion", "avg_feminine_proportion", "avg_gendered_proportion",
        ]].copy()
        industry_export.columns = [
            "Industry", "Total Postings",
            "% with Masculine", "% with Feminine", "% with Any Gendered",
            "Avg Masculine Sentences/Posting", "Avg Feminine Sentences/Posting", "Avg Gendered Sentences/Posting",
            "Avg Masculine Proportion", "Avg Feminine Proportion", "Avg Gendered Proportion",
        ]
        industry_export.to_excel(writer, sheet_name="Industry Summary", index=False)

        # Sheet 2: Posting-level stats (full)
        posting_export = posting_stats[[
            "job_id", "title", "industry", "total_sentences",
            "gendered_count", "masculine_count", "feminine_count",
            "gendered_proportion", "masculine_proportion", "feminine_proportion",
            "mean_gendered_prob", "max_gendered_prob"
        ]].sort_values(["industry", "gendered_proportion"], ascending=[True, False])
        posting_export.to_excel(writer, sheet_name="All Postings", index=False)

        # Sheet 3: Top 20 most masculine per industry
        top_masc = posting_stats.sort_values("masculine_proportion", ascending=False).groupby("industry").head(20)
        top_masc = top_masc[["job_id", "title", "industry", "total_sentences",
                             "masculine_count", "masculine_proportion"]].sort_values(
            ["industry", "masculine_proportion"], ascending=[True, False]
        )
        top_masc.to_excel(writer, sheet_name="Top Masculine", index=False)

        # Sheet 4: Top 20 most feminine per industry
        top_fem = posting_stats.sort_values("feminine_proportion", ascending=False).groupby("industry").head(20)
        top_fem = top_fem[["job_id", "title", "industry", "total_sentences",
                           "feminine_count", "feminine_proportion"]].sort_values(
            ["industry", "masculine_proportion" if "masculine_proportion" in top_fem.columns else "feminine_proportion"],
            ascending=[True, False]
        )
        top_fem.to_excel(writer, sheet_name="Top Feminine", index=False)

    print(f"  Saved: {output_path}")

    # ── Save JSON summary ──
    json_path = os.path.join(BASE_DIR, "industry_analysis_summary.json")
    summary = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_postings_analyzed": len(posting_stats),
        "total_sentences_analyzed": len(sentences_df),
        "excluded_posting_ids": len(excluded_ids),
        "per_industry": {}
    }
    for _, row in industry_stats.iterrows():
        summary["per_industry"][row["industry"]] = {
            "total_postings": int(row["total_postings"]),
            "pct_with_masculine": round(row["pct_with_masculine"], 2),
            "pct_with_feminine": round(row["pct_with_feminine"], 2),
            "pct_with_gendered": round(row["pct_with_gendered"], 2),
            "avg_masculine_per_posting": round(row["avg_masculine_count"], 3),
            "avg_feminine_per_posting": round(row["avg_feminine_count"], 3),
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
