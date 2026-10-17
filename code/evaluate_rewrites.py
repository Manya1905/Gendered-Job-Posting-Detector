"""
Stage 2 Evaluation: BERT Assessment of LLM Rewrites

Runs the enriched multiclass model on both original and rewritten postings,
then compares sentence-level classification shifts. Reports success rates
(masculine-to-neutral/feminine) per LLM and per industry. Excludes 3 postings
with GPT-4o formatting issues for fair comparison across all models.

Usage:
    python evaluate_rewrites.py

Author: Manya Reddy
Date:   2026-05-25
CS 72, Dartmouth College
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path: check both local and server structure
_multi_local = os.path.join(BASE_DIR, "bert_gendered", "output", "biased_multiclass", "checkpoint-best")
_multi_server = os.path.join(BASE_DIR, "output", "biased_multiclass", "checkpoint-best")
MULTI_MODEL_DIR = _multi_local if os.path.isdir(_multi_local) else _multi_server

REWRITES_DIR = os.path.join(BASE_DIR, "rewrites")
LABEL_MAP = {0: "neutral", 1: "masculine", 2: "feminine"}

# Exclude postings where GPT-4 collapsed formatting (can't do sentence-level eval)
# Excluded for ALL LLMs to ensure fair comparison on same 9 postings
EXCLUDED_JOB_IDS = {"3862123581", "3902370373", "3902783342"}  # Financial Professional, CEO Blue Ridge, Director Customer Success


# ──────────────────────────────────────────────
# BERT Inference
# ──────────────────────────────────────────────
def run_multiclass_inference(model, tokenizer, texts, batch_size=32):
    """Classify sentences as masculine/feminine/neutral."""
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────
# Parse rewrites
# ──────────────────────────────────────────────
def parse_rewrite_text(text):
    """Split a rewritten posting into sentences, tracking [EDITED] tags."""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        edited = line.startswith("[EDITED]")
        clean = line.replace("[EDITED]", "").replace("[FLAGGED]", "").strip()
        if clean:
            sentences.append({"text": clean, "edited": edited})
    return sentences


def parse_original_text(text):
    """Split an original posting into sentences, tracking [FLAGGED] tags."""
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        flagged = line.startswith("[FLAGGED]")
        clean = line.replace("[FLAGGED]", "").strip()
        if clean:
            sentences.append({"text": clean, "flagged": flagged})
    return sentences


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EVALUATE LLM REWRITES WITH BERT")
    print("=" * 60)

    # Load rewrites
    rewrites_path = os.path.join(REWRITES_DIR, "all_rewrites.json")
    if not os.path.exists(rewrites_path):
        print(f"\n  ERROR: {rewrites_path} not found. Run rewrite_postings.py first.")
        sys.exit(1)

    with open(rewrites_path, "r", encoding="utf-8") as f:
        all_rewrites = json.load(f)

    # Filter out errors and excluded postings
    valid = [r for r in all_rewrites
             if not r["rewrite"].startswith("ERROR:")
             and r["job_id"] not in EXCLUDED_JOB_IDS]
    errored = sum(1 for r in all_rewrites if r["rewrite"].startswith("ERROR:"))
    excluded = sum(1 for r in all_rewrites if r["job_id"] in EXCLUDED_JOB_IDS)
    print(f"\n  Loaded {len(valid)} rewrites ({errored} errors skipped, {excluded} excluded for formatting)")
    print(f"  Excluded job IDs: {EXCLUDED_JOB_IDS}")

    # Load BERT model
    print(f"\n  Loading biased_multiclass from:\n    {MULTI_MODEL_DIR}")
    tokenizer = BertTokenizer.from_pretrained(MULTI_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MULTI_MODEL_DIR)
    print("  Model loaded.")

    # Process each rewrite
    all_results = []

    for i, entry in enumerate(valid, 1):
        llm = entry["llm"]
        industry = entry["industry"]
        job_id = entry["job_id"]
        title = entry["title"]

        print(f"\n  [{i}/{len(valid)}] {llm} — {industry} — {title[:40]}")

        # Parse original and rewrite
        original_sentences = parse_original_text(entry["original"])
        rewrite_sentences = parse_rewrite_text(entry["rewrite"])

        # Run BERT on original sentences
        orig_texts = [s["text"] for s in original_sentences]
        orig_probs = run_multiclass_inference(model, tokenizer, orig_texts)
        orig_preds = np.argmax(orig_probs, axis=1)

        # Run BERT on rewritten sentences
        rewrite_texts = [s["text"] for s in rewrite_sentences]
        rewrite_probs = run_multiclass_inference(model, tokenizer, rewrite_texts)
        rewrite_preds = np.argmax(rewrite_probs, axis=1)

        # Count labels
        orig_masc = int((orig_preds == 1).sum())
        orig_fem = int((orig_preds == 2).sum())
        orig_neut = int((orig_preds == 0).sum())

        rew_masc = int((rewrite_preds == 1).sum())
        rew_fem = int((rewrite_preds == 2).sum())
        rew_neut = int((rewrite_preds == 0).sum())

        # Track sentence-level shifts for flagged sentences
        flagged_indices = [j for j, s in enumerate(original_sentences) if s["flagged"]]
        shifts = {"masculine_to_neutral": 0, "masculine_to_feminine": 0,
                  "stayed_masculine": 0, "was_not_masculine": 0}

        sentence_details = []
        # Compare sentence by sentence (for flagged originals)
        # We match by index — if rewrite has different count, we note it
        min_len = min(len(orig_preds), len(rewrite_preds))

        for j in range(min_len):
            orig_label = LABEL_MAP[orig_preds[j]]
            rew_label = LABEL_MAP[rewrite_preds[j]]

            is_flagged = original_sentences[j]["flagged"] if j < len(original_sentences) else False
            was_edited = rewrite_sentences[j]["edited"] if j < len(rewrite_sentences) else False

            detail = {
                "sentence_num": j,
                "original_text": orig_texts[j] if j < len(orig_texts) else "",
                "rewrite_text": rewrite_texts[j] if j < len(rewrite_texts) else "",
                "original_bert": orig_label,
                "rewrite_bert": rew_label,
                "was_flagged": is_flagged,
                "was_edited": was_edited,
                "shifted": orig_label != rew_label,
            }
            sentence_details.append(detail)

            if is_flagged and orig_label == "masculine":
                if rew_label == "neutral":
                    shifts["masculine_to_neutral"] += 1
                elif rew_label == "feminine":
                    shifts["masculine_to_feminine"] += 1
                else:
                    shifts["stayed_masculine"] += 1
            elif is_flagged:
                shifts["was_not_masculine"] += 1

        flagged_masc_count = shifts["masculine_to_neutral"] + shifts["masculine_to_feminine"] + shifts["stayed_masculine"]
        success_count = shifts["masculine_to_neutral"] + shifts["masculine_to_feminine"]
        success_rate = success_count / flagged_masc_count * 100 if flagged_masc_count > 0 else 0

        print(f"           Original:  {orig_masc} masculine, {orig_fem} feminine, {orig_neut} neutral")
        print(f"           Rewrite:   {rew_masc} masculine, {rew_fem} feminine, {rew_neut} neutral")
        print(f"           Flagged masculine -> neutral/feminine: {success_count}/{flagged_masc_count} ({success_rate:.0f}%)")

        all_results.append({
            "llm": llm,
            "industry": industry,
            "job_id": job_id,
            "title": title,
            "original_total": len(orig_preds),
            "rewrite_total": len(rewrite_preds),
            "original_masculine": orig_masc,
            "original_feminine": orig_fem,
            "original_neutral": orig_neut,
            "rewrite_masculine": rew_masc,
            "rewrite_feminine": rew_fem,
            "rewrite_neutral": rew_neut,
            "masculine_reduction": orig_masc - rew_masc,
            "shifts": shifts,
            "success_rate": round(success_rate, 1),
            "sentence_details": sentence_details,
        })

    # ──────────────────────────────────────────
    # Summary tables
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS: Per-LLM Summary")
    print("=" * 60)

    print(f"\n  {'LLM':<10} {'Postings':<10} {'Orig Masc':<12} {'Rew Masc':<12} {'Reduction':<12} {'Success Rate'}")
    print(f"  {'-' * 68}")

    llm_names = sorted(set(r["llm"] for r in all_results))
    for llm in llm_names:
        llm_res = [r for r in all_results if r["llm"] == llm]
        orig_m = sum(r["original_masculine"] for r in llm_res)
        rew_m = sum(r["rewrite_masculine"] for r in llm_res)
        total_flagged_masc = sum(
            r["shifts"]["masculine_to_neutral"] + r["shifts"]["masculine_to_feminine"] + r["shifts"]["stayed_masculine"]
            for r in llm_res
        )
        total_success = sum(
            r["shifts"]["masculine_to_neutral"] + r["shifts"]["masculine_to_feminine"]
            for r in llm_res
        )
        rate = total_success / total_flagged_masc * 100 if total_flagged_masc > 0 else 0
        print(f"  {llm:<10} {len(llm_res):<10} {orig_m:<12} {rew_m:<12} {orig_m - rew_m:<12} {rate:.1f}%")

    print(f"\n\n  {'LLM':<10} {'Industry':<12} {'Orig Masc':<12} {'Rew Masc':<12} {'Success'}")
    print(f"  {'-' * 58}")

    for llm in llm_names:
        for ind in sorted(set(r["industry"] for r in all_results)):
            ind_res = [r for r in all_results if r["llm"] == llm and r["industry"] == ind]
            if not ind_res:
                continue
            orig_m = sum(r["original_masculine"] for r in ind_res)
            rew_m = sum(r["rewrite_masculine"] for r in ind_res)
            total_flagged_masc = sum(
                r["shifts"]["masculine_to_neutral"] + r["shifts"]["masculine_to_feminine"] + r["shifts"]["stayed_masculine"]
                for r in ind_res
            )
            total_success = sum(
                r["shifts"]["masculine_to_neutral"] + r["shifts"]["masculine_to_feminine"]
                for r in ind_res
            )
            rate = total_success / total_flagged_masc * 100 if total_flagged_masc > 0 else 0
            print(f"  {llm:<10} {ind:<12} {orig_m:<12} {rew_m:<12} {rate:.0f}%")

    # ──────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────
    output_dir = REWRITES_DIR

    # Save full results JSON (with sentence details)
    full_path = os.path.join(output_dir, "evaluation_full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results: {full_path}")

    # Save summary CSV
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "LLM": r["llm"],
            "Industry": r["industry"],
            "Job ID": r["job_id"],
            "Title": r["title"],
            "Original Masculine": r["original_masculine"],
            "Original Feminine": r["original_feminine"],
            "Original Neutral": r["original_neutral"],
            "Rewrite Masculine": r["rewrite_masculine"],
            "Rewrite Feminine": r["rewrite_feminine"],
            "Rewrite Neutral": r["rewrite_neutral"],
            "Masculine Reduction": r["masculine_reduction"],
            "Masc->Neutral": r["shifts"]["masculine_to_neutral"],
            "Masc->Feminine": r["shifts"]["masculine_to_feminine"],
            "Stayed Masculine": r["shifts"]["stayed_masculine"],
            "Success Rate %": r["success_rate"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV: {summary_path}")

    # Save sentence-level details CSV
    detail_rows = []
    for r in all_results:
        for s in r["sentence_details"]:
            detail_rows.append({
                "LLM": r["llm"],
                "Industry": r["industry"],
                "Job ID": r["job_id"],
                "Sentence #": s["sentence_num"],
                "Original Text": s["original_text"],
                "Rewrite Text": s["rewrite_text"],
                "Original BERT": s["original_bert"],
                "Rewrite BERT": s["rewrite_bert"],
                "Was Flagged": s["was_flagged"],
                "Was Edited": s["was_edited"],
                "Shifted": s["shifted"],
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(output_dir, "evaluation_sentences.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"  Sentence details: {detail_path}")

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
