"""
Stage 2: LLM Rewriting of Masculine-Coded Job Postings

Sends each of the 12 selected postings to 3 LLMs (GPT-4o, Claude Sonnet 4,
Llama 3.3 70B) with the same zero-shot prompt. Flags masculine sentences with
[FLAGGED] tags, receives rewrites with [EDITED] tags. Saves individual .txt
files and a combined all_rewrites.json.

Usage:
    python rewrite_postings.py --api-keys keys.json
    python rewrite_postings.py --openai-key KEY --anthropic-key KEY --together-key KEY

Author: Manya Reddy
Date:   2026-05-20
CS 72, Dartmouth College
"""

import os
import sys
import json
import time
import argparse
import openpyxl
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in inclusive hiring language. Your task is to rewrite a job posting to remove masculine-coded language while preserving the original meaning, tone, and professional quality.

Masculine-coded language refers to words and phrases associated with stereotypically male traits or agentic qualities. Research by Gaucher, Friesen & Kay (2011) found that such language deters women from applying to jobs, regardless of the actual role or industry.

Below are the masculine-coded and feminine-coded word lists used in this study. Words ending with * indicate prefix matches (e.g., "compet*" matches "competitive", "competition", etc.).

MASCULINE-CODED WORDS:
active, adventurous, aggress*, ambition, ambitious, analy*, assert*, athlet*, autonom*, boast*, challenge*, compet*, confident, courag*, decide, decisive, decision*, determin*, dominant, dominance, force*, greedy, headstrong, hierarch*, hostil*, impulsive, independent, independence, individual*, intellect*, lead*, logic, masculine, objective, opinion, outspoken, persist, principle*, reckless, stubborn, superior, self-confident, self-confidence, self-sufficience, self-sufficient, self-reliance, self-reliant, adventuresome, strong-willed, manly

FEMININE-CODED WORDS:
affectionate, child*, cheer*, commit*, communal, compassion*, connect*, considerate, cooperat*, depend*, emotional, empath*, feminine, flatterable, gentle, honest, interpersonal, interdependence, interdependent, kind, kinship, loyal*, modesty, nag, nurtur*, pleasant*, polite, quiet*, respon*, sensitiv*, submissive, support*, sympath*, tender*, together*, trust*, understand*, warm*, whin*, yield*, loving, endearing, chatty, perky, jovial, kindness, caring, subservient, reliable

INSTRUCTIONS:
- Sentences flagged by our classifier as potentially masculine-coded are marked with [FLAGGED]. Focus your rewriting on these sentences.
- For each flagged sentence, use your judgment: if the masculine-coded language describes traits, culture, values, or work style, rewrite it. If it is used in a neutral/structural context (e.g., "competitive salary", "independent contractor", "lead quarterly reviews"), leave it as-is.
- Replace masculine-coded words with gender-neutral alternatives that preserve the original meaning.
- Prefer gender-neutral replacements over feminine-coded alternatives.
- Do NOT modify sentences that are not flagged.
- Do NOT add new information, qualifications, or requirements.
- Do NOT change the structure or order of the posting.
- IMPORTANT: Preserve the original formatting. Each sentence must remain on its own line. Do NOT merge sentences into a single paragraph.
- Return the complete job posting. Replace [FLAGGED] with [EDITED] on sentences you modified. Remove [FLAGGED] from sentences you left unchanged."""

USER_PROMPT_TEMPLATE = """Rewrite the following job posting to remove masculine-coded language. Sentences marked [FLAGGED] were identified by a classifier as potentially masculine-coded.

{posting_text}"""


# ──────────────────────────────────────────────
# Load postings from Excel
# ──────────────────────────────────────────────
def load_postings(excel_path):
    """Load the 12 selected postings from rewrite_candidates.xlsx."""
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["All Sentences"]

    postings = {}
    order = []

    for row in ws.iter_rows(min_row=2, values_only=True):
        industry, job_id, title, sent_num, text, masc_prob, fem_prob, bert_pred, human_label, masc_words, fem_words = row
        job_id = str(int(job_id)) if isinstance(job_id, float) else str(job_id)
        key = (industry, job_id, title)

        if key not in postings:
            postings[key] = []
            order.append(key)

        postings[key].append({
            "sentence_num": sent_num,
            "text": text,
            "bert_prediction": bert_pred,
        })

    # Sort sentences within each posting
    for key in postings:
        postings[key].sort(key=lambda x: x["sentence_num"])

    return order, postings


def format_posting_for_llm(sentences):
    """Format a posting's sentences with [FLAGGED] tags on masculine ones."""
    lines = []
    for s in sentences:
        if s["bert_prediction"] == "masculine":
            lines.append(f"[FLAGGED] {s['text']}")
        else:
            lines.append(s["text"])
    return "\n".join(lines)


# ──────────────────────────────────────────────
# LLM API calls
# ──────────────────────────────────────────────
def call_openai(api_key, system_prompt, user_prompt, model="gpt-4o"):
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_anthropic(api_key, system_prompt, user_prompt, model="claude-sonnet-4-6"):
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.content[0].text


def call_together(api_key, system_prompt, user_prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """Call Together.ai API for Llama 3."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


LLM_CALLERS = {
    "gpt4": call_openai,
    "claude": call_anthropic,
    "llama3": call_together,
}

LLM_KEY_NAMES = {
    "gpt4": "openai",
    "claude": "anthropic",
    "llama3": "together",
}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Rewrite masculine-coded job postings with 3 LLMs")
    parser.add_argument("--api-keys", type=str, help="Path to JSON file with API keys")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--anthropic-key", type=str, help="Anthropic API key")
    parser.add_argument("--together-key", type=str, help="Together.ai API key")
    parser.add_argument("--llms", type=str, default="gpt4,claude,llama3",
                        help="Comma-separated list of LLMs to use (default: gpt4,claude,llama3)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "rewrites"),
                        help="Output directory")
    args = parser.parse_args()

    # Load API keys
    keys = {}
    if args.api_keys:
        with open(args.api_keys) as f:
            keys = json.load(f)
    if args.openai_key:
        keys["openai"] = args.openai_key
    if args.anthropic_key:
        keys["anthropic"] = args.anthropic_key
    if args.together_key:
        keys["together"] = args.together_key

    llms_to_run = [x.strip() for x in args.llms.split(",")]

    # Validate keys
    for llm in llms_to_run:
        key_name = LLM_KEY_NAMES[llm]
        if key_name not in keys:
            print(f"  ERROR: Missing API key for {llm} (need '{key_name}' key)")
            sys.exit(1)

    # Load postings
    excel_path = os.path.join(BASE_DIR, "rewrite_candidates.xlsx")
    print(f"  Loading postings from {excel_path}...")
    order, postings = load_postings(excel_path)
    print(f"  Loaded {len(order)} postings\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run each LLM on each posting
    all_results = []

    for llm_name in llms_to_run:
        key_name = LLM_KEY_NAMES[llm_name]
        api_key = keys[key_name]
        caller = LLM_CALLERS[llm_name]

        print(f"{'=' * 60}")
        print(f"  Running: {llm_name}")
        print(f"{'=' * 60}\n")

        for i, (industry, job_id, title) in enumerate(order, 1):
            sentences = postings[(industry, job_id, title)]
            posting_text = format_posting_for_llm(sentences)
            user_prompt = USER_PROMPT_TEMPLATE.format(posting_text=posting_text)

            masc_count = sum(1 for s in sentences if s["bert_prediction"] == "masculine")
            print(f"  [{i}/{len(order)}] {industry} — {title}")
            print(f"           {masc_count} flagged sentences, {len(sentences)} total")

            try:
                start = time.time()
                result = caller(api_key, SYSTEM_PROMPT, user_prompt)
                elapsed = time.time() - start
                print(f"           Done ({elapsed:.1f}s)\n")

                all_results.append({
                    "llm": llm_name,
                    "industry": industry,
                    "job_id": job_id,
                    "title": title,
                    "total_sentences": len(sentences),
                    "flagged_sentences": masc_count,
                    "original": posting_text,
                    "rewrite": result,
                })

                # Save individual result
                safe_title = title[:30].replace(" ", "_").replace("/", "-")
                out_path = os.path.join(args.output_dir, f"{llm_name}_{industry}_{safe_title}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"LLM: {llm_name}\n")
                    f.write(f"Industry: {industry}\n")
                    f.write(f"Job ID: {job_id}\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Flagged: {masc_count}/{len(sentences)}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(result)

            except Exception as e:
                print(f"           ERROR: {e}\n")
                all_results.append({
                    "llm": llm_name,
                    "industry": industry,
                    "job_id": job_id,
                    "title": title,
                    "total_sentences": len(sentences),
                    "flagged_sentences": masc_count,
                    "original": posting_text,
                    "rewrite": f"ERROR: {e}",
                })

            # Rate limiting
            time.sleep(1)

    # Save all results to JSON
    results_path = os.path.join(args.output_dir, "all_rewrites.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to {results_path}")

    # Save comparison CSV
    rows = []
    for r in all_results:
        rows.append({
            "LLM": r["llm"],
            "Industry": r["industry"],
            "Job ID": r["job_id"],
            "Title": r["title"],
            "Flagged": r["flagged_sentences"],
            "Total": r["total_sentences"],
            "Has Error": r["rewrite"].startswith("ERROR:"),
            "Edited Count": r["rewrite"].count("[EDITED]"),
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, "rewrite_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}\n")
    for llm_name in llms_to_run:
        llm_results = [r for r in all_results if r["llm"] == llm_name]
        errors = sum(1 for r in llm_results if r["rewrite"].startswith("ERROR:"))
        edited = sum(r["rewrite"].count("[EDITED]") for r in llm_results)
        flagged = sum(r["flagged_sentences"] for r in llm_results)
        print(f"  {llm_name}:")
        print(f"    Postings processed: {len(llm_results) - errors}/{len(llm_results)}")
        print(f"    Total [EDITED] tags: {edited} / {flagged} flagged")
        if errors:
            print(f"    Errors: {errors}")
        print()


if __name__ == "__main__":
    main()
