# Detecting and Rewriting Gendered Language in Job Postings

A two-stage NLP pipeline that (1) fine-tunes BERT to classify job-posting sentences as **masculine-coded**, **feminine-coded**, or **neutral**, and (2) uses three LLMs (GPT-4o, Claude Sonnet 4, Llama 3.3 70B) to rewrite masculine-coded sentences as gender-neutral.

The work builds on Gaucher, Friesen & Kay (2011), who showed that masculine-coded wording in job ads (e.g. "competitive", "assertive") lowers women's interest in applying, while feminine-coded wording has no comparable effect on men. Keyword lists alone can't tell "competitive environment" (gendered) from "competitive salary" (structural), so this project uses a contextual classifier instead.

📄 **[Read the write-up](docs/writeup.pdf)** · [Appendix](docs/appendix.pdf) (rewrite prompt, human-evaluation survey, expanded lexicon, confusion-matrix heatmap)

*Final project for CS 72, Dartmouth College — Manya Reddy*

## Key results

**Classification.** The BERT model trained on the keyword-enriched sample reached **macro F1 = 0.85** and **93% accuracy** on a held-out test set of 3,716 sentences.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Masculine | 0.75 | 0.82 | 0.78 |
| Feminine | 0.80 | 0.83 | 0.81 |
| Neutral | 0.96 | 0.95 | — |

Most errors fall on the gendered-vs-neutral boundary. The model rarely confuses masculine with feminine.

**Industry analysis.** Applied to 10,000 unseen postings (2,500 per industry, 213,779 sentences), the classifier reproduces Gaucher et al.'s pattern in 2023–24 LinkedIn data: male-dominated industries contain more masculine language (**72.2% vs. 60.2%** of postings).

| Industry | Postings with masculine sentences | Postings with feminine sentences |
|---|---|---|
| Finance | 77.0% | 53.4% |
| Tech | 67.4% | 49.2% |
| Healthcare | 63.9% | 73.3% |
| Education | 56.6% | 70.6% |

**LLM rewrites.** Automated and human evaluation ranked the models in *opposite* orders:

| LLM | BERT success rate | Human pick: most gender-neutral | Human pick: best meaning preserved |
|---|---|---|---|
| Claude Sonnet 4 | **76.5%** | 16.7% | 5.6% |
| Llama 3.3 70B | 62.5% | **50.0%** | 33.3% |
| GPT-4o | 53.1% | 33.3% | **61.1%** |

Claude's minimal, single-word edits were enough to flip the classifier but didn't read as more neutral to people. Llama's broader rephrasing was preferred by humans. Automated metrics therefore need human evaluation alongside them. *(Human evaluation: 4 raters × 7 items, so these results are directional, not statistically significant.)*

## Pipeline

Scripts live in [`code/`](code/) and are listed in run order. Each script reads and writes files in its own directory.

| # | Script | What it does |
|---|---|---|
| 1 | `clean_data.py` | Filters ~124k postings to 4 industries (tech, finance, healthcare, education), strips HTML and EEO boilerplate, enforces a 150-word minimum → `cleaned_postings.csv` |
| 2 | `expand_lexicon_w2v.py` | Expands the Gaucher et al. word lists with Word2Vec (`word2vec-google-news-300`) |
| 3 | `segment_and_sample.py` | Builds a random sample and a keyword-enriched sample (300 postings each), splits them into sentences, and flags lexicon matches |
| 4 | `prelabel.py` | Has Claude pre-label flagged sentences and writes Excel workbooks for human review |
| — | *manual annotation* | Pre-labels were reviewed and corrected by hand |
| 5 | `train_bert.py` | Fine-tunes `bert-base-uncased` for 3-class classification with class-weighted loss |
| 6 | `industry_analysis.py` | Runs the classifier on 10,000 unseen postings to get per-industry statistics |
| 7 | `select_rewrite_candidates.py` | Picks the 3 most masculine-coded test postings per industry (12 total) |
| 8 | `rewrite_postings.py` | Sends those postings to GPT-4o, Claude, and Llama 3.3 for rewriting |
| 9 | `evaluate_rewrites.py` | Re-classifies original and rewritten sentences and reports success rates per LLM and industry |

## Data

**Not included in this repo.** The raw data is the [LinkedIn Job Postings (2023–2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) dataset on Kaggle. To run the pipeline, download it and place these files in `code/`:

```
code/postings.csv
code/jobs/job_industries.csv
code/mappings/industries.csv
```

The hand-annotated training data, the trained model checkpoints, and the LLM rewrite outputs are also not included. `train_bert.py` writes checkpoints to `output/`, which the later scripts load from automatically.

## Setup

```bash
pip install -r requirements.txt
```

You'll need a GPU for `train_bert.py` and `industry_analysis.py`.

**API keys** (only for the LLM steps):

- `prelabel.py` reads `ANTHROPIC_API_KEY` from the environment.
- `rewrite_postings.py` takes keys on the command line or from a JSON file:
  ```bash
  python rewrite_postings.py --api-keys keys.json
  # keys.json: {"openai": "...", "anthropic": "...", "together": "..."}
  ```
  `keys.json` is git-ignored. Never commit it.

## References

- Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality. *Journal of Personality and Social Psychology*, 101(1), 109–128.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
