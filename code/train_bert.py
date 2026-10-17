"""
BERT Fine-Tuning for Gendered Language Classification

Fine-tunes bert-base-uncased for 3-class sentence classification (masculine,
feminine, neutral) on annotated job posting sentences. Trains two models
(random and enriched samples), evaluates on a combined test set, and outputs
classification reports, confusion matrices, and PR curves.

Author: Manya Reddy
Date:   2026-05-10
CS 72, Dartmouth College
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import openpyxl
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
VALID_LABELS = {"masculine", "feminine", "both", "neutral"}
LABEL_TO_ID_MULTI = {"neutral": 0, "masculine": 1, "feminine": 2}
ID_TO_LABEL_MULTI = {v: k for k, v in LABEL_TO_ID_MULTI.items()}
LABEL_TO_ID_BINARY = {"neutral": 0, "gendered": 1}
ID_TO_LABEL_BINARY = {v: k for k, v in LABEL_TO_ID_BINARY.items()}

# ──────────────────────────────────────────────
# Data Loading:

# annotation_random.xlsx
#     skip junk rows
#     extract 11 columns per sentence
#     resolve label (you > haiku > neutral)
#     collapse "both" → "masculine"
#     add numeric ID columns for BERT
#     report any warnings
#  DataFrame ready for splitting and training
# ──────────────────────────────────────────────

def load_workbook_data(path, source_label):
    """Load all sentence rows from an annotation workbook."""
    print(f"  Loading {os.path.basename(path)}...")
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    skipped_separator = 0
    skipped_blank = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        industry = sheet_name.lower()

        for row in ws.iter_rows(min_row=2, values_only=True):
            cell_a = row[0]

            # Skip separator rows
            if cell_a is not None and str(cell_a).startswith("--- Posting:"):
                skipped_separator += 1
                continue

            # Skip blank rows
            if cell_a is None or str(cell_a).strip() == "":
                skipped_blank += 1
                continue

            # Skip rows with no sentence text
            sentence_text = row[3]
            if sentence_text is None or str(sentence_text).strip() == "":
                continue

            # Normalize job_id to string
            job_id = row[0]
            if isinstance(job_id, float):
                job_id = str(int(job_id))
            else:
                job_id = str(job_id)

            rows.append({
                "job_id": job_id,
                "title": row[1],
                "sentence_num": row[2],
                "sentence_text": str(sentence_text),
                "flagged": str(row[4]).strip().upper() == "YES" if row[4] else False,
                "masculine_words": row[5] or "",
                "feminine_words": row[6] or "",
                "suggested_label": str(row[7]).strip().lower() if row[7] else "",
                "your_label": str(row[9]).strip().lower() if row[9] else "",
                "industry": industry,
                "source": source_label,
            })

    df = pd.DataFrame(rows)
    print(f"    Sheets: {', '.join(wb.sheetnames)}")
    print(f"    Sentences: {len(df):,}")
    print(f"    Separator rows skipped: {skipped_separator}")
    print(f"    Blank rows skipped: {skipped_blank}")
    return df


def resolve_label(row):
    """Resolve final label: your_label > suggested_label > neutral."""
    your = row["your_label"]
    if your in VALID_LABELS:
        return your
    suggested = row["suggested_label"]
    if suggested in VALID_LABELS:
        return suggested
    return "neutral"


def collapse_both(label):
    """Collapse 'both' into 'masculine' per Gaucher et al. asymmetry finding."""
    return "masculine" if label == "both" else label


def to_binary_label(label):
    """Map 3-class label to binary: gendered (1) vs neutral (0)."""
    return "neutral" if label == "neutral" else "gendered"


def prepare_labels(df):
    """Resolve labels and add binary/multiclass columns."""
    df["resolved_label"] = df.apply(resolve_label, axis=1)
    # Collapse 'both' → 'masculine' for multi-class (Gaucher asymmetry)
    df["resolved_label"] = df["resolved_label"].apply(collapse_both)
    df["binary_label"] = df["resolved_label"].apply(to_binary_label)
    df["binary_id"] = df["binary_label"].map(LABEL_TO_ID_BINARY)
    df["multi_id"] = df["resolved_label"].map(LABEL_TO_ID_MULTI)

    # Count human overrides
    overrides = ((df["your_label"].isin(VALID_LABELS)) &
                 (df["your_label"] != df["suggested_label"])).sum()

    # Count flagged but unlabeled
    flagged_unlabeled = ((df["flagged"]) &
                         (~df["suggested_label"].isin(VALID_LABELS)) &
                         (~df["your_label"].isin(VALID_LABELS))).sum()

    return df, overrides, flagged_unlabeled


def print_label_stats(df, label, overrides, flagged_unlabeled):
    """Print label distribution statistics."""
    print(f"\n  {label} — Label Distribution (both → masculine):")
    for lbl in ["neutral", "masculine", "feminine"]:
        count = (df["resolved_label"] == lbl).sum()
        pct = count / len(df) * 100
        print(f"    {lbl:12s}: {count:,} ({pct:.1f}%)")
    print(f"    Human overrides: {overrides}")
    if flagged_unlabeled > 0:
        print(f"    WARNING: {flagged_unlabeled} flagged sentences had no label (defaulted to neutral)")


# ──────────────────────────────────────────────
# Train/Test Splitting:

# Full DataFrame (sentences)
#     collapse to posting level
#     stratify by industry x has_gendered
#     split posting IDs 80/20
#     recover all sentences for those postings
#  train_df: 80% of postings (all their sentences)
#  test_df:  20% of postings (all their sentences, never seen in training)
# ──────────────────────────────────────────────
def split_by_posting(df, test_size=0.2, seed=42):
    """Split at posting level, stratified by industry x gendered status."""
    posting_info = df.groupby("job_id").agg(
        industry=("industry", "first"),
        has_gendered=("binary_id", "max"),
    ).reset_index()

    posting_info["strat_key"] = (
        posting_info["industry"] + "_" + posting_info["has_gendered"].astype(str)
    )

    # Handle rare strata
    strat_counts = posting_info["strat_key"].value_counts()
    rare_strata = strat_counts[strat_counts < 2].index
    if len(rare_strata) > 0:
        print(f"    WARNING: Rare strata found ({list(rare_strata)}), falling back to industry-only stratification")
        posting_info["strat_key"] = posting_info["industry"]

    train_ids, test_ids = train_test_split(
        posting_info["job_id"],
        test_size=test_size,
        random_state=seed,
        stratify=posting_info["strat_key"],
    )

    train_df = df[df["job_id"].isin(set(train_ids))].copy()
    test_df = df[df["job_id"].isin(set(test_ids))].copy()

    return train_df, test_df


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class GenderedLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize_texts(texts, tokenizer, max_length):
    """Tokenize a list of texts."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ──────────────────────────────────────────────
# Custom Trainer with Class Weights
# ──────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

# Without class weights, every mistake costs the same. But, with them:
# predicting "neutral" when it's actually "gendered" → big penalty
# predicting "gendered" when it's actually "neutral" → small penalty
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, labels, weight=self.class_weights.to(logits.device)
            )
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def plot_confusion_matrix(cm, labels, output_path, title="Confusion Matrix"):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_scores, output_path):
    """Save precision-recall curve for binary classifier."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.axhline(y=0.85, color="r", linestyle="--", alpha=0.5, label="Precision = 0.85")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Binary: Gendered vs Neutral)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_at_threshold(y_true, y_scores, threshold):
    """Evaluate binary classifier at a specific confidence threshold."""
    preds = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_flagged": int(preds.sum()),
        "total": len(y_true),
    }


def evaluate_model(model, tokenizer, test_df, task, output_dir, args):
    """Full evaluation pipeline for one model."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine label columns and mappings
    if task == "binary":
        label_col = "binary_id"
        label_names = ["neutral", "gendered"]
        id_to_label = ID_TO_LABEL_BINARY
    else:
        label_col = "multi_id"
        label_names = ["neutral", "masculine", "feminine"]
        id_to_label = ID_TO_LABEL_MULTI

    texts = test_df["sentence_text"].tolist()
    y_true = test_df[label_col].values

    # Tokenize
    encodings = tokenize_texts(texts, tokenizer, args.max_length)
    dataset = GenderedLanguageDataset(encodings, torch.tensor(y_true, dtype=torch.long))

    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    # Overall classification report
    report = classification_report(y_true, preds, target_names=label_names, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    plot_confusion_matrix(cm, label_names, os.path.join(output_dir, "confusion_matrix.png"))

    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, preds)

    metrics = {
        "task": task,
        "num_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
    }

    # Threshold analysis for binary task
    if task == "binary":
        gendered_probs = probs[:, 1]
        plot_pr_curve(y_true, gendered_probs, os.path.join(output_dir, "pr_curve.png"))

        metrics["threshold_050"] = evaluate_at_threshold(y_true, gendered_probs, 0.50)
        metrics["threshold_085"] = evaluate_at_threshold(y_true, gendered_probs, 0.85)
        metrics["threshold_090"] = evaluate_at_threshold(y_true, gendered_probs, 0.90)

        print(f"  Threshold analysis:")
        for t in [0.50, 0.85, 0.90]:
            key = f"threshold_{int(t*100):03d}"
            m = metrics[key]
            print(f"    @{t}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (flagged {m['num_flagged']}/{m['total']})")

    # Per-industry breakdown
    industry_dir = os.path.join(output_dir, "per_industry")
    os.makedirs(industry_dir, exist_ok=True)
    metrics["per_industry"] = {}

    for industry in sorted(test_df["industry"].unique()):
        mask = test_df["industry"].values == industry
        if mask.sum() == 0:
            continue

        ind_true = y_true[mask]
        ind_preds = preds[mask]

        # Only include labels that appear in this industry's data
        present_labels = sorted(set(ind_true) | set(ind_preds))
        present_names = [label_names[i] for i in present_labels if i < len(label_names)]

        ind_report = classification_report(
            ind_true, ind_preds, target_names=present_names, zero_division=0
        )

        with open(os.path.join(industry_dir, f"{industry}.txt"), "w") as f:
            f.write(f"Industry: {industry}\n")
            f.write(f"Test samples: {mask.sum()}\n\n")
            f.write(ind_report)

        p, r, f1, _ = precision_recall_fscore_support(
            ind_true, ind_preds, average="macro", zero_division=0
        )
        metrics["per_industry"][industry] = {
            "num_samples": int(mask.sum()),
            "precision_macro": round(p, 4),
            "recall_macro": round(r, 4),
            "f1_macro": round(f1, 4),
        }

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(train_df, test_df, task, sample_name, model_name, output_dir, args):
    """Train and evaluate one BERT model."""
    model_id = f"{sample_name}_{task}"
    model_dir = os.path.join(output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Training: {model_id}")
    print(f"{'=' * 60}")

    # Determine labels
    if task == "binary":
        label_col = "binary_id"
        num_labels = 2
        label_names = ["neutral", "gendered"]
    else:
        label_col = "multi_id"
        num_labels = 3
        label_names = ["neutral", "masculine", "feminine"]

    train_labels = train_df[label_col].values
    test_labels = test_df[label_col].values

    # Print class distribution
    print(f"  Train samples: {len(train_df):,}")
    for i, name in enumerate(label_names):
        count = (train_labels == i).sum()
        print(f"    {name}: {count:,} ({count / len(train_labels) * 100:.1f}%)")

    print(f"  Test samples: {len(test_df):,}")

    # Compute class weights
    classes = np.arange(num_labels)
    present_classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=present_classes, y=train_labels)

    # Map weights to all classes (set weight=1.0 for missing classes)
    full_weights = np.ones(num_labels)
    for cls, w in zip(present_classes, weights):
        full_weights[cls] = w

    class_weights = torch.tensor(full_weights, dtype=torch.float32)
    print(f"  Class weights: {dict(zip(label_names, [round(w, 3) for w in full_weights]))}")

    # Tokenize
    print(f"  Tokenizing...")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_texts = train_df["sentence_text"].tolist()
    test_texts = test_df["sentence_text"].tolist()

    # Check for truncation
    train_lengths = [len(tokenizer.encode(t)) for t in train_texts]
    truncated = sum(1 for l in train_lengths if l > args.max_length)
    if truncated > 0:
        print(f"    {truncated} train sentences will be truncated (>{args.max_length} tokens)")

    train_encodings = tokenize_texts(train_texts, tokenizer, args.max_length)
    test_encodings = tokenize_texts(test_texts, tokenizer, args.max_length)

    train_dataset = GenderedLanguageDataset(
        train_encodings, torch.tensor(train_labels, dtype=torch.long)
    )
    test_dataset = GenderedLanguageDataset(
        test_encodings, torch.tensor(test_labels, dtype=torch.long)
    )

    # Initialize model
    print(f"  Initializing {model_name}...")
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        seed=args.seed,
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    # Train
    print(f"  Training for {args.epochs} epochs...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()

    # Save training log
    log_history = trainer.state.log_history
    with open(os.path.join(model_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"  Training complete. Loss: {train_result.training_loss:.4f}")

    # Save best model
    best_dir = os.path.join(model_dir, "checkpoint-best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"  Best model saved to {best_dir}")

    # Evaluate
    print(f"  Evaluating on test set...")
    metrics = evaluate_model(model, tokenizer, test_df, task, model_dir, args)
    metrics["training_loss"] = round(train_result.training_loss, 4)
    metrics["model_id"] = model_id
    metrics["training_data"] = sample_name
    metrics["num_train_samples"] = len(train_df)

    # Re-save metrics with training info
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ──────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────
def print_comparison(all_metrics, task, output_dir):
    """Print side-by-side comparison of random vs biased models."""
    random_m = next((m for m in all_metrics if m["model_id"] == f"random_{task}"), None)
    biased_m = next((m for m in all_metrics if m["model_id"] == f"biased_{task}"), None)

    if not random_m or not biased_m:
        return

    print(f"\n{'=' * 60}")
    print(f"  MODEL COMPARISON — {task.capitalize()} Classifiers")
    print(f"{'=' * 60}")
    print(f"  {'':25s} {'Random':>12s} {'Biased':>12s}")
    print(f"  {'-' * 49}")
    print(f"  {'Train samples':25s} {random_m['num_train_samples']:>12,} {biased_m['num_train_samples']:>12,}")
    print(f"  {'Test samples':25s} {random_m['num_test_samples']:>12,} {biased_m['num_test_samples']:>12,}")
    print(f"  {'Accuracy':25s} {random_m['accuracy']:>12.4f} {biased_m['accuracy']:>12.4f}")
    print(f"  {'Precision (macro)':25s} {random_m['precision_macro']:>12.4f} {biased_m['precision_macro']:>12.4f}")
    print(f"  {'Recall (macro)':25s} {random_m['recall_macro']:>12.4f} {biased_m['recall_macro']:>12.4f}")
    print(f"  {'F1 (macro)':25s} {random_m['f1_macro']:>12.4f} {biased_m['f1_macro']:>12.4f}")

    if task == "binary":
        print(f"\n  At 0.85 threshold:")
        for key, label in [("random", "Random"), ("biased", "Biased")]:
            m = random_m if key == "random" else biased_m
            t = m.get("threshold_085", {})
            if t:
                print(f"    {label}: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} (flagged {t['num_flagged']}/{t['total']})")

    # Per-industry comparison
    print(f"\n  Per-Industry F1 (macro):")
    print(f"  {'Industry':25s} {'Random':>12s} {'Biased':>12s}")
    print(f"  {'-' * 49}")
    industries = sorted(set(list(random_m.get("per_industry", {}).keys()) +
                            list(biased_m.get("per_industry", {}).keys())))
    for ind in industries:
        r_f1 = random_m.get("per_industry", {}).get(ind, {}).get("f1_macro", 0)
        b_f1 = biased_m.get("per_industry", {}).get(ind, {}).get("f1_macro", 0)
        print(f"  {ind:25s} {r_f1:>12.4f} {b_f1:>12.4f}")

    # Save comparison
    comparison = {
        "task": task,
        "random": random_m,
        "biased": biased_m,
    }
    comp_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    with open(os.path.join(comp_dir, f"{task}_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BERT fine-tuning for gendered language detection"
    )

    # Data paths
    parser.add_argument("--random-workbook", type=str, default="annotation_random.xlsx")
    parser.add_argument("--biased-workbook", type=str, default=None,
                        help="Path to biased workbook. If omitted, only random sample is used.")
    parser.add_argument("--output-dir", type=str, default="output")

    # Model
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)

    # Training
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # Splitting
    parser.add_argument("--test-size", type=float, default=0.2)

    # Evaluation
    parser.add_argument("--confidence-threshold", type=float, default=0.85)

    # Which models to train
    parser.add_argument("--tasks", nargs="+", default=["binary", "multiclass"],
                        choices=["binary", "multiclass"])
    parser.add_argument("--samples", nargs="+", default=["random", "biased"],
                        choices=["random", "biased"])

    # Eval-only mode: reload saved models and evaluate on new test set
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip training. Load saved models from output-dir and evaluate on test set.")

    # Misc
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")

    args = parser.parse_args()

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected — training will be slow")

    # ── Load Data ──
    print(f"\n{'=' * 60}")
    print(f"  DATA LOADING")
    print(f"{'=' * 60}")

    random_df = load_workbook_data(args.random_workbook, "random")
    random_df, r_overrides, r_unlabeled = prepare_labels(random_df)
    print_label_stats(random_df, "Random Sample", r_overrides, r_unlabeled)

    biased_df = None
    if args.biased_workbook and os.path.exists(args.biased_workbook):
        biased_df = load_workbook_data(args.biased_workbook, "biased")
        biased_df, b_overrides, b_unlabeled = prepare_labels(biased_df)
        print_label_stats(biased_df, "Biased Sample", b_overrides, b_unlabeled)
    elif "biased" in args.samples:
        print("\n  WARNING: Biased workbook not found, skipping biased models")
        args.samples = [s for s in args.samples if s != "biased"]

    # ── Split Data ──
    print(f"\n{'=' * 60}")
    print(f"  SPLITTING (posting-level, {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)})")
    print(f"{'=' * 60}")

    random_train, random_test = split_by_posting(random_df, args.test_size, args.seed)

    if biased_df is not None:
        biased_train, biased_test = split_by_posting(biased_df, args.test_size, args.seed)

        # Combined test set
        combined_test = pd.concat([random_test, biased_test], ignore_index=True)

        # Deduplicate by job_id + sentence_num (prefer random if overlap)
        before_dedup = len(combined_test)
        combined_test = combined_test.drop_duplicates(
            subset=["job_id", "sentence_num"], keep="first"
        )
        after_dedup = len(combined_test)
        if before_dedup != after_dedup:
            print(f"  Deduplicated test set: {before_dedup} -> {after_dedup}")
    else:
        biased_train = None
        combined_test = random_test

    print(f"\n  Random: {random_train['job_id'].nunique()} train postings / {random_test['job_id'].nunique()} test postings")
    print(f"    Train sentences: {len(random_train):,}")
    print(f"    Test sentences:  {len(random_test):,}")

    if biased_df is not None:
        print(f"\n  Biased: {biased_train['job_id'].nunique()} train postings / {biased_test['job_id'].nunique()} test postings")
        print(f"    Train sentences: {len(biased_train):,}")
        print(f"    Test sentences:  {len(biased_test):,}")

    print(f"\n  Test set: {len(combined_test):,} sentences from {combined_test['job_id'].nunique()} postings")
    for ind in sorted(combined_test["industry"].unique()):
        count = (combined_test["industry"] == ind).sum()
        print(f"    {ind}: {count:,}")

    # Save data stats
    os.makedirs(args.output_dir, exist_ok=True)
    data_stats = {
        "random": {
            "total_sentences": len(random_df),
            "train_sentences": len(random_train),
            "test_sentences": len(random_test),
            "train_postings": int(random_train["job_id"].nunique()),
            "test_postings": int(random_test["job_id"].nunique()),
            "label_distribution": random_df["resolved_label"].value_counts().to_dict(),
        },
        "combined_test_sentences": len(combined_test),
    }
    if biased_df is not None:
        data_stats["biased"] = {
            "total_sentences": len(biased_df),
            "train_sentences": len(biased_train),
            "test_sentences": len(biased_test),
            "train_postings": int(biased_train["job_id"].nunique()),
            "test_postings": int(biased_test["job_id"].nunique()),
            "label_distribution": biased_df["resolved_label"].value_counts().to_dict(),
        }
    with open(os.path.join(args.output_dir, "data_stats.json"), "w") as f:
        json.dump(data_stats, f, indent=2)

    # ── Train Models ──
    train_data = {"random": random_train}
    if biased_train is not None:
        train_data["biased"] = biased_train

    all_metrics = []

    if args.eval_only:
        # ── Eval-Only Mode: load saved models and re-evaluate ──
        print(f"\n{'=' * 60}")
        print(f"  EVAL-ONLY MODE — loading saved models")
        print(f"{'=' * 60}")

        for sample_name in args.samples:
            for task in args.tasks:
                model_id = f"{sample_name}_{task}"
                best_dir = os.path.join(args.output_dir, model_id, "checkpoint-best")

                if not os.path.exists(best_dir):
                    print(f"\n  Skipping {model_id} — no saved model at {best_dir}")
                    continue

                print(f"\n  Loading {model_id} from {best_dir}...")
                num_labels = 2 if task == "binary" else 3
                model = BertForSequenceClassification.from_pretrained(best_dir)
                tokenizer = BertTokenizer.from_pretrained(best_dir)

                model_dir = os.path.join(args.output_dir, model_id)
                metrics = evaluate_model(model, tokenizer, combined_test, task, model_dir, args)
                metrics["model_id"] = model_id
                metrics["training_data"] = sample_name

                with open(os.path.join(model_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

                all_metrics.append(metrics)
    else:
        # ── Training Mode ──
        for sample_name in args.samples:
            if sample_name not in train_data:
                print(f"\n  Skipping {sample_name} — no training data loaded")
                continue
            for task in args.tasks:
                try:
                    metrics = train_model(
                        train_df=train_data[sample_name],
                        test_df=combined_test,
                        task=task,
                        sample_name=sample_name,
                        model_name=args.model_name,
                        output_dir=args.output_dir,
                        args=args,
                    )
                    all_metrics.append(metrics)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"\n  ERROR: GPU out of memory for {sample_name}_{task}.")
                        print(f"  Try reducing --batch-size (current: {args.batch_size})")
                        continue
                    raise

    # ── Comparisons ──
    if len(all_metrics) > 1:
        for task in args.tasks:
            print_comparison(all_metrics, task, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {os.path.abspath(args.output_dir)}")
    print(f"  Models trained: {len(all_metrics)}")


if __name__ == "__main__":
    main()
