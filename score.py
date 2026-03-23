"""
Score generated text samples for behavioral metrics.

Usage:
    uv run score.py --input samples/seed_42.json --output scores/seed_42.csv
"""

import argparse
import csv
import json
import math
import re
from collections import Counter


def tokenize_words(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def distinct_n(words, n):
    """Fraction of unique n-grams out of total n-grams."""
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def lexical_diversity(words):
    """Type-token ratio."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def repetition_rate(text):
    """Fraction of repeated lines in the text."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) <= 1:
        return 0.0
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(lines)


def word_entropy(words):
    """Shannon entropy of word distribution (bits)."""
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def avg_sentence_length(text):
    """Average number of words per sentence."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(tokenize_words(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def punctuation_rate(text):
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')
    return punct / len(text)


def degeneration_score(text):
    """Detect repetitive degeneration — high score means more degenerate."""
    words = tokenize_words(text)
    if len(words) < 10:
        return 0.0
    # Check for long runs of the same word
    max_run = 1
    current_run = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    # Check for repeated bigrams
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    if bigrams:
        max_bigram_repeat = max(bigram_counts.values())
    else:
        max_bigram_repeat = 0
    # Normalize: max_run > 3 or bigram repeated > 5x is degenerate
    run_score = min(max_run / 5.0, 1.0)
    bigram_score = min(max_bigram_repeat / 8.0, 1.0)
    return (run_score + bigram_score) / 2


def vocab_coverage(words, total_vocab_estimate=5000):
    """Unique words as fraction of estimated vocabulary."""
    if not words:
        return 0.0
    return min(len(set(words)) / total_vocab_estimate, 1.0)


def score_sample(text, prefix):
    """Compute all behavioral metrics for a single sample."""
    # Strip prefix from generated text for scoring
    continuation = text[len(prefix):].strip() if text.startswith(prefix) else text
    words = tokenize_words(continuation)

    return {
        "word_count": len(words),
        "unique_words": len(set(words)),
        "lexical_diversity": round(lexical_diversity(words), 4),
        "distinct_1": round(distinct_n(words, 1), 4),
        "distinct_2": round(distinct_n(words, 2), 4),
        "distinct_3": round(distinct_n(words, 3), 4),
        "word_entropy": round(word_entropy(words), 4),
        "avg_sentence_length": round(avg_sentence_length(continuation), 2),
        "repetition_rate": round(repetition_rate(continuation), 4),
        "punctuation_rate": round(punctuation_rate(continuation), 4),
        "degeneration_score": round(degeneration_score(continuation), 4),
        "vocab_coverage": round(vocab_coverage(words), 4),
    }


METRIC_NAMES = [
    "word_count", "unique_words", "lexical_diversity",
    "distinct_1", "distinct_2", "distinct_3",
    "word_entropy", "avg_sentence_length", "repetition_rate",
    "punctuation_rate", "degeneration_score", "vocab_coverage",
]


def main():
    parser = argparse.ArgumentParser(description="Score generated text samples")
    parser.add_argument("--input", required=True, help="Input JSON file from sample.py")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    rows = []
    for sample in data["samples"]:
        scores = score_sample(sample["text"], sample["prefix"])
        row = {
            "prefix": sample["prefix"][:60],
            "temperature": sample["temperature"],
            "sample_idx": sample["sample_idx"],
            **scores,
        }
        rows.append(row)

    fieldnames = ["prefix", "temperature", "sample_idx"] + METRIC_NAMES
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} samples → {args.output}")

    # Print summary stats
    print("\nPer-metric means:")
    for metric in METRIC_NAMES:
        vals = [r[metric] for r in rows]
        mean = sum(vals) / len(vals)
        print(f"  {metric:24s}: {mean:.4f}")


if __name__ == "__main__":
    main()
