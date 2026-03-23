"""
Generate text continuations from a trained model checkpoint.

Usage:
    uv run sample.py --checkpoint checkpoints/seed_42.pt --output samples/seed_42.json
    uv run sample.py --checkpoint checkpoints/seed_42.pt --temperature 0.8 1.0 --samples-per-prefix 5
"""

import argparse
import hashlib
import json

import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer

# Import model classes from train.py — we only need GPT and GPTConfig
# but train.py runs training on import, so we replicate config/model here minimally.
# Instead, we load the full model from checkpoint which includes config.


# ---------------------------------------------------------------------------
# Prefixes for behavioral evaluation
# ---------------------------------------------------------------------------

PREFIXES = [
    # Neutral prose
    "The city had changed since the last time she visited.",
    "After the rain stopped, the streets were quiet and still.",
    "He opened the book to a random page and began to read.",
    # Historical / political
    "The envoy arrived at dawn with gifts from the northern court.",
    "The minister bowed and said the neighboring state was weak.",
    "In the third year of famine, the granaries stood nearly empty.",
    "The emperor summoned his advisors to discuss the border threat.",
    # Strategic / conflict
    "The ruler asked whether it was better to submit or resist.",
    "The general surveyed the battlefield and considered his options.",
    "When the alliance collapsed, each kingdom looked to its own defense.",
    # Abstract / philosophical
    "The nature of power is that it changes those who hold it.",
    "What is remembered depends on who survives to tell the story.",
    # Conversational / mundane
    "The merchant counted his coins and frowned at the total.",
    "She walked to the edge of the river and looked across.",
    # Technical / descriptive
    "The structure of the bridge relied on three main supports.",
]


def load_model_and_tokenizer(checkpoint_path):
    """Load model from checkpoint. Checkpoint must contain 'config' and 'model_state_dict'."""
    from model_def import GPT, GPTConfig

    # Register GPTConfig so pickle can resolve it when loading checkpoints
    # saved from train.py (where GPTConfig lives in __main__)
    import __main__
    __main__.GPTConfig = GPTConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = Tokenizer.from_directory()

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")

    # Load weights (handle compiled model state dict keys)
    state_dict = checkpoint["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile'd state dict
    cleaned = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "")
        cleaned[clean_key] = v
    model.load_state_dict(cleaned)
    model.eval()

    return model, tokenizer, config


@torch.no_grad()
def generate(model, tokenizer, prefix_text, max_new_tokens=200, temperature=1.0, top_k=50):
    """Autoregressive generation from a text prefix."""
    device = next(model.parameters()).device

    # Encode prefix
    bos = tokenizer.get_bos_token_id()
    token_ids = tokenizer.encode(prefix_text, prepend=bos)
    if isinstance(token_ids, list) and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Truncate to max sequence length if needed
        if tokens.size(1) >= MAX_SEQ_LEN:
            break

        logits = model(tokens)  # (1, T, vocab)
        logits = logits[:, -1, :]  # last position

        if temperature > 0:
            logits = logits / temperature
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode output, skipping BOS token
    output_ids = tokens[0, 1:].tolist()
    return tokenizer.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text continuations from a trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate per sample")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0], help="Sampling temperatures")
    parser.add_argument("--samples-per-prefix", type=int, default=5, help="Number of samples per prefix per temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (separate from training seed)")
    args = parser.parse_args()

    model, tokenizer, _ = load_model_and_tokenizer(args.checkpoint)

    results = {
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "temperatures": args.temperatures,
        "samples_per_prefix": args.samples_per_prefix,
        "samples": [],
    }

    total = len(PREFIXES) * len(args.temperatures) * args.samples_per_prefix
    count = 0

    for prefix in PREFIXES:
        for temp in args.temperatures:
            for sample_idx in range(args.samples_per_prefix):
                # Set sampling seed for reproducibility
                if args.seed is not None:
                    prefix_hash = int(hashlib.sha256(prefix.encode()).hexdigest()[:8], 16)
                    torch.manual_seed(args.seed * 10000 + prefix_hash % 10000 + sample_idx * 100 + int(temp * 10))

                text = generate(model, tokenizer, prefix, args.max_new_tokens, temp, args.top_k)
                results["samples"].append({
                    "prefix": prefix,
                    "temperature": temp,
                    "sample_idx": sample_idx,
                    "text": text,
                })
                count += 1
                print(f"\r  Sampling: {count}/{total}", end="", flush=True)

    print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results['samples'])} samples to {args.output}")


if __name__ == "__main__":
    main()
