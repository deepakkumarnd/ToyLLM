"""
Usage:
  python -m namelm.name_generator generate "Ra"
  python -m namelm.name_generator generate "Ra" --checkpoint models/name-8281k_*.pt
  python -m namelm.name_generator verify "Rahul Singh"
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hf.tokenizer import name_tokenizer
from namelm.config import LLMConfig
from namelm.model import NameModel

_MODELS_DIR = Path(__file__).parent.parent / "models"


def _latest_checkpoint() -> Path:
    checkpoints = sorted(_MODELS_DIR.glob("name-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {_MODELS_DIR}")
    print(f"Using latest checkpoint: {checkpoints[-1].name}")
    return checkpoints[-1]


def _load_model(checkpoint: Path, config: LLMConfig, device: torch.device) -> NameModel:
    model = NameModel(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    logits[logits < values[-1]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def _build_prefix_mask(tok, partial: str) -> set[int]:
    """Return token IDs whose text starts with `partial` (word-initial tokens only)."""
    return {
        tid
        for token, tid in tok.get_vocab().items()
        if not token.startswith("##") and token.startswith(partial)
    }


def _parse_pattern(pattern: str) -> tuple[str, str]:
    """Split pattern into (complete_words_text, partial_last_word).

    'Rahu'      -> ('',       'rahu')
    'Rahul S'   -> ('rahul',  's')
    'Rahul '    -> ('rahul',  '')   # trailing space = last word is complete
    """
    lower = pattern.lower()
    if lower.endswith(" "):
        return lower.strip(), ""
    parts = lower.split()
    if len(parts) <= 1:
        return "", parts[0] if parts else ""
    return " ".join(parts[:-1]), parts[-1]


def _generate_one(model, tok, pattern, sep_id, config, device, temperature, top_k) -> str:
    complete_text, partial = _parse_pattern(pattern)

    # Tokenize the fully completed words as the starting context.
    context = tok.encode(complete_text).ids if complete_text else []

    # Build constraint mask for the first generated token.
    constrained_ids = _build_prefix_mask(tok, partial) if partial else set()

    for _ in range(config.context_length * 3):
        input_ids = torch.tensor([context[-config.context_length:] or [0]], device=device)
        with torch.no_grad():
            logits = model(input_ids)[0, -1].clone()

        # Apply prefix constraint on the very first new token.
        if constrained_ids:
            mask = torch.full_like(logits, float("-inf"))
            for tid in constrained_ids:
                mask[tid] = logits[tid]
            logits = mask
            constrained_ids = set()   # constraint used — free-generate from here

        next_id = _sample_next(logits, temperature, top_k)
        if next_id == sep_id:
            break
        context.append(next_id)

    return tok.decode(context)


def cmd_generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LLMConfig()
    tok = name_tokenizer()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    model = _load_model(checkpoint, config, device)

    sep_id = tok.token_to_id("[SEP]")

    _, partial = _parse_pattern(args.pattern)
    matches = _build_prefix_mask(tok, partial) if partial else {"(any)"}
    print(f"Generating names matching '{args.pattern}' "
          f"({len(matches)} vocab token(s) satisfy the prefix):\n")

    for i in range(args.count):
        name = _generate_one(model, tok, args.pattern, sep_id, config, device,
                             args.temperature, args.top_k)
        print(f"  {i + 1}. {name.title()}")


def cmd_verify(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LLMConfig()
    tok = name_tokenizer()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    model = _load_model(checkpoint, config, device)

    sep_id = tok.token_to_id("[SEP]")
    ids = tok.encode(args.name.lower()).ids + [sep_id]

    if len(ids) < 2:
        print("Name too short to evaluate.")
        return

    total_log_prob = 0.0
    print(f"Token probabilities for '{args.name}':\n")

    for pos in range(len(ids) - 1):
        context = ids[max(0, pos - config.context_length + 1): pos + 1]
        input_ids = torch.tensor([context], device=device)
        with torch.no_grad():
            logits = model(input_ids)
        probs = F.softmax(logits[0, -1], dim=-1)
        target_id = ids[pos + 1]
        token_prob = probs[target_id].item()
        total_log_prob += math.log(token_prob + 1e-10)
        token_str = tok.id_to_token(target_id)
        print(f"  P({token_str!r:12s}) = {token_prob:.4f}")

    overall_prob = math.exp(total_log_prob)
    perplexity = math.exp(-total_log_prob / (len(ids) - 1))
    print(f"\n  Log-probability : {total_log_prob:.4f}")
    print(f"  Probability     : {overall_prob:.6e}")
    print(f"  Perplexity      : {perplexity:.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m namelm.name_generator",
        description=(
            "Indian name generator and verifier powered by NameModel.\n\n"
            "Two subcommands are available:\n"
            "  generate  — sample names that start with a given pattern\n"
            "  verify    — score how likely a name is according to the model"
        ),
        epilog=(
            "Examples:\n"
            "  # generate 5 names starting with 'Ra' (uses latest checkpoint)\n"
            "  python -m namelm.name_generator generate 'Ra'\n\n"
            "  # generate 10 more creative names starting with 'Pri'\n"
            "  python -m namelm.name_generator generate 'Pri' --count 10 --temperature 1.2\n\n"
            "  # check how natural a name sounds to the model\n"
            "  python -m namelm.name_generator verify 'Rahul Singh'\n\n"
            "  # use a specific checkpoint instead of the latest\n"
            "  python -m namelm.name_generator --checkpoint models/name-8.3m_20260517.pt verify 'Priya Nair'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a .pt model checkpoint. Defaults to the most recently saved file in models/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    gen = sub.add_parser(
        "generate",
        help="Generate names that start with a given pattern.",
        description=(
            "Autoregressively samples tokens from the model starting from the\n"
            "given pattern, stopping at a name-boundary token. Each run is\n"
            "stochastic — re-run to get different suggestions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sampling tips:\n"
            "  --temperature < 1.0  more predictable, common names\n"
            "  --temperature > 1.0  more varied, occasionally unusual names\n"
            "  --top-k 1            always picks the single most likely token (greedy)"
        ),
    )
    gen.add_argument("pattern", help="Starting prefix, e.g. 'Ra', 'Pri', 'Muh'.")
    gen.add_argument("--count", type=int, default=5, metavar="N",
                     help="Number of names to generate (default: 5).")
    gen.add_argument("--temperature", type=float, default=0.8, metavar="T",
                     help="Sampling temperature — higher = more random (default: 0.8).")
    gen.add_argument("--top-k", type=int, default=50, metavar="K",
                     help="Restrict sampling to the top-K most likely tokens at each step (default: 50).")
    gen.set_defaults(func=cmd_generate)

    # ── verify ────────────────────────────────────────────────────────────────
    ver = sub.add_parser(
        "verify",
        help="Print the model probability and perplexity of a given name.",
        description=(
            "Runs the name through the model and reports the probability the\n"
            "model assigns to each token, the overall log-probability, and the\n"
            "perplexity. Lower perplexity means the model finds the name more\n"
            "natural given its training data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Interpreting perplexity:\n"
            "  ~1       perfect (model has memorised this exact name)\n"
            "  2 – 5    very natural, common pattern in the training set\n"
            "  5 – 20   plausible but less common\n"
            "  > 20     unusual or out-of-distribution name"
        ),
    )
    ver.add_argument("name", help="Full name to evaluate, e.g. 'Rahul Singh'.")
    ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
