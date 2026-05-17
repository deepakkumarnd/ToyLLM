"""
WordPiece Tokenizer from scratch using HuggingFace `tokenizers` library.

Pipeline:
  Normalizer -> PreTokenizer -> WordPiece Model (trained) -> Decoder

WordPiece (used by BERT) differs from BPE in its merge strategy: it picks the
pair that maximises likelihood of the training data rather than raw frequency.
Continuation subwords are prefixed with "##" (e.g. "playing" → ["play", "##ing"]).
"""

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFC, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
UNK_TOKEN = "[UNK]"


def build_tokenizer() -> Tokenizer:
    """Assemble all pipeline components around an empty WordPiece model."""
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))

    # NFC keeps precomposed forms intact (preserves Malayalam matras/virama).
    # Lowercase handles Latin; no StripAccents so diacritics are never removed.
    tokenizer.normalizer = Sequence([NFC(), Lowercase()])

    # WordPiece pre-tokenizes on whitespace: each whitespace-delimited word is
    # then split into subwords independently, with "##" marking continuations.
    tokenizer.pre_tokenizer = Whitespace()

    # Decoder strips "##" prefixes and rejoins subwords into words.
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return tokenizer


def train(tokenizer: Tokenizer, files: list[str], vocab_size: int = 10000) -> Tokenizer:
    """Train the WordPiece model on the given text files."""
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
        show_progress=True,
    )
    tokenizer.train(files, trainer)
    return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Train a WordPiece tokenizer on a text corpus and save it to JSON.",
        epilog=(
            "Examples:\n"
            "  # single file\n"
            "  python -m hf.build_tokenizer -o tok.json corpus.txt\n\n"
            "  # multiple files\n"
            "  python -m hf.build_tokenizer -o tok.json file1.txt file2.txt\n\n"
            "  # directory (all *.txt recursively)\n"
            "  python -m hf.build_tokenizer -o tok.json ./data/\n\n"
            "  # mix of files and directories\n"
            "  python -m hf.build_tokenizer -o tok.json corpus.txt ./extra-data/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "corpus",
        nargs="+",
        help="One or more .txt files or directories to train on. "
             "Directories are searched recursively for *.txt files.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the trained tokenizer JSON (required).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        metavar="N",
        help="Vocabulary size (default: 10000).",
    )
    args = parser.parse_args()

    files = []
    for entry in args.corpus:
        p = Path(entry)
        if p.is_dir():
            files.extend(str(f) for f in sorted(p.rglob("*.txt")))
        elif p.is_file():
            files.append(str(p))
        else:
            parser.error(f"Not a file or directory: {entry}")

    if not files:
        parser.error("No .txt files found in the provided corpus paths.")

    print(f"=== Corpus: {len(files)} file(s) ===")
    for f in files:
        print(f"  {f}")

    print("\n=== Building tokenizer ===")
    tok = build_tokenizer()

    print(f"\n=== Training (vocab_size={args.vocab_size}) ===")
    tok = train(tok, files, vocab_size=args.vocab_size)

    vocab = tok.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    samples = [
        "Harry Potter and the Sorcerer's Stone",
        "Mr. and Mrs. Dursley, of number four, Privet Drive,",
        "He couldn't see how it could end well.",
        "മസ്സാജ് പാർലറുകളുടെ ബോർഡുകൾ എല്ലായിടത്തും കാണാം.",
        "സ്കൂബാഡൈവിങ്ങ്, സ്നോർക്കേലിങ്ങ്, സ്പീഡ്ബോട്ടിങ്ങ്, സർഫിങ്ങ് തുടങ്ങിയ വിവിധ ജലകായക വിനോദങ്ങൾക്കായി ഈ പ്രദേശം പ്രശസ്തമാണ്.",
    ]

    print("\n=== Encode / Decode ===")
    for text in samples:
        ids = encode(tok, text)
        recovered = decode(tok, ids)
        tokens = [tok.id_to_token(i) for i in ids]
        print(f"\nInput   : {text!r}")
        print(f"Token IDs ({len(ids)}): {ids}")
        print(f"Tokens  : {tokens}")
        print(f"Decoded : {recovered!r}")

    print("\n=== Special tokens ===")
    for st in SPECIAL_TOKENS:
        print(f"  {st!r:10s} -> id {tok.token_to_id(st)}")

    tok.save(args.output)
    print(f"\nSaved to {args.output}")

    reloaded = Tokenizer.from_file(args.output)
    ids2 = encode(reloaded, samples[0])
    print(f"Reloaded encode matches: {ids2 == encode(tok, samples[0])}")
