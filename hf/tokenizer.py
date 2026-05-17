from pathlib import Path
from tokenizers import Tokenizer

_JSON = Path(__file__).parent / "hf_wordpiece_tokenizer.json"
_NAMES_JSON = Path(__file__).parent / "hf_indian_names_tokenizer.json"


def default_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(_JSON))


def name_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(_NAMES_JSON))
