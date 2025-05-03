import re
from typing import List

TOKEN_SPLITTER_REGEXP = r'([,.:;?_!"()\']|--|\s)'
PUNCTUATIONS_REGEXP = r'\s+([,.!?"()\'])'
TOK_UNKNOWN = '|<unk>|'
TOK_EOT = '|<endoftext>|'


class SimpleTokenizer:

    def __init__(self, vocab: list[str]):
        vocab.extend([TOK_UNKNOWN, TOK_EOT])
        self.int_to_token = {i: token for i, token in enumerate(vocab)}
        self.token_to_int = {t: i for i, t in enumerate(vocab)}

    # text to list of integer encoding
    def encode(self, text: str) -> list[int]:
        # preprocessing
        tokens = re.split(TOKEN_SPLITTER_REGEXP, text)
        preprocessed = []

        for token in tokens:
            stripped_token = token.strip()
            if stripped_token:
                preprocessed.append(stripped_token)

        encodings = []

        for token in preprocessed:
            encoding = self.token_to_int.get(token)
            if encoding is not None:
                encodings.append(encoding)
            else:
                print(encoding)
                print(self.token_to_int.get(token))
                encodings.append(self.token_to_int.get(TOK_UNKNOWN))

        return encodings

    # integer encodings to text
    def decode(self, encodings: list[int]) -> str:
        text = ' '.join([self.int_to_token[enc] for enc in encodings])
        # Replace spaces before the specified punctuations
        text = re.sub(PUNCTUATIONS_REGEXP, r'\1', text)
        return text


def build_vocabulary(raw_text: str) -> list[str]:
    all_tokens = re.split(TOKEN_SPLITTER_REGEXP, raw_text)
    preprocessed = [token.strip() for token in all_tokens if token.strip()]
    return sorted(list(set(preprocessed)))
