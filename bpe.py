# BYTE Pair Encoding using tiktoken library

from importlib.metadata import version
import tiktoken


print("Tiktoken version", version('tiktoken'))

# GPT2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Harry Potter and the Sorcerer's Stone\n"
    "CHAPTER ONE<|endoftext|>"
    "THE BOY WHO LIVED <|endoftext|>"
    "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say <|endoftext|>"
)

encodings = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print("BPE encoding:\n", encodings)
[print(tokenizer.decode_single_token_bytes(token)) for token in encodings]
print("Number of tokens in encoding:", len(encodings))
print("BPE decoding:\n", tokenizer.decode(encodings))
