from typing import List

from tokenizer import SimpleTokenizer, build_vocabulary
import re

if __name__ == '__main__':
    with open("hp1.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Content length: ", len(raw_text))

    # all_tokens = re.split(TOKEN_SPLITTER_REGEXP, raw_text)
    #
    # preprocessed = [token.strip() for token in all_tokens if token.strip()]
    #
    # print("Number of tokens: ", len(preprocessed))
    #
    # vocab = sorted(list(set(preprocessed)))
    #
    # print("Number of unique tokens: ", len(vocab))
    #
    # print(vocab[:1000])

    vocab = build_vocabulary(raw_text)

    tokenizer = SimpleTokenizer(vocab)

    print(raw_text[:100])
    sample_text = raw_text[:100]
    encodings = tokenizer.encode(sample_text)
    print("Encoding: \n", encodings)
    text = tokenizer.decode(encodings)
    print("Decoded text: \n", text)




