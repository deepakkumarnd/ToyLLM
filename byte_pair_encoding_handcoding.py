# this is a rudimentary implementation to grasp the idea of byte pair encoding
# here we start with some text and split the text using word limiter regexp to build
# an initial set of words.

import re
from tokenizer import TOKEN_SPLITTER_REGEXP

DEBUG = False


def print_debug(*args):
    if DEBUG:
        print(args)


def build_vocab(text):
    # cleanup and building character level tokens
    words = re.split(TOKEN_SPLITTER_REGEXP, text)
    words = list(filter(lambda x: len(x.strip()) > 0, words))
    vocab = sorted(list(set(''.join(words))))
    print_debug("Starting Vocab:", vocab)
    # count words and build token counter
    word_counter = {}

    for word in words:
        word_counter[word] = word_counter.get(word, 0) + 1

    # print("Word counter:", word_counter)

    # build tokens with count
    tokens_with_count = []

    # convert each word, count in dict to tuple of list of chars and count
    for word, count in word_counter.items():
        tokens_with_count.append((list(word), count))

    max_count = 1000
    count = 0

    while count < max_count:
        print_debug("Tokens with count", tokens_with_count)
        # build pairs of tokens and find the most repeating pair

        pair_counter = {}

        for tokens, count in tokens_with_count:
            pairs = zip(tokens, tokens[1:])

            for tok1, tok2 in pairs:
                pair = ''.join([tok1, tok2])
                pair_counter[pair] = pair_counter.get(pair, 0) + count

        if len(pair_counter) == 0:
            break

        print_debug("Pair counter", pair_counter)
        most_repeating_pair = max(pair_counter.items(), key=lambda x: x[1])

        print_debug("Most repeating pair", most_repeating_pair)
        most_repeating_pair = most_repeating_pair[0]
        vocab.append(most_repeating_pair)

        # merge the most repeating pair in keys of tokens wth count
        for tokens, count in tokens_with_count:
            i = 0
            while i < len(tokens) - 1:
                pair = ''.join([tokens[i], tokens[i + 1]])
                if pair == most_repeating_pair:
                    tokens[i] = pair
                    del tokens[i + 1]
                    i = i + 1

                i = i + 1

        print_debug("Vocab:", vocab)
        count = count + 1
        # repeat the process
    print("Final vocab using BPE", sorted(vocab))


sample_text = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5

with open("hp1.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
build_vocab(raw_text)
