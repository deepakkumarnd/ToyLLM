import argparse
from tokenizer import default_tokenizer, name_tokenizer


def test_file(args):
    tok = default_tokenizer()
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                print()
                continue
            print(tok.encode(line).tokens)


def test_name(args):
    tok = name_tokenizer()
    print(tok.encode(args.name).tokens)


def main():
    parser = argparse.ArgumentParser(description="Test the WordPiece tokenizers.")
    sub = parser.add_subparsers(dest="command", required=True)

    file_cmd = sub.add_parser("file", help="Tokenize a text file line by line.")
    file_cmd.add_argument("file", help="Path to the text file.")
    file_cmd.set_defaults(func=test_file)

    name_cmd = sub.add_parser("name", help="Tokenize a single name.")
    name_cmd.add_argument("name", help="The name to tokenize.")
    name_cmd.set_defaults(func=test_name)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
