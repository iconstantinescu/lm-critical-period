import argparse
import json
import os


def fix_tokenizer(path):
    added_tokens = {}

    if os.path.exists(f'{path}/added_tokens.json'):
        f = open(f'{path}/added_tokens.json')
        added_tokens = json.load(f)

        os.remove(f'{path}/added_tokens.json')
        print('Removed added_tokens.json')

    if os.path.exists(f'{path}/vocab.json'):
        f = open(f'{path}/vocab.json', 'r')
        vocab = json.load(f)

        for (key, value) in added_tokens.items():
            vocab.update({key:value})
            print(f'Added {key}:{value} to the vocabulary')

        out = open(f'{path}/vocab.json', 'w')
        json.dump(vocab, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="the path to the tokenizer to be fixed")
    args = parser.parse_args()

    fix_tokenizer(args.path)
