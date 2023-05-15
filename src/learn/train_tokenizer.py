from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import AutoTokenizer

import argparse
import json

sample = {
    "en": "Hello, y'all! How are you 😁 ?",
    "de": "",
    "fi": "",
}


def train_tokenizer(model, dataset, lang):

    tokenizer = ByteLevelBPETokenizer()

    files = [f"./data/{dataset}/{lang}/raw/{split}.txt" for split in ["test", "train", "validation"]]

    tokenizer.train(files=files, vocab_size=32000, min_frequency=2,
                    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.save_model(f"./data/{dataset}/{lang}")

    # create a custom config.json file for AutoTokenizer
    with open(f"./data/{dataset}/{lang}/config.json", 'w') as config:
        json.dump({
            'model_type': model
        }, config)

    # Load the tokenizer and output a sample tokenization
    test_tokenizer = AutoTokenizer.from_pretrained(f"./data/{dataset}/{lang}")
    output = test_tokenizer.encode_plus(sample[lang])
    print(output.tokens())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the tokenizer configuration to use")
    parser.add_argument("dataset", type=str, help="the dataset to be processed")
    parser.add_argument("language", type=str, help="the language of the files")
    args = parser.parse_args()

    print(f'\nTraining a custom {args.model} tokenizer for data {args.dataset}_{args.language}\n')
    train_tokenizer(args.model, args.dataset, args.language)
