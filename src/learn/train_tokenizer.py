from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import RobertaProcessing

import argparse

sample = {
    "en": "Hello, y'all! How are you 😁 ?",
    "de": "",
    "fi": "",
}


def train_tokenizer(dataset, lang):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    files = [f"./data/{dataset}/{lang}/raw/{split}.txt" for split in ["test", "train", "validation"]]

    trainer = BpeTrainer(vocab_size=32000,
                         min_frequency=2,
                         special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files, trainer)

    tokenizer.post_processor = RobertaProcessing(
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    )
    tokenizer.enable_truncation(max_length=512)

    tokenizer.save(f"./data/{dataset}/{lang}/tokenizer-wiki.json")

    # Load the tokenizer and output a sample tokenization
    test_tokenizer = Tokenizer.from_file(f"./data/{dataset}/{lang}/tokenizer-wiki.json")
    output = test_tokenizer.encode(sample[lang])
    print(output.tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="the dataset to be processed")
    parser.add_argument("language", type=str, help="the language of the files")
    args = parser.parse_args()

    print(f'\nTraining a custom tokenizer for data {args.dataset}_{args.language}\n')
    train_tokenizer(args.dataset, args.language)
