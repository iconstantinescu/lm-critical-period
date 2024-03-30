import argparse

from datasets import load_dataset
import os
import regex


def get_data(lang):
    ds = load_dataset("bigcode/the-stack", streaming=True, data_dir=f"data/{lang}", split="train")

    i = 0
    code_lines = []
    sample_size = 10**6

    for sample in iter(ds.take(sample_size)):
        code_string = sample['content']

        code_string = regex.sub(r"//(.?)+", "", code_string)  # remove  "//" comments
        code_string = regex.sub(r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/", "", code_string)  # remove "/* */" comments

        code_lines.extend([line.strip() for line in code_string.splitlines() if line.strip()])

        if i % (sample_size / 10) == 0:
            print(i // (sample_size // 10))

        i += 1

    out = open(f'./data/the_stack/{lang}.txt', "w")
    out.write(os.linesep.join(code_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="the coding language to be processed")
    args = parser.parse_args()

    print(f'\nGetting the code data for {args.language}\n')
    get_data(args.language)
