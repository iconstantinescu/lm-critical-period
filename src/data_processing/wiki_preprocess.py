import os
import re
import argparse

from bs4 import BeautifulSoup


def preprocess(lang):
    out_file = open(os.path.join(f"preprocessed_data/{lang}wiki.txt"), "w")
    wiki_dir = os.path.join(f"text_{lang}")
    for d1 in os.listdir(wiki_dir):
        for f in os.listdir(os.path.join(wiki_dir, d1)):
            with open(os.path.join(wiki_dir, d1, f)) as input:
                xml_string = input.read()

                soup = BeautifulSoup(xml_string, 'html.parser')

                text = soup.get_text()

                text = re.sub(r"(\u00a0)+", " ", text)  # remove non-breaking spaces
                text = re.sub(r"\x20\x20+", " ", text)  # remove extra spaces

                text = os.linesep.join(
                    [
                        line.strip() for line in text.splitlines()
                        if line
                    ]
                )

                out_file.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="the language file to be processed")
    args = parser.parse_args()

    print(f'\nPreprocessing the data for {args.language}\n')
    preprocess(args.language)
