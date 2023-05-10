import argparse
import os
import re
import time

from bs4 import BeautifulSoup


def extract_text(language):
    out_file = open(os.path.join(f"preprocessed_data/{language}.txt"), "w")
    dir = os.path.join(language)

    print('Start processing the data...')
    start_time = time.time()

    for d1 in os.listdir(dir):
        print(d1)

        for d2 in os.listdir(os.path.join(dir, d1)):
            for f in os.listdir(os.path.join(dir, d1, d2)):
                with open(os.path.join(dir, d1, d2, f)) as input:
                    xml_string = input.read()
                    soup = BeautifulSoup(xml_string, "xml")

                    # remove metadata text
                    meta_tag = soup.meta
                    meta_tag.decompose()

                    text = soup.get_text()
                    text = re.sub(r"- ", "", text)  # remove dialogue line
                    text = re.sub(r"(\u266A)+", "", text)  # remove song character
                    text = re.sub(r"[\[(](.*)[\])]", "", text)  # remove [] ()

                    text = re.sub(r"\x20\x20+", " ", text)  # remove extra spaces
                    text = re.sub(r"\x0a\x20+", "", text)  # remove leading spaces
                    text = re.sub(r"(\u00a0)+", " ", text)  # remove non-breaking spaces

                    text = os.linesep.join(
                        [
                            line.strip() for line in text.splitlines()
                            if line
                        ]
                    )

                    out_file.write(text)


    end_time = time.time()
    print('Elapsed time during the processing in seconds:', end_time - start_time)

    print('Saving the data...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="the language folder to be processed")
    args = parser.parse_args()

    print(f'\nExtracting the data for {args.language}\n')
    extract_text(args.language)
