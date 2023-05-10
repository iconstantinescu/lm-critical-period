import pandas as pd
from collections import defaultdict
import os
import re


def preprocess_file(filename):

    with open(f'./data/text/{filename}') as input:

        doc = []
        paragraph = []
        for line in input:
            if line.strip():
                paragraph.append(line.strip())
            else:
                if paragraph:
                    doc.append(" ".join(paragraph))
                paragraph = []


        text = os.linesep.join(doc)
        text = re.sub(r"(\u00a0)+", " ", text)  # remove non-breaking spaces
        text = re.sub(r"\*\s*", "", text)  # remove * delimiters
        text = re.sub(r"\x20\x20+", " ", text)  # remove extra spaces

        return text

filepath = './metadata/metadata.csv'
metadata_df = pd.read_csv(filepath)

languages = {'en', 'de', 'fi'}
files_dict = defaultdict(set)

for index, row in metadata_df.iterrows():
    file_lang = row['language'].split('\'')[1]
    no_file_type = pd.isna(row['type'])

    if file_lang in languages and no_file_type:
        files_dict[file_lang].add(row['id'])

for lang in languages:
    print(f'Processing files for the {lang} language ...')
    count = 0

    if not os.path.exists('./data/preprocessed'):
        os.makedirs('./data/preprocessed')

    out_file = open(f'./data/preprocessed/{lang}.txt', "w")

    for file in os.listdir('./data/text'):
        file_id = file.split('_')[0]
        if file_id in files_dict[lang]:
            text = preprocess_file(file)
            out_file.write(text)
            count += 1

    out_file.close()
    print(f'Processed {count} files for the {lang} language\n')

