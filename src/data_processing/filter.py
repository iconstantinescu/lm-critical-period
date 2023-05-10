import argparse
import os

import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector


MODELS_DICT = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm',
    'fi': 'fi_core_news_sm'
}


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


def clean_data(folder, filename):

    spacy.prefer_gpu()

    language = filename[0:2]
    nlp = spacy.load(MODELS_DICT[language],
                     disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'morphologizer', 'ner', 'lemmatizer'])

    nlp.max_length = 8000000
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('language_detector', last=True)

    print(f'Loaded spacy {MODELS_DICT[language]} model...')
    print(nlp.pipe_names, '\n')

    with open(f'./data/{folder}/{filename}.txt') as input:
        lines = input.readlines()

        num_words0 = 0
        num_words1 = 0
        num_sents0 = 0
        num_sents1 = 0
        num_lines0 = 0
        num_lines1 = 0

        clean_text = []

        for index, line in enumerate(lines):
            doc = nlp(line)

            num_lines0 += 1
            num_sents0 += len(list(doc.sents)) - 1
            num_words0 += len(doc)

            if doc._.language["language"] == language:
                clean_text.append(doc.text)

                num_lines1 += 1
                num_words1 += len(doc)
                num_sents1 += len(list(doc.sents)) - 1

        if not os.path.exists(f'./data/{folder}_clean'):
            os.makedirs(f'./data/{folder}_clean')

        out_file = open(f'./data/{folder}_clean/{filename}.txt', "w")
        out_file.write(''.join(clean_text))
        out_file.close()

        print(f'Initial number of lines: {num_lines0}')
        print(f'Initial number of sentences: {num_sents0}')
        print(f'Initial number of words: {num_words0}')

        print(f'Number of lines after cleaning: {num_lines1}')
        print(f'Number of sentences after cleaning: {num_sents1}')
        print(f'Number of words after cleaning: {num_words1}')

    print('\nFinished!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="the name of the dataset/folder to be processed")
    parser.add_argument("filename", type=str, help="the name of the file (should start with language prefix)")
    args = parser.parse_args()

    print(f'\nAnalyzing and cleaning the data from {args.dataset}_{args.filename}\n')
    clean_data(args.dataset, args.filename)
