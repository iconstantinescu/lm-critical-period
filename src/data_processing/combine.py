import random
import math
import os

DATASETS = ['gutenberg', 'open_subtitles', 'wikipedia']
SAMPLE_RATIOS = {
    'en': [0.031, 0.092, 0.042],
    'de': [1, 1, 0.114],
    'fi': [1, 1, 1]
}

SPLIT_RATIOS = [0.83, 0.085, 0.085]

random.seed(42)

for lang, ratios in SAMPLE_RATIOS.items():

    doc = []
    for idx, dataset in enumerate(DATASETS):
        with open(f'./data/{dataset}/{lang}.txt') as input:
            lines = [line.strip() for line in input if random.random() <= ratios[idx]]
            print(dataset, lang, len(lines))
            doc.extend(lines)

    random.shuffle(doc)

    train_split_idx = math.floor(len(doc) * SPLIT_RATIOS[0])
    valid_split_idx = math.floor(len(doc) * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    train = doc[:train_split_idx]
    valid = doc[train_split_idx:valid_split_idx]
    test = doc[valid_split_idx:]

    if not os.path.exists(f'./data/unified/{lang}/raw'):
        os.makedirs(f'./data/unified/{lang}/raw')

    train_out = open(f'./data/unified/{lang}/raw/train.txt', "w")
    valid_out = open(f'./data/unified/{lang}/raw/validation.txt', "w")
    test_out = open(f'./data/unified/{lang}/raw/test.txt', "w")

    train_out.write('\n'.join(train))
    test_out.write('\n'.join(valid))
    valid_out.write('\n'.join(test))

    train_out.close()
    valid_out.close()
    test_out.close()

    print('\n')

print('Finished successfully!')

