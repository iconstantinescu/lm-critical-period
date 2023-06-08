import random
import math
import os
from itertools import islice

DATASETS = ['gutenberg', 'open_subtitles', 'wikipedia']
SAMPLE_RATIOS = {
    'en': [0.036, 0.118, 0.049],
    'de': [1, 1, 0.160],
    'fi': [1, 1, 1]
}

BLOCK_SIZE = 10000

SPLIT_RATIOS = [0.83, 0.085, 0.085]

random.seed(42)

for lang, ratios in SAMPLE_RATIOS.items():
    print('\n')

    doc = []
    for idx, dataset in enumerate(DATASETS):

        # count the number of lines in the document
        with open(f'./data/{dataset}_clean/{lang}.txt') as f:
            num_lines = sum(1 for _ in f)

        # the number of blocks
        num_blocks = num_lines // BLOCK_SIZE

        # the number of lines we need to sample
        num_samples = int(ratios[idx] * num_lines)

        # the blocks we sample (by index)
        sampled_blocks = random.sample(range(0, num_blocks), num_samples // BLOCK_SIZE)

        with open(f'./data/{dataset}_clean/{lang}.txt') as f:
            block_idx = 0
            samples = []

            while True:
                next_line_block = list(islice(f, BLOCK_SIZE))

                if not next_line_block:
                    break

                elif block_idx in sampled_blocks or len(next_line_block) < BLOCK_SIZE:
                    samples.append(''.join(next_line_block))

                block_idx += 1

            print(dataset, lang, len(samples))
            doc.extend(samples)

    train_split_idx = math.floor(len(doc) * SPLIT_RATIOS[0])
    valid_split_idx = math.floor(len(doc) * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    # each entry in the doc is a block, so we shuffle by blocks
    random.shuffle(doc)

    train = doc[:train_split_idx]
    valid = doc[train_split_idx:valid_split_idx]
    test = doc[valid_split_idx:]

    if not os.path.exists(f'./data/unified_clean/{lang}/raw'):
        os.makedirs(f'./data/unified_clean/{lang}/raw')

    full_out = open(f'./data/unified_clean/{lang}/raw/full.txt', "w")
    train_out = open(f'./data/unified_clean/{lang}/raw/train.txt', "w")
    valid_out = open(f'./data/unified_clean/{lang}/raw/validation.txt', "w")
    test_out = open(f'./data/unified_clean/{lang}/raw/test.txt', "w")

    full_out.write(''.join(doc))
    train_out.write(''.join(train))
    valid_out.write(''.join(valid))
    test_out.write(''.join(test))

    full_out.close()
    train_out.close()
    valid_out.close()
    test_out.close()

print('\nFinished successfully!\n')

