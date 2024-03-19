import random
import math
import os
import itertools as IT


random.seed(42)

# OUTPUT_FOLDER = 'unified_clean'
OUTPUT_FOLDER = 'unified_extended'

# DATASETS = ['gutenberg', 'open_subtitles', 'wikipedia']
DATASETS = ['open_subtitles', 'wikipedia']


# SAMPLE_RATIOS = {
#     'en': [0.035, 0.117, 0.048],
#     'de': [1, 1, 0.160],
#     'fi': [1, 1, 1]
# }

SAMPLE_RATIOS = {
    'ar': [0.202, 0.172],
    'de': [0.188, 0.040],
    'el': [0.055, 0.394],
    'en': [0.019, 0.016],
    'es': [0.035, 0.052],
    'fi': [0.157, 0.322],
    'ko': [1, 0.564],
    'nl': [0.067, 0.133],
    'pl': [0.032, 0.125],
    'ru': [0.154, 0.054],
    'tr': [0.047, 0.383],
    'java': [0, 0.134]  # added java.txt to wikipedia dir for simplicity
}

BLOCK_SIZE = 10000
SPLIT_RATIOS = [0.83, 0.085, 0.085]


DOC = {}
for key in SAMPLE_RATIOS.keys():
    DOC[key] = []


def train_test_split(doc, lang):
    train_split_idx = math.floor(len(doc) * SPLIT_RATIOS[0])
    valid_split_idx = math.floor(len(doc) * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    train = doc[:train_split_idx]
    valid = doc[train_split_idx:valid_split_idx]
    test = doc[valid_split_idx:]

    if not os.path.exists(f'./data/{OUTPUT_FOLDER}/{lang}/raw'):
        os.makedirs(f'./data/{OUTPUT_FOLDER}/{lang}/raw')

    full_out = open(f'./data/{OUTPUT_FOLDER}/{lang}/raw/full.txt', "w")
    train_out = open(f'./data/{OUTPUT_FOLDER}/{lang}/raw/train.txt', "w")
    valid_out = open(f'./data/{OUTPUT_FOLDER}/{lang}/raw/validation.txt', "w")
    test_out = open(f'./data/{OUTPUT_FOLDER}/{lang}/raw/test.txt', "w")

    full_out.write(''.join(doc))
    train_out.write(''.join(train))
    valid_out.write(''.join(valid))
    test_out.write(''.join(test))

    full_out.close()
    train_out.close()
    valid_out.close()
    test_out.close()


for lang, ratios in SAMPLE_RATIOS.items():
    print('\n')

    for idx, dataset in enumerate(DATASETS):

        # count the number of lines in the document
        with open(f'./data/{dataset}/{lang}.txt') as f:
            num_lines = sum(1 for _ in f)

        # the number of blocks
        num_blocks = num_lines // BLOCK_SIZE

        # the number of lines we need to sample
        num_samples = int(ratios[idx] * num_lines)

        # the blocks we sample (by index)
        sampled_blocks = random.sample(range(0, num_blocks), num_samples // BLOCK_SIZE)

        with open(f'./data/{dataset}/{lang}.txt') as f:
            block_idx = 0
            samples = []

            while True:
                next_line_block = list(IT.islice(f, BLOCK_SIZE))

                if not next_line_block:
                    break

                elif block_idx in sampled_blocks or len(next_line_block) < BLOCK_SIZE:
                    samples.append(''.join(next_line_block))

                block_idx += 1

            print(dataset, lang, len(samples))
            DOC[lang].extend(samples)

    random.shuffle(DOC[lang])  # each entry in the doc is a block, so we shuffle by blocks
    train_test_split(DOC[lang], lang)


# Create the interleaved datasets
def evenly_spaced(*iterables):
    return [item[1] for item in
            sorted(IT.chain.from_iterable(
                zip(IT.count(start=1.0 / (len(seq) + 1),
                             step=1.0 / (len(seq) + 1)), seq)
                for seq in iterables))]


pairs = [(l1, 'en') for l1 in SAMPLE_RATIOS.keys()]
print(pairs)

for pair in pairs:
    lang1 = pair[0]
    lang2 = pair[1]

    a = DOC[lang1]
    b = DOC[lang2]
    interleaved = evenly_spaced(a, b)

    train_test_split(interleaved, f'{lang1}_{lang2}')


print('\nFinished successfully!\n')
