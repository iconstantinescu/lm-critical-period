import pandas as pd
import argparse
import fnmatch
import os


def get_files(model_type, lang):
    pattern = f'evaluate_blimp_{model_type}-*-{lang}*.out'

    matched_files = []
    for name in os.listdir('./logs/evaluations'):
        if fnmatch.fnmatch(name, pattern):
            matched_files.append(name)

    return matched_files


def extract_results(files, model_type, lang):
    results_dict = {}

    for file in files:
        split = file[15:].split('-')
        name = f'{split[1]}-{split[3]}'
        results_dict[name] = {}

        with open(f'./logs/evaluations/{file}') as f:
            lines = f.readlines()

            assert lines[-18] == "Scores:\n"
            results = lines[-17:]

            for res in results:
                task = res.split(':')[0]
                score = float(res.split(':')[1].strip()[:-1])

                results_dict[name][task] = score

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'./plots/blimp_{model_type}_{lang}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-l", "--lang", type=str)
    args = parser.parse_args()

    files = get_files(args.model_type, args.lang)
    extract_results(files, args.model_type, args.lang)
