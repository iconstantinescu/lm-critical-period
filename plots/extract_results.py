import pandas as pd
import argparse
import fnmatch
import os
import json

glue_tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte", "boolq", "multirc", "wsc"]


def get_files(eval_type, model_type, lang, do_checkpoints):
    matched_files = []
    path = f'./logs/evaluations/{model_type}/{eval_type}'

    if do_checkpoints:
        for dir in os.listdir(path):
            dir_path = os.path.join(path, dir)
            if os.path.isdir(dir_path) and f'-{lang}en-' in dir:
                pattern = f'checkpoint-*.out'
                for name in os.listdir(dir_path):
                    if fnmatch.fnmatch(name, pattern):
                        matched_files.append(os.path.join(dir_path, name))
    else:
        pattern = f'evaluate_{eval_type}_{model_type}-*-{lang}*.out'
        for name in os.listdir(path):
            if fnmatch.fnmatch(name, pattern):
                matched_files.append(os.path.join(path, name))

    return matched_files


def extract_blimp_results(files, model_type, lang, do_checkpoints):
    results_dict = {}

    # we extract the results directly from the output logs
    for file in sorted(files):
        split = file.split('-')
        name = f'c{split[1][-1]}-{split[3]}'

        if do_checkpoints:
            name += f'-{split[-1].split("_")[0]}'

        results_dict[name] = {}

        with open(file) as f:
            lines = f.readlines()

            assert lines[-18] == "Scores:\n"
            results = lines[-17:]

            for res in results:
                task = res.split(':')[0]
                score = float(res.split(':')[1].strip()[:-1])

                results_dict[name][task] = score

    results_df = pd.DataFrame(results_dict)

    name = f'blimp_{model_type}_{lang}'
    if do_checkpoints:
        name += '_checkpoints'

    results_df.to_csv(f'./plots/{name}.csv', index_label='task')


def extract_glue_results(files, model_type, lang):
    results_dict = {}

    # we use the 'files' variable to find which models were evaluated
    # we extract the results from the evaluation files added to the model checkpoints

    for file in sorted(files):
        checkpoint = file.split('_')[2]
        checkpoint_dir = os.path.join(f"checkpoints/{checkpoint}/finetune")

        split = file.split('-')
        name = f'c{split[1][-1]}-{split[3]}'
        results_dict[name] = {}

        for task in glue_tasks:
            result_file = os.path.join(checkpoint_dir, f'{task}/eval_results.json')

            try:
                with open(result_file) as f:
                    data = json.load(f)
                    results_dict[name][task] = round(data['eval_accuracy']*100, 2)
            except FileNotFoundError:
                results_dict[name][task] = None

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'./plots/glue_{model_type}_{lang}.csv', index_label='task')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_type", type=str)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-l", "--lang", type=str)
    parser.add_argument("-c", "--checkpoints", action='store_true', default=False,
                        help="Extract results from the checkpoints evaluation")
    args = parser.parse_args()

    files = get_files(args.eval_type, args.model_type, args.lang, args.checkpoints)

    if args.eval_type == 'blimp':
        extract_blimp_results(files, args.model_type, args.lang, args.checkpoints)
    elif args.eval_type == 'glue':
        extract_glue_results(files, args.model_type, args.lang)
