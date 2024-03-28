import pandas as pd
import argparse
import fnmatch
import os
import json

glue_tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte", "boolq", "multirc", "wsc"]
checkpoints_dirname = 'checkpoints'


def get_files(eval_type, model_type, l1, l2, do_checkpoints):
    matched_files = []
    path = f'./logs/evaluations/{model_type}/{eval_type}'

    if do_checkpoints:
        for dir in os.listdir(path):
            dir_path = os.path.join(path, dir)
            if os.path.isdir(dir_path) and f'-{l1}{l2}-' in dir:
                pattern = f'checkpoint-*.out'
                for name in os.listdir(dir_path):
                    if fnmatch.fnmatch(name, pattern):
                        matched_files.append(os.path.join(dir_path, name))
    else:
        pattern = f'evaluate_{eval_type}_{model_type}-*-{l1}{l2}*.out'
        for name in os.listdir(path):
            if fnmatch.fnmatch(name, pattern):
                matched_files.append(os.path.join(path, name))

    return matched_files


def remap_checkpoints(results_dict):
    # sort and map checkpoints to epochs
    sorted_dict = dict(sorted(results_dict.items(),
                              key=lambda x: ('-'.join(x[0].split('-')[:-1]), int(x[0].split('-')[-1]))))

    print(sorted_dict.keys())
    new_dict = {}

    i = 1
    for (key, value) in sorted_dict.items():
        new_key = '-'.join(key.split('-')[:-1]) + f'-{i}'
        new_dict[new_key] = value

        i += 1
        if 'c4' in key and i == 9:
            i = 1
        elif i == 7:
            i = 1

    return new_dict


def create_name(checkpoint, do_checkpoints=False):
    split = checkpoint.split('-')
    config = split[1]
    condition = split[3]
    name = f'{config}-{condition}'

    if 'ewc' in checkpoint:
        name += '-ewc'

    if condition == 'interleaved':
        if len(split) > 6 and split[6] in {'2', '2/checkpoint'}:
            name += '2'

    if do_checkpoints:
        name += f'-{split[-1]}'

    return name


def extract_blimp_results(files, model_type, l1, l2, do_checkpoints):
    results_dict = {}

    # we extract the results directly from the output logs
    for file in sorted(files):
        checkpoint = file.split('_')[2]

        name = create_name(checkpoint, do_checkpoints)

        results_dict[name] = {}

        with open(file) as f:
            lines = f.readlines()

            assert lines[-18] == "Scores:\n"
            results = lines[-17:]

            for res in results:
                task = res.split(':')[0]
                score = float(res.split(':')[1].strip()[:-1])

                results_dict[name][task] = score

    name = f'blimp_{model_type}_{l1}{l2}'
    if do_checkpoints:
        name += '_checkpoints'
        results_dict = remap_checkpoints(results_dict)

    results_df = pd.DataFrame(results_dict)

    results_df.to_csv(f'./plots/{name}.csv', index_label='task')


def extract_glue_results(files, model_type, l1, l2):
    results_dict = {}

    # we use the 'files' variable to find which models were evaluated
    # we extract the results from the evaluation files added to the model checkpoints

    for file in sorted(files):
        checkpoint = file.split('_')[2]
        checkpoint_dir = os.path.join(f"{checkpoints_dirname}/{checkpoint}/finetune")

        name = create_name(checkpoint)

        results_dict[name] = {}

        for task in glue_tasks:
            result_file = os.path.join(checkpoint_dir, f'{task}/eval_results.json')

            try:
                f = open(result_file)
                data = json.load(f)
                results_dict[name][task] = round(data['eval_accuracy'] * 100, 2)
            except FileNotFoundError:
                results_dict[name][task] = None

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'./plots/glue_{model_type}_{l1}{l2}.csv', index_label='task')


def extract_l1_results(files, model_type, l1, l2, do_checkpoints):
    results_dict = {}

    # we use the 'files' variable to find which models were evaluated
    # we extract the results from the evaluation files added to the model checkpoints

    for file in sorted(files):
        checkpoint = file.split('_')[2]
        checkpoint_dir = os.path.join(f"{checkpoints_dirname}/{checkpoint}")

        name = create_name(checkpoint, do_checkpoints)

        results_dict[name] = {}
        result_file = os.path.join(checkpoint_dir, f'eval_results.json')
        try:
            f = open(result_file)
            data = json.load(f)
            results_dict[name] = [data['eval_loss']]
        except FileNotFoundError:
            results_dict[name] = None

    name = f'l1_{model_type}_{l1}{l2}'
    if do_checkpoints:
        name += '_checkpoints'
        results_dict = remap_checkpoints(results_dict)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'./plots/{name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_type", type=str)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-l1", "--lang1", type=str)
    parser.add_argument("-l2", "--lang2", type=str, default='en')
    parser.add_argument("-c", "--checkpoints", action='store_true', default=False,
                        help="Extract results from the checkpoints evaluation")
    args = parser.parse_args()

    files = get_files(args.eval_type, args.model_type, args.lang1, args.lang2, args.checkpoints)

    if args.eval_type == 'blimp':
        extract_blimp_results(files, args.model_type, args.lang1, args.lang2, args.checkpoints)
    elif args.eval_type == 'glue':
        extract_glue_results(files, args.model_type, args.lang1, args.lang2)
    elif args.eval_type == 'l1':
        extract_l1_results(files, args.model_type, args.lang1, args.lang2, args.checkpoints)
