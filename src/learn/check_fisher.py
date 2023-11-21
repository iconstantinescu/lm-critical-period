import argparse
import json
import torch
import numpy as np
from transformers import AutoModel


def load_fim(checkpoint):
    fname_fim = f'./checkpoints/{checkpoint}/eval_results.json'
    with open(fname_fim, 'r') as f:
        print('Loading file')
        eval_results = json.load(f)

    fisher_information_matrix = eval_results['fisher_information_matrix']
    fisher_information_matrix = [torch.FloatTensor(x) for x in fisher_information_matrix]

    return fisher_information_matrix

def load_model(checkpoint):
    print('Loading model')
    #model = AutoModel.from_pretrained('gpt2')
    model = AutoModel.from_pretrained(f'./checkpoints/{checkpoint}')
    return model


def check_fim(fisher_information_matrix, model):
    fim_count = 0
    for x in fisher_information_matrix:
        # print(x.shape)
        fim_count += x.numel()
    print(f"Fisher Information Matrix total parameters: {fim_count}")

    model_count = 0
    for x in list(model.parameters()):
        # print(x.shape)
        model_count += x.numel()
    print(f"Model total parameters: {model_count}")

    loss = 0
    for param, fim_weight in zip(model.parameters(), fisher_information_matrix):
        loss += (fim_weight * (param ** 2)).sum()

    print(f"Loss: {loss.item()}")

    f_min = torch.tensor(np.inf)
    f_max = torch.tensor(-np.inf)
    f_mean = 0
    for x in fisher_information_matrix:
        # print(torch.min(x))
        f_min = torch.min(f_min, torch.min(x))
        f_max = torch.max(f_max, torch.max(x))
        f_mean += x.sum()

    print(f"Minimum value: {f_min}")
    print(f"Maximum value: {f_max}")
    print(f"Mean value: {f_mean / fim_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="The checkpoint containing the Fisher matrix")
    args = parser.parse_args()

    fisher_information_matrix = load_fim(args.checkpoint)
    model = load_model(args.checkpoint)
    check_fim(fisher_information_matrix, model)
