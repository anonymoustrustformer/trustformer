# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:43:33 2024

@author: aliab
"""
from Transformer import Transformer
import Dataset as ds
from Worker import Worker
from torch import autograd
import torch
from opacus.accountants.utils import get_noise_multiplier

import json

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename):
    with open(filename, 'r') as f:
        # Read the entire contents of the file
        content = f.read()
        # Replace single quotes with double quotes
        content = content.replace("'", '"')
        # Load the JSON data
        config = json.loads(content)
    return config


def compute_fisher_diag(config, model, dataset):
    device = config['device']

    model.eval()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    for data_source, data_target in zip(dataset['source_ids'],dataset['target_ids']):
        src_data = torch.tensor(
            dataset['source_ids'][
            0: config['batch_size']],
            device=config['device'],
            dtype=torch.long)  # Ensure correct dtype to match model expectations

        tgt_data = torch.tensor(
            dataset['target_ids'][
            0: config['batch_size']],
            device=config['device'], dtype=torch.long)

        # Calculate output log probabilities
        log_probs = torch.nn.functional.log_softmax(model(src_data,tgt_data), dim=-1)
        for batch_idx in range(tgt_data.size(0)):  # Iterate over batches
            for seq_idx in range(tgt_data.size(1)):  # Iterate over sequence length
                label = tgt_data[batch_idx, seq_idx]  # Get the target label at this position
                log_prob = log_probs[batch_idx, seq_idx, label]

        # for i, label in enumerate(tgt_data):
        #     log_prob = log_probs[i, label]

            # Calculate first-order derivatives (gradients)
                model.zero_grad()
                grad1 = autograd.grad(log_prob, model.parameters(), create_graph=False, retain_graph=True)

            # Update Fisher diagonal elements
                for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                    fisher_diag_value.add_(grad_value.detach() ** 2)

            # Free up memory by removing computation graph
                del log_prob, grad1, label
                torch.cuda.empty_cache()
        # Release CUDA memory
        del log_probs
        torch.cuda.empty_cache()

    # Calculate the mean value
    num_samples = config["data_in_each_worker"]
    # fisher_diag = [fisher_tensor.cpu() for fisher_tensor in fisher_diag]
    fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]
    torch.cuda.empty_cache()

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
        normalized_fisher_diag.append(normalized_fisher_value)
    del fisher_diag
    torch.cuda.empty_cache()
    return normalized_fisher_diag

def compute_noise_multiplier(config):
    target_epsilon = config['epsilon']
    target_delta = config['noise_delta']
    client_data_sizes = config['data_in_each_worker']
    global_epoch = config['n_epochs']
    local_epoch = config['aggregation_frequency']
    batch_size = config['batch_size']
    total_dataset_size = client_data_sizes * config['n_workers']
    user_sample_rate = 1
    sample_rate = batch_size / total_dataset_size * user_sample_rate
    total_steps = user_sample_rate * (sum([global_epoch * local_epoch * (client_data_sizes / batch_size)]))

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp"
    )

    return noise_multiplier
