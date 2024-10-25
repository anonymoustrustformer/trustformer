# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:45:23 2024

@author: aliab
"""
from Transformer import Transformer
import Dataset as ds
from Worker import Worker
import numpy as np
import Utils
import os
import json
import pickle
import torch
import gc
import SGXEmulator as sgx

config = Utils.load_config("config.json")
from Server import Server


def train_workers(workers, optimizers):
    print("start training workers...")
    for worker,optimizer in zip(workers,optimizers):
        worker.start_training(optimizer)
def train_workersDPfedAvg(workers, optimizers):
    print("start training workers...")
    for worker,optimizer in zip(workers,optimizers):
        worker.start_trainingDPfedAvg(optimizer)
        gc.collect()
        torch.cuda.empty_cache()
def train_workersDPfedSAM(workers, optimizers):
    print("start training workers...")
    for worker,optimizer in zip(workers,optimizers):
        worker.start_trainingDPfedSAM(optimizer)
        gc.collect()
        torch.cuda.empty_cache()
def train_workersDPBLUR(workers, optimizers):
    print("start training workers...")
    worker_updates = []
    for worker,optimizer in zip(workers,optimizers):
        worker_updates.append(worker.start_trainingDPBlurLus(optimizer))
        gc.collect()
        torch.cuda.empty_cache()
    return  worker_updates
def adjust_workers_weights(workers, global_centroids):

    for worker in workers:
        print(f"setting up worker {worker.name} weights...")
        worker.adjust_local_model_with_centroids(global_centroids= global_centroids, num_clusters_ratio=config['n_cluster_ratio'])
def evaluate_workers(workers, store=True, logging=True):
    print("start evaluating workers...")
    results = []
    for worker in workers:
        temp_results, loss, metrics = worker.evaluate_model(logging=logging)
        if store is True:
            file_path = config['results']+f'/evaluation_{worker.name}.pkl'
            with open(file_path, 'wb') as file:
                pickle.dump({'worker': worker.name,'results': temp_results, 'loss': loss, 'metrics': metrics}, file)
        results.append({'worker': worker.name,'results': temp_results, 'loss': loss, 'metrics': metrics})
    return results

def seal_store_models(workers, sparsification_enabled=True):
    path = []
    print("Sealing and storing workers models...")
    for worker in workers:
        path.append(worker.encrypt_store_model(sparsification_enabled=sparsification_enabled))
    return path

def seal_store_model_updates(worker_udpates, paths = ['unsealed_models/worker0']):
    path = []
    print("Sealing and storing workers models...")
    for update, path in zip(worker_udpates, paths):
        sgx.store_plain_model(update, filename=path)
    return path

def load_unseal_models(paths, workers):
    new_workers = []
    print("receiving info for workers...")
    for path, worker in zip(paths, workers):
        new_workers.append(worker.load_decrypt_model(path))

    return new_workers

def load_models(paths, workers):
    new_workers = []
    print("receiving info for workers...")
    for path3, worker in zip(paths, workers):
        new_workers.append(worker.load_plain_model(path3))
    return new_workers
def setup_optimizers(optimizer, workers, optimizer_name='sgd'):
    for i in range(len(workers)):
        workers[i].set_optimizer(optimizer[i], optimizer_name=optimizer_name)


def initialize_server(config):
    return Server(config=config)


def adjust_parameters(transformer, operator="bigger"):
    parameters = []
    # Print the names and shapes of the model's parameters
    for name, param in transformer.named_parameters():
        param.requires_grad = False
        # print(f"Name: {name}")
        # print(f"Params: {param}")
        parameters.append(param.data.cpu().detach().numpy())

        len(parameters)
    stds = []
    for params in parameters:
        stds.append(np.average(np.std(params, axis=0)))
    avgStd = np.average(stds)
    if operator == "bigger":
        paramsToFineTune = stds > avgStd
    elif operator == "lower":
        paramsToFineTune = stds < avgStd
    named_parameters_iter = list(transformer.named_parameters())
    selected_parameters = []
    for select, (name, param) in zip(paramsToFineTune, named_parameters_iter):
        if select:
            param.requires_grad = True


def initialize_workers(config):
    workers = []

    for i in range(config['n_workers']):
        print("Initializing worker ", i)
        trans = Transformer(src_vocab_size=config['src_vocab_size'], tgt_vocab_size=config['tgt_vocab_size']
                            , d_model=config['d_model'], num_heads=config['num_heads']
                            , num_layers=config['num_layers'], d_ff=config['d_ff']
                            , max_seq_length=config['max_seq_length'], dropout=config['dropout']
                            , transformer_id=i)
        # trans.half()
        # data = ds.tokenized_dataset(name='wmt19', n_records_train=(i + 1) * config['data_in_each_worker'],
        #                             n_records_test=(i + 1) * config['test_in_each_worker'],
        #                             max_seq_length=config['max_seq_length'],
        #                             train_offset=i * config['data_in_each_worker'],
        #                             test_offset=i * config['test_in_each_worker'])
        data = ds.tokenized_dataset(name='wmt19', n_records_train=(i + 1) * config['data_in_each_worker'],
                                    n_records_test=config['test_in_each_worker'],
                                    max_seq_length=config['max_seq_length'],
                                    train_offset=i * config['data_in_each_worker'],
                                    test_offset=0)

        workers.append(Worker(data, config,
                              trans, name='worker' + str(i),
                              provisioning_key=b'Sixteen byte key'))

    return workers


def send_global_model_to_clients(config, server):
    n_clients = config['n_workers']
    for i in range(n_clients):
        name = f"worker{i}"
        if config['method'] == 'trustformer':
            server.encrypt_store_model(name=name)
        if config['method'] in ['fedAvg', 'DP-fedAvg', 'DP-fedSAM', 'DP-BLUR-LUS']:
            server.store_plain_model(name=name)



def store_worker_info(workers, epoch):
    results_folder = config['results']
    for worker in workers:
        results_folder = config['results'] + f'/{epoch}/{worker.name}'  # .format(worker.name)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Define the file path to save the data
        file_path = os.path.join(results_folder, f'{worker.name}.json')

        # Write the dictionary data to a JSON file
        with open(file_path, 'w') as file:
            json.dump(worker.history, file)
        print(f"history of worker {worker.name} epoch {epoch} saved to: {file_path}")


def save_models(workers):
    print("saving workers...")
    for worker in workers:
        worker.save_model(filepath=f'{worker.name}_model.pth')