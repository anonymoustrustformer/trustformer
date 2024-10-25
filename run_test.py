# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:48:15 2024

@author: aliab
"""
import copy
import time

import Utils
import Federated_training
import torch
import torch.optim as optim
import Plotting


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed
set_seed(42)

config = Utils.load_config("config.json")

# base_seal_folder = "sealed_models"
paths1 = ["sealed_models/worker0", "sealed_models/worker1", "sealed_models/worker2"]
paths2 = ["worker0_model.pth", "worker1_model.pth", "worker2_model.pth"]

####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)

main_server = Federated_training.initialize_server(config)

optimizers = []
for worker in workers:
    optimizers.append(
        optim.Adam(worker.transformer.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-4))

paths = ["sealed_models/worker0", "sealed_models/worker1", "sealed_models/worker2"]

# workers = Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side
aggregation_time = 0

method = config['method']
if method == "trustformer":
    paths = ["sealed_models/worker0", "sealed_models/worker1", "sealed_models/worker2"]
    for epoch in range(config['n_epochs']):
        print("start training for epoch {}".format(epoch))
        Federated_training.train_workers(workers, optimizers)
        paths1 = Federated_training.seal_store_models(workers, sparsification_enabled=True)
        if (epoch + 1) % config['aggregation_frequency'] == 0:
            start_agg = time.time()
            # server = main_server.aggregate_models(paths, agg_method='Avg2',workers=workers)  # FL aggregation happens here
            global_centroids = main_server.aggregate_models(paths1, agg_method='Avg3',
                                                            workers=workers)  # FL aggregation happens here
            end_agg = time.time()
            Federated_training.adjust_workers_weights(workers, global_centroids=global_centroids)
            aggregation_time += end_agg - start_agg
elif method == "fedAvg":
    paths = ["unsealed_models/worker0.pkl", "unsealed_models/worker1.pkl", "unsealed_models/worker2.pkl"]
    for epoch in range(config['n_epochs']):
        print("start training for epoch {}".format(epoch))
        Federated_training.train_workers(workers, optimizers)
        paths1 = Federated_training.seal_store_models(workers, sparsification_enabled=False)
        if (epoch + 1) % config['aggregation_frequency'] == 0:
            start_agg = time.time()
            server = main_server.aggregate_models(paths, agg_method='fedAvg', workers=workers)
            end_agg = time.time()
            Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
            workers = Federated_training.load_models(paths, workers)  # this function is equal to receieve model in client side
            aggregation_time += end_agg - start_agg
elif method == "DP-fedAvg":
    paths = ["unsealed_models/worker0.pkl", "unsealed_models/worker1.pkl", "unsealed_models/worker2.pkl"]
    for epoch in range(config['n_epochs']):
        print("start training for epoch {}".format(epoch))
        Federated_training.train_workersDPfedAvg(workers, optimizers)
        paths1 = Federated_training.seal_store_models(workers, sparsification_enabled=False)
        if (epoch + 1) % config['aggregation_frequency'] == 0:
            start_agg = time.time()
            server = main_server.aggregate_models(paths, agg_method='DP-fedAvg', workers=workers)
            end_agg = time.time()
            Federated_training.send_global_model_to_clients(config,
                                                            server=server['model'])  # send aggregated model to clients
            workers = Federated_training.load_models(paths,
                                                     workers)  # this function is equal to receieve model in client side
            aggregation_time += end_agg - start_agg
elif method == 'DP-fedSAM':
    paths = ["unsealed_models/worker0.pkl", "unsealed_models/worker1.pkl", "unsealed_models/worker2.pkl"]
    main_server = Federated_training.initialize_server(config)
    server = {}
    server['model'] = main_server
    for epoch in range(config['n_epochs']):
        print("start training for epoch {}".format(epoch))
        Federated_training.train_workersDPfedSAM(workers, optimizers)
        paths1 = Federated_training.seal_store_models(workers, sparsification_enabled=False)
        if (epoch + 1) % config['aggregation_frequency'] == 0:
            start_agg = time.time()
            server = main_server.aggregate_models(paths, agg_method='DP-fedSAM',
                                                  workers=workers)  # FL aggregation happens here
            # global_centroids = main_server.aggregate_models(paths1, agg_method='Avg3',workers=workers)  # FL aggregation happens here
            end_agg = time.time()
            Federated_training.send_global_model_to_clients(config,
                                                            server=server['model'])  # send aggregated model to clients
            workers = Federated_training.load_models(paths,
                                                     workers)  # this function is equal to receieve model in client side
            aggregation_time += end_agg - start_agg
        # workers = Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side
elif method == 'DP-BLUR-LUS':
    paths = ["unsealed_models/worker0.pkl", "unsealed_models/worker1.pkl", "unsealed_models/worker2.pkl"]
    main_server = Federated_training.initialize_server(config)
    server = {}
    server['model'] = main_server
    for epoch in range(config['n_epochs']):
        print("start training for epoch {}".format(epoch))
        worker_updates = Federated_training.train_workersDPBLUR(workers, optimizers)
        paths1 = Federated_training.seal_store_model_updates(worker_updates, paths=paths)
        if (epoch + 1) % config['aggregation_frequency'] == 0:
            start_agg = time.time()
            server = main_server.aggregate_models(paths, agg_method='DP-BLUR-LUS', workers=workers)
            end_agg = time.time()
            Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
            workers = Federated_training.load_models(paths, workers)  # this function is equal to receieve model in client side
            aggregation_time += end_agg - start_agg
print("aggregation time", aggregation_time)

results = Federated_training.evaluate_workers(workers, store=True, logging=False)

average_loss = torch.mean(torch.tensor([torch.mean(torch.tensor(results[i]['loss'])) for i in range(3)]))
loss_on_worker = [torch.mean(torch.tensor(results[i]['loss'])) for i in range(3)]
print("Avg loss: ", average_loss)


Federated_training.store_worker_info(workers=workers, epoch="n_clusters=100")
