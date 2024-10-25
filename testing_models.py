from Transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from datasets import load_dataset
# from transformers import AutoTokenizer
import gc
import Dataset as ds
from Worker import Worker
import Utils
import Federated_training
from Server import Server

config = Utils.load_config("config.json")

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)

server = Server(config=config)

# server.load_decrypt_model(file_path='sealed_models/server')

optimizer = optim.Adam(server.get_model().parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  # optimizer
criterion = nn.CrossEntropyLoss(ignore_index=1)
dataset = ds.tokenized_dataset(name='wmt19', n_records_train=config['data_in_each_worker'],
                               n_records_test=config['test_in_each_worker'], max_seq_length=config['max_seq_length'],
                               train_offset=0, test_offset=0)

# base_seal_folder = "sealed_models"
paths = ["sealed_models/worker0", "sealed_models/worker1", "sealed_models/worker2"]
####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)
main_server = Federated_training.initialize_server(config)


# paths = Federated_training.seal_store_models(workers)
# Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
# Federated_training.load_unseal_models(paths, workers)  # this function is equal to receieve model in client side

Federated_training.train_workers(workers)

server = main_server.aggregate_models(paths, agg_method='Avg')  # FL aggregation happens here
new_optimizer = main_server.aggregate_optimizers([worker.optimizer for worker in workers])
# Federated_training.setup_optimizers(new_optimizer, workers=workers, optimizer_name='adam')

workers[0].set_model(server['model'].get_model())
workers[1].set_model(server['model'].get_model())
workers[2].set_model(server['model'].get_model())

Federated_training.train_workers(workers)
# paths = Federated_training.seal_store_models(workers[0:1])

# workers = Federated_training.load_unseal_models(paths,workers)
# workers[0].get_model().train()


# Federated_training.train_workers(workers[0:1])

# paths=paths[0:1]
workers[0].get_model().to(config['device'])
workers[0].get_model().eval()
with torch.autograd.no_grad():  # torch.no_grad():
    source_ids_validation = torch.tensor(dataset['source_ids_validation']).to(config['device'])
    target_ids_validation = torch.tensor(dataset['target_ids_validation']).to(config['device'])
    # Forward pass
    val_output = workers[0].get_model()(source_ids_validation, target_ids_validation)

    # Compute the loss
    val_loss = criterion(val_output.contiguous().view(-1, config['tgt_vocab_size']),
                         target_ids_validation.contiguous().view(-1))
    # Print validation loss
    print(f"Validation Loss: {val_loss.item()}")
    generated_tokens = torch.argmax(val_output, dim=-1)

    # Convert token IDs to actual tokens using your vocabulary
    # Convert token IDs to actual tokens using the BART tokenizer
    generated_texts = ds.decode_tokens(generated_tokens)

