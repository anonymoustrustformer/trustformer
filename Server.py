# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:01:47 2024

@author: aliab
"""
import copy
from collections import OrderedDict

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:33:00 2024

@author: aliab
"""
import SGXEmulator as sgx
import pickle
import torch
import Dataset as ds
import torch.optim as optim
import torch.nn as nn
import torch
import os
from Transformer import Transformer
from transformers import AdamW
from sklearn.cluster import KMeans
import numpy as np

# Serialize and encrypt transformer model
# serialized_model = pickle.dumps(transformer_model)


class Server:
    def __init__(self, config, name='server', provisioning_key=b'Sixteen byte key'):
        self.key = provisioning_key
        self.name = name
        self.config = config
        self.server_model = Transformer(src_vocab_size=self.config['src_vocab_size'],
                                        tgt_vocab_size=self.config['tgt_vocab_size']
                                        , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                                        , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                                        , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                                        , transformer_id=100)
        # self.optimizer = optim.SGD(self.server_model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
        #                             weight_decay=config['weight_decay']),
        # self.optimizer =optim.Adam(self.server_model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-8)
        self.optimizer = AdamW(self.server_model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
        if config["model_type"] == "16bit":
            self.server_model.half()
        # self.server_model.to(self.config['device'])

    def aggregate_optimizers(self, optimizers):
        for param_group in optimizers[0].param_groups:
            for param in param_group['params']:
                exp_avgs = []
                exp_avg_sqs = []

                for opt in optimizers:
                    # if param in opt.state:
                    exp_avgs.append(opt.state[param]['exp_avg'])
                    exp_avg_sqs.append(opt.state[param]['exp_avg_sq'])

                # Only proceed if there are collected states to aggregate
                if exp_avgs:
                    exp_avg_agg = torch.mean(torch.stack(exp_avgs), dim=0)
                    for opt in optimizers:
                        # if param in opt.state:
                        opt.state[param]['exp_avg'].copy_(exp_avg_agg)

                if exp_avg_sqs:
                    exp_avg_sq_agg = torch.mean(torch.stack(exp_avg_sqs), dim=0)
                    for opt in optimizers:
                        # if param in opt.state:
                        opt.state[param]['exp_avg_sq'].copy_(exp_avg_sq_agg)

        return optimizers

    # def calculate_optimizer_state_difference(self, optimizer1, optimizer2):
    #     state_diffs = {}
    #     for group in optimizer1.param_groups:
    #         for p in group['params']:
    #             param_state1 = optimizer1.state[p]
    #             param_state2 = optimizer2.state[p]
    #             for key in param_state1:
    #                 if key in param_state2:  # Ensure both states contain the key
    #                     # Calculate element-wise difference
    #                     state_diffs[(p, key)] = param_state1[key] - param_state2[key]
    #     return state_diffs
    def calculate_optimizer_state_difference(self, optimizer1, optimizer2):
        state_diffs = {}
        for group1, group2 in zip(optimizer1.param_groups, optimizer2.param_groups):
            for p1, p2 in zip(group1['params'], group2['params']):
                param_state1 = optimizer1.state.get(p1, {})
                param_state2 = optimizer2.state.get(p2, {})

                # Iterate over keys in both param_state1 and param_state2
                all_keys = set(param_state1.keys()).union(set(param_state2.keys()))
                for key in all_keys:
                    if key in param_state1 and key in param_state2:
                        # Calculate element-wise difference
                        state_diffs[(p1, key)] = param_state1[key] - param_state2[key]
                    # elif key in param_state1:
                    #     # If key is missing in param_state2, assume zero tensor with the same shape
                    #     state_diffs[(p1, key)] = param_state1[key] - torch.zeros_like(param_state1[key])
                    # elif key in param_state2:
                    #     # If key is missing in param_state1, assume zero tensor with the same shape
                    #     state_diffs[(p2, key)] = torch.zeros_like(param_state2[key]) - param_state2[key]

        return state_diffs

    def aggregate_optimizer_states(self, global_optimizer, client_optimizers):
        num_clients = len(client_optimizers)

        # Initialize a dictionary to accumulate state differences
        accumulated_state_diffs = {}

        # Sum state differences from each client optimizer to the global optimizer
        for client_optimizer in client_optimizers:
            state_diffs = self.calculate_optimizer_state_difference(global_optimizer, client_optimizer[1])
            for key, diff in state_diffs.items():
                if key not in accumulated_state_diffs:
                    accumulated_state_diffs[key] = torch.zeros_like(diff)
                accumulated_state_diffs[key] += diff

        # Average the state differences
        averaged_state_diffs = {key: diff / num_clients for key, diff in accumulated_state_diffs.items()}

        # Update the global optimizer's states by subtracting the average differences
        # with torch.no_grad():
        #     for group in global_optimizer.param_groups:
        #         for p in group['params']:
        #             for key in averaged_state_diffs:
        #                 if key[0] == p and key[1] in global_optimizer.state[p]:
        #                     global_optimizer.state[p][key[1]] -= averaged_state_diffs[key]
        with torch.no_grad():
            for group in global_optimizer.param_groups:
                for p in group['params']:
                    if p in averaged_state_diffs:
                        for key in averaged_state_diffs[p]:
                            if key in global_optimizer.state[p]:
                                global_optimizer.state[p][key[1]] -= averaged_state_diffs[p][key[1]]

        return global_optimizer

    def average_centroids(self, client_centroids):
        aggregated_centroids = {}
        num_clients = self.config['n_workers'] #len(client_centroids)

        # Initialize aggregated centroids with zeros
        for name in client_centroids[0].keys():
            aggregated_centroids[name] = np.zeros_like(client_centroids[0][name])

        # Sum centroids from all clients
        for centroids in client_centroids:
            for name, centroid in centroids.items():
                aggregated_centroids[name] += centroid

        # Average centroids
        for name in aggregated_centroids.keys():
            aggregated_centroids[name] /= num_clients

        return aggregated_centroids


    def aggregate_models(self, transformers, agg_method='fedAvg', workers=None):

        loaded_transformers = []
        for transformer in transformers:
            if agg_method in ['DP-fedAvg', 'fedAvg','DP-fedSAM', 'DP-BLUR-LUS']:
                loaded_transformers.append(sgx.load_plain_model(file_path=transformer))
            else:
                loaded_transformers.append(sgx.load_model(file_path=transformer, key=self.key))
        # for worker in workers:
        #     loaded_transformers.append(worker.load_model(filepath=f'{worker.name}_model.pth'))

        if agg_method == 'fedAvg':
            # model1_state_dict = loaded_transformers[0].state_dict()  # State dict of model1
            # model2_state_dict = loaded_transformers[1].state_dict()  # State dict of model2
            # model3_state_dict = loaded_transformers[2].state_dict()  # State dict of model3
            #
            # # List of all model parameters
            # all_model_parameters = [model1_state_dict, model2_state_dict, model3_state_dict]

            # Averaging the parameters
            averaged_parameters = self.average_parameters2(loaded_transformers)
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    if name in averaged_parameters:
                        param.data.copy_(averaged_parameters[name])
            file_path = self.store_plain_model(name='server')
            return {"model": self, "file_path": file_path}
            # self.server_model.load_state_dict(averaged_parameters)
        if agg_method == 'Avg3':
            # model1_state_dict = loaded_transformers[0].state_dict()  # State dict of model1
            # model2_state_dict = loaded_transformers[1].state_dict()  # State dict of model2
            # model3_state_dict = loaded_transformers[2].state_dict()  # State dict of model3
            #
            # # List of all model parameters
            # all_model_parameters = [model1_state_dict, model2_state_dict, model3_state_dict]

            # Averaging the parameters
            # Aggregate centroids from all clients
            global_centroids = self.average_centroids(loaded_transformers)
            file_path = self.encrypt_store_model(name='server')
            return global_centroids

            # self.server_model.load_state_dict(averaged_parameters)
        if agg_method == 'DP-fedAvg':
            averaged_parameters = self.average_parameters2(loaded_transformers)
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    if name in averaged_parameters:
                        param.data.copy_(averaged_parameters[name])
            file_path = self.store_plain_model(name='server')
            return {"model": self, "file_path": file_path}
        if agg_method == 'DP-fedSAM':
            averaged_parameters = self.average_parameters2(loaded_transformers)
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    if name in averaged_parameters:
                        param.data.copy_(averaged_parameters[name])
            file_path = self.store_plain_model(name='server')
            return {"model": self, "file_path": file_path}
        if agg_method == 'DP-BLUR-LUS':
            averaged_parameters = self.average_parametersblur(loaded_transformers)
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    if name in averaged_parameters:
                        param.data.add_(averaged_parameters[name])
            file_path = self.store_plain_model(name='server')
            return {"model": self, "file_path": file_path}

    def apply_weight_sharing(self, model, num_clusters=8):
        centroids_dict = {}
        for name, param in model.named_parameters():
            if len(param.shape) > 1:  # Only apply to weight matrices
                weight = param.data.cpu().numpy().flatten()
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(weight.reshape(-1, 1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_]
                param.data = torch.tensor(new_weight.reshape(param.shape)).to(param.device)
                centroids_dict[name] = kmeans.cluster_centers_.flatten()
        return centroids_dict
    def average_parameters2(self, model_parameters):
        """
        Averages the parameters of multiple models.

        Args:
            model_parameters (list of OrderedDict): Each OrderedDict contains the state_dict of a model.

        Returns:
            OrderedDict containing the averaged parameters.
        """
        # Initialize the dictionary to store the average parameters
        average_params = None

        # Number of models
        num_models = len(model_parameters)

        for params in model_parameters:
            if average_params is None:
                # Initialize average_params with the same structure as model parameters but filled with zeros
                average_params = {name: torch.zeros_like(value) for name, value in params.items()}

            # Accumulate all model parameters
            for name, value in params.items():
                average_params[name] += value

        # Divide by the number of models to compute the mean
        for name in average_params.keys():
            average_params[name] /= num_models

        return average_params
    def average_parametersblur(self, model_parameters):
        """
        Averages the parameters of multiple models.

        Args:
            model_parameters (list of OrderedDict): Each OrderedDict contains the state_dict of a model.

        Returns:
            OrderedDict containing the averaged parameters.
        """
        average_params = OrderedDict()

        # Flatten the model_parameters list
        flattened_parameters = [param for sublist in model_parameters for param in sublist]

        # Create a dictionary to count the occurrences of each parameter name
        param_count = {}

        # Accumulate all model parameters
        for param_name, tensor in flattened_parameters:
            if param_name not in average_params:
                # Initialize with the same structure as model parameters but filled with zeros
                average_params[param_name] = torch.zeros_like(tensor)
                param_count[param_name] = 0

            average_params[param_name] += tensor
            param_count[param_name] += 1

        # Divide by the count to compute the mean
        for name in average_params.keys():
            average_params[name] /= param_count[name]

        return average_params

    def aggregate_DPFL(self, model_parameters):
        # Initialize the dictionary to store the average parameters
        average_params = None

        # Number of models
        num_models = len(model_parameters)

        for params in model_parameters:
            if average_params is None:
                # Initialize average_params with the same structure as model parameters but filled with zeros
                average_params = {name: torch.zeros_like(value) for name, value in params.items()}

            # Accumulate all model parameters
            for name, value in params.items():
                average_params[name] += value

        return average_params
    def extract_attention_params(self, model):
        """Extract attention weights and biases from a model."""
        attention_weights = []
        attention_biases = []

        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                if "weight" in name:
                    attention_weights.append(param.data.clone())
                elif "bias" in name:
                    attention_biases.append(param.data.clone())

        return attention_weights, attention_biases

    def average_parameters(self, *params):
        """Average a list of parameters."""
        stacked = torch.stack(params)
        return torch.mean(stacked, dim=0)

    def copy_model_params_and_buffers(self, src_model, dst_model):
        # Copy parameters
        for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
            dst_param.data.copy_(src_param.data)

        # # Copy buffers
        # for src_key, src_buffer in src_model.named_buffers():
        #     dst_buffer = dst_model._buffers[src_key]
        #     dst_buffer.copy_(src_buffer)

    def set_new_weights(self, average):
        for avg, server in zip(average.parameters(), self.server_model.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # subtract all transformer parameters
                server.data.copy_(server.data - avg.data)
        # nn.utils.clip_grad_norm_(average.parameters(), max_norm=1.0)

    def check_inf_in_model(model):
        inf_parameters = {}
        for name, param in model.named_parameters():
            if torch.isinf(param.data).any():
                inf_parameters[name] = torch.isinf(param.data).nonzero(as_tuple=True)
        return inf_parameters

    # Example usage in a training loop
    def blank_transformer(self):
        placeholder = Transformer(src_vocab_size=self.config['src_vocab_size'],
                                  tgt_vocab_size=self.config['tgt_vocab_size']
                                  , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                                  , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                                  , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                                  , transformer_id=100)
        # placeholder.to(self.config['device'])
        # placeholder.half()
        # for param in placeholder.parameters():
        #     param.data.fill_(0)
        return placeholder

    def average_differences(self, differences):
        placeholder = self.blank_transformer()
        for difference in differences:
            for holder, param in zip(placeholder.parameters(), difference.parameters()):
                # Check if the parameter is trainable.
                if param.requires_grad:
                    # sum all transformer parameters
                    holder.data.copy_(param.data + holder.data)

        for holder in placeholder.parameters():
            # Check if the parameter is trainable.
            if holder.requires_grad:
                # Average the parameter values
                holder.data.copy_(holder.data / len(differences))
        return placeholder

    def get_model(self):
        return self.server_model.state_dict()

    def load_decrypt_model(self, file_path):
        self.server_model, self.optimizer = sgx.load_model(file_path=file_path, key=self.key)
        return self
    def load_plain_model(self, file_path):
        self.server_model, self.optimizer = sgx.load_plain_model(file_path=file_path)
        return self
    # def encrypt_store_model(self):
    #     print('SGX encrypting and storing the transformer of : ', self.name)
    #     directory = "sealed_models"
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)        
    #     file_path = "sealed_models/{}".format(self.name)
    #     sgx.store_model(self.key, self.server_model, filename=file_path)
    #     return file_path

    def evaluate_model(self):
        self.server_model.eval()
        with torch.autograd.no_grad():  # torch.no_grad():
            source_ids_validation = torch.tensor(self.dataset['source_ids_validation'],
                                                 device=self.config['device'])  # .to()
            target_ids_validation = torch.tensor(self.dataset['target_ids_validation'],
                                                 device=self.config['device'])  # .to(self.config['device'])
            # Forward pass
            val_output = self.server_model(source_ids_validation, target_ids_validation)
            # Compute the loss
            val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                      target_ids_validation.contiguous().view(-1))

            # Print validation loss
            print(f"Validation Loss: {val_loss.item()}")

            generated_tokens = torch.argmax(val_output, dim=-1)

            # Convert token IDs to actual tokens using your vocabulary
            # Convert token IDs to actual tokens using the BART tokenizer
            generated_texts = ds.decode_tokens(generated_tokens)
            del source_ids_validation, target_ids_validation
        return generated_texts

    def encrypt_store_model(self, name):
        print('SGX encrypting and storing the transformer of : ', name)
        directory = "sealed_models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = "sealed_models/{}".format(name)
        sgx.store_model(self.key, self.server_model.state_dict(), filename=file_path)
        return file_path
    def store_plain_model(self, name):
        print('Storing the plain transformer of : ', name)
        directory = "unsealed_models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = f"unsealed_models/{name}.pkl"
        sgx.store_plain_model(self.server_model.state_dict(), filename=file_path)
        return file_path
    def calculate_difference(self, loaded_transformer):
        output_t = self.blank_transformer()

        # output_t.to(self.config['device'])

        for server, param, output in zip(self.server_model.parameters(), loaded_transformer.parameters(),
                                         output_t.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # sum all transformer parameters
                output.data.copy_(server.data - param.data)
        return output_t

    def calculate_distance(self, model1, model2):
        distances = {}
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
                # Ensure that the parameters correspond to the same layer and element
                if name1 == name2 and param2.requires_grad:
                    # Compute the Euclidean distance between the two parameters
                    distance = param1.data - param2.data
                    distances[name1] = distance
        return distances

    def aggregate_models2(self, global_model, client_models):
        num_clients = len(client_models)

        # Initialize a dictionary to accumulate distances
        # accumulated_distances = {name: torch.tensor(0.0, device=param.device) for name, param in
        #                          global_model.named_parameters()}
        accumulated_diffs = {name: torch.zeros_like(param.data) for name, param in global_model.named_parameters()}

        # Sum distances from each client model to the global model
        for client_model in client_models:
            differences = self.calculate_distance(global_model, client_model[0])
            for name, diff in differences.items():
                accumulated_diffs[name] += diff

        # Average the distances
        # averaged_distances = {name: distance / num_clients for name, distance in accumulated_distances.items()}
        averaged_diffs = {name: diff / num_clients for name, diff in accumulated_diffs.items()}

        # Subtract these averaged distances from the global model's parameters
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in averaged_diffs:
                    param -= averaged_diffs[name]

        return global_model
