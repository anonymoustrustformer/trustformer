# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:33:00 2024

@author: aliab
"""
import copy
import time
import math
import numpy as np
import Utils
import SGXEmulator as sgx
import pickle
import torch
import Dataset as ds
import torch.optim as optim
import torch.nn as nn
import torch
import os
import torch
import gc
import Metrics
from transformers import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import KMeans


class Worker:
    def __init__(self, dataset, config, transformer, name='worker1', provisioning_key=b'Sixteen byte key'):
        self.dataset = dataset  # this is a dictionary of {"","","",""}
        self.key = provisioning_key
        self.name = name
        self.optimizer = optim.Adam(transformer.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)
        self.transformer = transformer
        self.history = {}
        self.centroids = {}
        self.membership = []
        self.config = config
        if config['model_type'] == "16bit":
            self.transformer.half()

    def start_training(self, optimizer):
        # Set the model to training mode and move it to the device
        if self.config['model_type'] == "16bit":
            self.transformer.half()
        self.transformer.train()
        self.transformer.to(self.config['device'])
        torch.cuda.empty_cache()
        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'training_time': [],
            'clustering_time': 0
        }
        scaler = GradScaler()

        # Loop through the dataset in batches
        start_time = time.time()
        test_time = 0
        print(f'Training {self.name}')
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            for batch in range(self.config['batch_epoch']):
                # self.optimizer.step()

                # optimizer.step()
                # Prepare batch data directly on the device, minimizing memory copy operations
                if self.config['model_type'] == "16bit":
                    src_data = torch.tensor(
                        self.dataset['source_ids'][
                        iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                        device=self.config['device'],
                        dtype=torch.long)  # Ensure correct dtype to match model expectations
                else:
                    src_data = torch.tensor(
                        self.dataset['source_ids'][
                        iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                        device=self.config['device'],
                        dtype=torch.long)  # Ensure correct dtype to match model expectations

                tgt_data = torch.tensor(
                    self.dataset['target_ids'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)

                # Zero out gradients
                # self.optimizer.zero_grad()
                optimizer.zero_grad()

                # Forward pass

                with autocast():
                    output = self.transformer(src_data, tgt_data)
                    loss = self.criterion(output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                          tgt_data.contiguous().view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Manually clear variables to ensure they are deleted from memory
                del src_data, tgt_data, output, loss
                torch.cuda.empty_cache()
                # Trigger garbage collection to reclaim memory`
                gc.collect()
            test_time_start = time.time()
            test_loss = self.test_model(False)
            self.history['iteration'].append(iteration + 1)
            self.history['loss'].append(test_loss)
            test_time_finish = time.time()
            test_time += test_time_finish - test_time_start
        print(f"first batch test Loss: {self.history['loss'][0]} of  {self.name}")
        finish_time = time.time()
        self.transformer.to('cpu')
        self.history['training_time'] = (finish_time - start_time) - test_time
        del scaler
        torch.cuda.empty_cache()

        return self.transformer

    def start_trainingDPfedAvg(self, optimizer):
        # Set the model to training mode and move it to the device
        if self.config['model_type'] == "16bit":
            self.transformer.half()
        self.transformer.train()
        self.transformer.to(self.config['device'])
        torch.cuda.empty_cache()
        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'training_time': [],
            'clustering_time': 0
        }
        scaler = GradScaler()

        # Loop through the dataset in batches
        start_time = time.time()
        test_time = 0
        print(f'Training {self.name}')
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            for batch in range(self.config['batch_epoch']):
                # self.optimizer.step()

                # optimizer.step()
                # Prepare batch data directly on the device, minimizing memory copy operations
                if self.config['model_type'] == "16bit":
                    src_data = torch.tensor(
                        self.dataset['source_ids'][
                        iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                        device=self.config['device'],
                        dtype=torch.long)  # Ensure correct dtype to match model expectations
                else:
                    src_data = torch.tensor(
                        self.dataset['source_ids'][
                        iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                        device=self.config['device'],
                        dtype=torch.long)  # Ensure correct dtype to match model expectations

                tgt_data = torch.tensor(
                    self.dataset['target_ids'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)

                # Zero out gradients
                # self.optimizer.zero_grad()
                optimizer.zero_grad()

                # Forward pass

                with autocast():
                    output = self.transformer(src_data, tgt_data)
                    loss = self.criterion(output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                          tgt_data.contiguous().view(-1))
                scaler.scale(loss).backward()
                self.clip_gradients(self.transformer.parameters(),
                                    self.config['clipping_bound'])  # Clipping is the same for BLUR_LUS and DP-FedSAM
                scaler.step(optimizer)
                scaler.update()
                # Manually clear variables to ensure they are deleted from memory
                del src_data, tgt_data, output, loss
                torch.cuda.empty_cache()
                # Trigger garbage collection to reclaim memory`
                gc.collect()
            test_time_start = time.time()
            test_loss = self.test_model(False)
            self.history['iteration'].append(iteration + 1)
            self.history['loss'].append(test_loss)
            test_time_finish = time.time()
            test_time += test_time_finish - test_time_start
        self.transformer.to('cpu')
        del scaler
        torch.cuda.empty_cache()
        noisy_updates = []

        noise_multiplier = Utils.compute_noise_multiplier(self.config)
        print(f'Test loss: {self.history["loss"][0]}')

        for name, sparse_update in self.transformer.named_parameters():
            # noise_stddev = 0.01
            noise_stddev = torch.sqrt(torch.tensor((self.config['clipping_bound'] ** 2) * (noise_multiplier ** 2) / (
                    self.config['data_in_each_worker'] * self.config['n_workers'])))
            noise = [torch.randn_like(param) * noise_stddev for param in sparse_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(sparse_update, noise)]
            noisy_updates.append((name, noisy_update))
        converted_tensors = []
        for name, layer_updates in noisy_updates:
            layer_tensor = torch.stack(layer_updates, dim=0)  # Modify dim as needed
            converted_tensors.append(layer_tensor)
        self.transformer.cpu()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for param, new_weights in zip(self.transformer.parameters(), converted_tensors):
                param.copy_(new_weights)
        finish_time = time.time()
        self.history['training_time'] = (finish_time - start_time) - test_time
        return self.transformer

    def start_trainingDPfedSAM(self, optimizer):
        # Set the model to training mode and move it to the device
        self.transformer.train()
        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'training_time': [],
            'clustering_time': 0
        }
        # scaler = GradScaler()
        # self.transformer.cpu()
        torch.cuda.empty_cache()

        start_time = time.time()
        test_time = 0
        print(f'Training {self.name}')
        torch.cuda.empty_cache()
        # self.transformer.cuda()
        self.transformer.to(self.config['device'])
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            src_data = torch.tensor(
                self.dataset['source_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'],
                dtype=torch.long)  # Ensure correct dtype to match model expectations

            tgt_data = torch.tensor(
                self.dataset['target_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'], dtype=torch.long)

            for batch in range(self.config['batch_epoch']):
                optimizer.zero_grad()
                outputs = self.transformer(src_data, tgt_data)
                # param_diffs = [u_new - u_old for u_new, u_old in zip(self.transformer.parameters(), w_glob)]
                # loss = self.custom_loss(outputs, tgt_data, param_diffs, "R1")
                loss = self.criterion(outputs.contiguous().view(-1, self.config['tgt_vocab_size']),
                                      tgt_data.contiguous().view(-1))
                loss.backward()
                self.clip_gradients(self.transformer.parameters(),
                                    self.config['clipping_bound'])  # Clipping is the same for BLUR_LUS and DP-FedSAM

                # self.history['loss'].append(loss.item())
                # print(f'loss optimizer 1: {loss.item()}')
                torch.cuda.empty_cache()
                # with torch.no_grad():
                #     for model_param, u_param in zip(self.transformer.parameters(), u_loc):
                #         model_param.grad *= (u_param != 0)
                optimizer.step()
                del outputs, loss
                torch.cuda.empty_cache()
            del src_data, tgt_data
            torch.cuda.empty_cache()
            test_time_start = time.time()
            self.history['iteration'].append(iteration + 1)
            test_loss = self.test_model(True)
            self.history['loss'].append(test_loss)
            test_time_finish = time.time()
            test_time += test_time_finish - test_time_start

        self.transformer.cpu()
        torch.cuda.empty_cache()
        # Track history
        # self.history['batch'].append(batch + 1)
        # update = [u_new - u_old for u_new, u_old in zip(self.transformer.parameters(), w_loc)]

        update = [self.compute_delta_w(param.grad, rho=0.5) + param for param in self.transformer.parameters()]
        # update = [param for param in self.transformer.parameters()]
        torch.cuda.empty_cache()
        gc.collect()

        noisy_updates = []
        noise_multiplier = Utils.compute_noise_multiplier(self.config)
        print(f'Test loss: {self.history["loss"][0]}')

        for clipped_update in update:
            # noise_stddev = 0.01
            noise_stddev = torch.sqrt(torch.tensor((self.config['clipping_bound'] ** 2) * (noise_multiplier ** 2) / (
                    self.config['data_in_each_worker'] * self.config['n_workers'])))
            # noise_stddev = torch.sqrt(torch.tensor((0.01 ** 2) * (noise_multiplier ** 2) / self.config['n_workers']))
            noise = [torch.randn_like(param) * noise_stddev for param in clipped_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_update, noise)]
            noisy_updates.append(noisy_update)
        converted_tensors = []

        for layer_updates in noisy_updates:
            layer_tensor = torch.stack(layer_updates, dim=0)  # Modify dim as needed
            converted_tensors.append(layer_tensor)
        self.transformer.cpu()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for param, new_weights in zip(self.transformer.parameters(), converted_tensors):
                param.copy_(new_weights)
        finish_time = time.time()
        self.history['training_time'] = (finish_time - start_time) - test_time
        return self.transformer

    def start_trainingDPBlurLus(self, optimizer):
        # Set the model to training mode and move it to the device
        self.transformer.train()
        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'training_time': [],
            'clustering_time': 0
        }
        # scaler = GradScaler()
        # self.transformer.cpu()
        torch.cuda.empty_cache()

        start_time = time.time()
        test_time = 0
        print(f'Training {self.name}')
        # self.transformer.to(self.config['device'])
        # global_model.to(self.config['device'])
        w_loc_begin = [param.clone().detach() for param in self.transformer.parameters()]

        torch.cuda.empty_cache()
        # self.transformer.cuda()
        self.transformer.to(self.config['device'])
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            src_data = torch.tensor(
                self.dataset['source_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'],
                dtype=torch.long)  # Ensure correct dtype to match model expectations

            tgt_data = torch.tensor(
                self.dataset['target_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'], dtype=torch.long)

            for batch in range(self.config['batch_epoch']):
                optimizer.zero_grad()
                outputs = self.transformer(src_data, tgt_data)
                # param_diffs = [u_new - u_old for u_new, u_old in zip(self.transformer.parameters(), w_glob)]
                # loss = self.custom_loss(outputs, tgt_data, param_diffs, "R1")
                loss = self.criterion(outputs.contiguous().view(-1, self.config['tgt_vocab_size']),
                                      tgt_data.contiguous().view(-1))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 0.5)
                self.clip_gradients(self.transformer.parameters(), self.config['clipping_bound'])

                # self.history['loss'].append(loss.item())
                # print(f'loss optimizer 1: {loss.item()}')
                torch.cuda.empty_cache()
                # with torch.no_grad():
                #     for model_param, u_param in zip(self.transformer.parameters(), u_loc):
                #         model_param.grad *= (u_param != 0)
                optimizer.step()
                del outputs, loss
                torch.cuda.empty_cache()
            del src_data, tgt_data
            torch.cuda.empty_cache()
            test_time_start = time.time()
            test_loss = self.test_model(False)
            # Track history
            self.history['iteration'].append(iteration + 1)
            self.history['loss'].append(test_loss)
            test_time_finish = time.time()
            test_time += test_time_finish - test_time_start
        self.transformer.cpu()
        torch.cuda.empty_cache()
        # Track history
        # self.history['batch'].append(batch + 1)
        update_delta = [u_new - u_old for u_new, u_old in zip(self.transformer.parameters(), w_loc_begin)]
        sparsify_updates = self.sparsify_updates(model_updates=update_delta, c=self.config['sparsification_ratio'])
        # for (name, param), (update_name, update_delta) in zip(self.transformer.named_parameters(), sparsify_updates):
        #     if name == update_name:
        #         param.data += update_delta
        # update = [param for param in self.transformer.parameters()]
        torch.cuda.empty_cache()
        gc.collect()
        noisy_updates = []
        noise_multiplier = Utils.compute_noise_multiplier(self.config)
        print(f'Test loss: {self.history["loss"][0]}')

        for name, sparse_update in sparsify_updates:
            # noise_stddev = 0.01
            noise_stddev = torch.sqrt(torch.tensor((self.config['clipping_bound'] ** 2) * (noise_multiplier ** 2) / (
                    self.config['data_in_each_worker'] * self.config['n_workers'])))
            noise = [torch.randn_like(param) * noise_stddev for param in sparse_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(sparse_update, noise)]
            noisy_updates.append((name, noisy_update))
        converted_tensors = []

        for name, layer_updates in noisy_updates:
            # Concatenate the tensors along the appropriate dimension
            # Assuming layer_updates is a list of tensors that should be concatenated along dim=1
            # Modify this based on the actual required dimension for concatenation
            layer_tensor = torch.stack(layer_updates, dim=0)  # Modify dim as needed
            converted_tensors.append((name, layer_tensor))
        self.transformer.cpu()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for param_name, param in self.transformer.named_parameters():
                for new_weights_name, new_param in converted_tensors:
                    if param_name == new_weights_name:
                        param.data.add_(new_param)


        finish_time = time.time()
        self.history['training_time'] = (finish_time - start_time) - test_time
        return converted_tensors

    def utility_cost(self, delta_w, grad_w):
        """
        Calculate the utility cost T(Δw; w) = |grad_w * delta_w|.

        Args:
        - delta_w (torch.Tensor): The update tensor for the layer.
        - grad_w (torch.Tensor): The gradient of the loss with respect to the weights.

        Returns:
        - utility (torch.Tensor): The utility cost for the update.
        """
        utility = torch.abs(grad_w * delta_w)
        return utility

    def mask_matrix(self, delta_w, grad_w, s=0.1):
        """
        Constructs a mask matrix for layer updates based on the utility cost and threshold.

        Args:
        - delta_w (torch.Tensor): The update tensor for the layer.
        - grad_w (torch.Tensor): The gradient of the loss with respect to the weights.
        - threshold (float): The threshold value.

        Returns:
        - mask (torch.Tensor): The mask matrix with the same shape as `delta_w`.
        """
        # Calculate the utility cost
        utility = self.utility_cost(delta_w, grad_w)
        # threshold, _ = torch.topk(torch.abs(delta_w).view(-1), s)
        threshold = s  # Take the s-th largest value
        # Determine the mask matrix
        mask = (utility >= threshold).float()

        return mask

    def sparsify_updates(self, model_updates, c=0.1):
        """
        Constructs the mask matrices for the entire model and sparsifies the updates.

        Args:
        - model_updates (list of torch.Tensor): List of update tensors for each layer.
        - model_grads (list of torch.Tensor): List of gradient tensors for each layer.
        - threshold (float, optional): The threshold value. Defaults to 0.3.

        Returns:
        - sparse_updates (list of torch.Tensor): List of sparsified update tensors for each layer.
        """
        sparse_updates = []
        for (name, param), update in zip(self.transformer.named_parameters(), model_updates):
            if param.grad is not None:
                grad_w = param.grad.data
                delta_w = update.data
                # s = math.floor(c * delta_w.shape[0])
                mask = self.mask_matrix(delta_w, grad_w, s=c)
                sparse_update = mask * delta_w
                if torch.any(sparse_update != 0):  # Check if any element in sparse_update is non-zero
                    sparse_updates.append((name, sparse_update))

        return sparse_updates

    def clip_gradients(self, parameters, clip_value):
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = clip_value / (total_norm + 1e-6)  # Adding a small value to avoid division by zero
        clip_coef = min(1.0, clip_coef)

        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    def clip_parameters(self, parameters, clip_value):
        total_norm = 0.0
        for param in parameters:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = clip_value / (total_norm + 1e-6)  # Adding a small value to avoid division by zero
        clip_coef = min(1.0, clip_coef)

        for param in parameters:
            param.data.mul_(clip_coef)

    def compute_delta_w(self, gradients, rho):
        """
        Compute delta w using the given gradients and scaling factor rho.

        Parameters:
        gradients (torch.Tensor): Gradients tensor.
        rho (float): Scaling factor.

        Returns:
        torch.Tensor: Computed delta w.
        """
        # Compute the L2 norm of the gradients
        l2_norm = torch.norm(gradients, p=2)

        # Compute delta w
        delta_w = (rho * gradients) / l2_norm

        return delta_w

    def start_trainingDPFL(self, optimizer, global_model):
        # Set the model to training mode and move it to the device
        global optimizer2
        self.transformer.train()
        self.transformer.to(self.config['device'])
        global_model.to(self.config['device'])
        torch.cuda.empty_cache()
        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'training_time': [],
            'clustering_time': 0
        }
        # scaler = GradScaler()
        fisher_threshold = self.config['fisher_threshold']
        fisher_diag = Utils.compute_fisher_diag(self.config, self.transformer, self.dataset)
        fisher_diag = [fisher_tensor.cpu() for fisher_tensor in fisher_diag]
        self.transformer.cpu()
        torch.cuda.empty_cache()
        u_loc, v_loc = [], []
        for param, fisher_value in zip(self.transformer.parameters(), fisher_diag):
            u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
            u_loc.append(u_param)
            v_loc.append(v_param)
            del u_param, v_param
            torch.cuda.empty_cache()
        global_model.cpu()
        torch.cuda.empty_cache()
        u_glob, v_glob = [], []
        for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
            u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)
            del u_param, v_param
            torch.cuda.empty_cache()
        for u_param, v_param, model_param in zip(u_loc, v_glob, self.transformer.parameters()):
            model_param.data = u_param + v_param

        # saved_u_loc = [u.clone() for u in u_loc]
        start_time = time.time()
        test_time = 0
        print(f'Training {self.name}')
        self.transformer.to(self.config['device'])
        global_model.to(self.config['device'])
        w_glob = [param.clone().detach() for param in global_model.parameters()]
        del global_model
        u_loc = [u_loc_tensor.cuda() for u_loc_tensor in u_loc]
        torch.cuda.empty_cache()
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            src_data = torch.tensor(
                self.dataset['source_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'],
                dtype=torch.long)  # Ensure correct dtype to match model expectations

            tgt_data = torch.tensor(
                self.dataset['target_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'], dtype=torch.long)

            for batch in range(self.config['batch_epoch']):
                optimizer.zero_grad()
                outputs = self.transformer(src_data, tgt_data)
                param_diffs = [u_new - u_old for u_new, u_old in zip(self.transformer.parameters(), w_glob)]
                loss = self.custom_loss(outputs, tgt_data, param_diffs, "R1")
                loss.backward()
                self.history['loss'].append(loss.item())
                # print(f'loss optimizer 1: {loss.item()}')
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for model_param, u_param in zip(self.transformer.parameters(), u_loc):
                        model_param.grad *= (u_param != 0)
                optimizer.step()
                del outputs, loss, param_diffs
                torch.cuda.empty_cache()
            del src_data, tgt_data
            torch.cuda.empty_cache()
        u_loc = [u_loc_tensor.cpu() for u_loc_tensor in u_loc]
        torch.cuda.empty_cache()

        optimizer2 = optim.Adam(self.transformer.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98),
                                eps=1e-4)
        update = []
        v_glob = [v_glob_tensor.cuda() for v_glob_tensor in v_glob]
        torch.cuda.empty_cache()
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            src_data = torch.tensor(
                self.dataset['source_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'],
                dtype=torch.long)  # Ensure correct dtype to match model expectations

            tgt_data = torch.tensor(
                self.dataset['target_ids'][
                iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                device=self.config['device'], dtype=torch.long)
            for batch in range(self.config['batch_epoch']):
                optimizer2.zero_grad()
                outputs = self.transformer(src_data, tgt_data)
                param_diffs = [model_param - w_old for model_param, w_old in zip(self.transformer.parameters(), w_glob)]
                loss = self.custom_loss(outputs, tgt_data, param_diffs, "R2")
                # print(f'loss optimizer 2: {loss.item()}')
                loss.backward()
                with torch.no_grad():
                    for model_param, v_param in zip(self.transformer.parameters(), v_glob):
                        model_param.grad *= (v_param != 0)
                optimizer2.step()

                with torch.no_grad():
                    update = [(new_param - old_param).clone() for new_param, old_param in
                              zip(self.transformer.parameters(), w_glob)]
                torch.cuda.empty_cache()

                # self.transformer.to('cpu')
                del outputs, loss, param_diffs
                torch.cuda.empty_cache()
            del src_data, tgt_data
            torch.cuda.empty_cache()
        v_glob = [v_glob_tensor.cpu() for v_glob_tensor in v_glob]
        update = [update_tensor.cpu() for update_tensor in update]
        torch.cuda.empty_cache()
        # test_time_start = time.time()

        # test_loss = self.test_model(loss, False)
        #         # Track history
        # self.history['iteration'].append(iteration + 1)
        #         # self.history['batch'].append(batch + 1)
        # self.history['loss'].append(test_loss)
        # test_time_finish = time.time()
        # Optional: Clear cache periodically to free unused memory

        # Manually clear variables to ensure they are deleted from memory

        # torch.cuda.empty_cache()
        # Trigger garbage collection to reclaim memory`
        # torch.cuda.empty_cache()
        gc.collect()
        # test_time += test_time_finish - test_time_start
        # print(f"first batch Loss: {self.history['loss'][0]} of  {self.name}")
        # finish_time = time.time()
        # self.transformer.to('cpu')
        # self.history['training_time'] = (finish_time - start_time) - test_time
        clipped_updates = []
        for idx, client_update in enumerate(update):
            if not self.config['no_clip']:
                norm = torch.sqrt(sum([torch.sum(param ** 2) for param in client_update]))
                clip_rate = max(1, (norm / self.config['clipping_bound']))
                clipped_update = [(param / clip_rate) for param in client_update]
            else:
                clipped_update = client_update
            clipped_updates.append(clipped_update)

        noisy_updates = []
        noise_multiplier = Utils.compute_noise_multiplier(self.config)
        print(f'optimizer 1 loss: {self.history["loss"][0]}')

        for clipped_update in clipped_updates:
            noise_stddev = torch.sqrt(
                torch.tensor((self.config['clipping_bound'] ** 2) * (noise_multiplier ** 2) / self.config['n_workers']))
            noise = [torch.randn_like(param) * noise_stddev for param in clipped_update]
            noisy_update = [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_update, noise)]
            noisy_updates.append(noisy_update)
        # updates = torch.tensor(noisy_updates)
        converted_tensors = []

        for layer_updates in noisy_updates:
            # Concatenate the tensors along the appropriate dimension
            # Assuming layer_updates is a list of tensors that should be concatenated along dim=1
            # Modify this based on the actual required dimension for concatenation
            layer_tensor = torch.stack(layer_updates, dim=0)  # Modify dim as needed
            converted_tensors.append(layer_tensor)
        self.transformer.cpu()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for param, new_weights in zip(self.transformer.parameters(), converted_tensors):
                param.copy_(new_weights)
        return self.transformer

    def custom_loss(self, outputs, labels, param_diffs, reg_type):
        criterion = nn.CrossEntropyLoss(ignore_index=1)
        ce_loss = criterion(outputs.contiguous().view(-1, self.config['tgt_vocab_size']), labels.contiguous().view(-1))
        lambda_1 = 0.1
        lambda_2 = 0.05
        clipping_bound = self.config['clipping_bound']
        if reg_type == "R1":
            reg_loss = (lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

        elif reg_type == "R2":
            C = clipping_bound
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (lambda_2 / 2) * torch.norm(norm_diff - C)

        else:
            raise ValueError("Invalid regularization type")
        return ce_loss + reg_loss

        # return self.transformer

    def test_model(self, training_loss, logging=False):
        source_ids_validation = torch.tensor(
            self.dataset['source_ids_validation'][
            0:self.config['batch_size']],
            device=self.config['device'], dtype=torch.long)  # Ensure correct dtype to match model expectations

        target_ids_validation = torch.tensor(
            self.dataset['target_ids_validation'][
            0:self.config['batch_size']],
            device=self.config['device'], dtype=torch.long)
        val_output = self.transformer(source_ids_validation, target_ids_validation)
        # Compute the loss
        val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                  target_ids_validation.contiguous().view(-1))

        if logging == True:
            print(f"Validation Loss: {val_loss.item()}")

        return val_loss.item()

    def evaluate_model(self, logging=False):
        self.transformer.eval()
        self.transformer.to(self.config['device'])
        text_results = []
        loss_results = []
        metrics = {'bert_score': [], 'bleu_score': [], 'rouge_scores': [], 'meteor_score': [], 'time': []}
        with (torch.autograd.no_grad()):  # torch.no_grad():
            for iteration in range(len(self.dataset['source_ids_validation']) // self.config['batch_size']):
                source_ids_validation = torch.tensor(
                    self.dataset['source_ids_validation'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)  # Ensure correct dtype to match model expectations

                target_ids_validation = torch.tensor(
                    self.dataset['target_ids_validation'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)

                # source_ids_validation = torch.tensor(self.dataset['source_ids_validation']).to(self.config['device'])
                # target_ids_validation = torch.tensor(self.dataset['target_ids_validation']).to(self.config['device'])
                start_time = time.time()
                # Forward pass
                val_output = self.transformer(source_ids_validation, target_ids_validation)
                # Compute the loss
                val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                          target_ids_validation.contiguous().view(-1))
                finish_time = time.time()
                # Print validation loss
                print(f"Validation Loss: {val_loss.item()}")
                loss_results.append(val_loss.item())
                # loss_results.append(val_loss.item())
                generated_tokens = torch.argmax(val_output, dim=-1)

                # Convert token IDs to actual tokens using the BART tokenizer
                # generated_texts = ds.decode_tokens(generated_tokens)
                # text_results.append(generated_texts)
                candidate_corpus = ds.decode_tokens(generated_tokens)
                text_results.append(candidate_corpus)
                reference_corpus = ds.decode_tokens(target_ids_validation)

                # candidate_corpuses = ds.decode_tokens(generated_tokens)
                if logging:
                    print(candidate_corpus)
                P, R, F1 = Metrics.score(candidate_corpus, reference_corpus, lang="en")
                if logging:
                    print(f"BERTScore Precision: {P.mean()}, Recall: {R.mean()}, F1 Score: {F1.mean()}")
                # metrics['bert_score'] = {"P": P.mean(), "R":R.mean(), "F1 score": F1.mean()}
                # for key in metrics['bert_score']:
                metrics['bert_score'].append({"P": P.mean(), "R": R.mean(), "F1 score": F1.mean()})
                # bleu_score = Metrics.sentence_bleu(reference_corpus, candidate_corpus)
                bleu_score = np.average(Metrics.bleu_score(references=reference_corpus, candidates=candidate_corpus))
                if logging:
                    print(f"BLEU Score: {bleu_score}")
                metrics['bleu_score'].append(bleu_score)
                rouge_scores = Metrics.compute_rouge(candidate_corpus, reference_corpus)
                if logging:
                    print("ROUGE Scores:", rouge_scores)
                metrics['rouge_scores'].append(rouge_scores)
                meteor_score_1 = Metrics.compute_meteor(candidate_corpus, reference_corpus)
                if logging:
                    print("METEOR Score:", meteor_score_1)
                metrics['meteor_score'].append(meteor_score_1)
                metrics['time'].append(finish_time - start_time)
                del source_ids_validation, target_ids_validation, val_output, val_loss, generated_tokens, candidate_corpus, P, R, F1, bleu_score, rouge_scores, meteor_score_1
                torch.cuda.empty_cache()
                # Trigger garbage collection to reclaim memory
                gc.collect()

            self.transformer.to('cpu')
            torch.cuda.empty_cache()
        return text_results, loss_results, metrics

    def get_model(self):
        return self.transformer.state_dict()

    def get_optimizer(self):
        return self.optimizer

    def set_parameters(self, new_transformer):
        for param_new, param1 in zip(new_transformer.parameters(), self.transformer.parameters()):
            # Check if the parameter is trainable
            if param_new.requires_grad:
                # Average the parameter values
                param1.data.copy_(param_new.data)

    def load_decrypt_model(self, file_path):
        self.transformer.load_state_dict(sgx.load_model(file_path=file_path, key=self.key))
        return self

    def load_plain_model(self, file_path):
        self.transformer.load_state_dict(sgx.load_plain_model(file_path=file_path))
        return self

    def set_optimizer(self, optimizer, optimizer_name='sgd'):
        if optimizer_name == 'sgd':
            optimizer2 = optim.SGD(params=self.transformer.parameters(), lr=self.config["learning_rate"],
                                   momentum=self.config["momentum"],
                                   dampening=self.config["dampening"], weight_decay=self.config["weight_decay"],
                                   nesterov=self.config["nesterov"])
            # scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.1)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
            # scheduler.step()

        if optimizer_name == 'adam':
            # scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
            # scheduler.step()
            self.optimizer = optimizer

    # def add_noise_to_model(self, noisy_update):
    #     # for param in self.transformer.parameters():
    #     #     if param.requires_grad:
    #     #         noise = torch.normal(mean=0.0, std=noise_std, size=param.data.size())
    #     #         param.data.add_(noise)
    #     for param, noise in zip(self.transformer.parameters(), noisy_update):
    #         if param.requires_grad:
    #             param.data.copy_(noise)
    def add_noise_to_model(self, noisy_update):
        state_dict = self.transformer.state_dict()

        # Iterate over the state dictionary and copy the noisy updates to the transformer's parameters
        for name, param in state_dict.items():
            param.copy_(noisy_update[name])

        # Load the updated state dictionary back into the transformer
        self.transformer.load_state_dict(state_dict)

    def encrypt_store_model(self, sparsification_enabled=False):
        print('SGX encrypting and storing the transformer of : ', self.name)

        if sparsification_enabled:  # This is the aggregation method in
            directory = "sealed_models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = "sealed_models/{}".format(self.name)
            # model_updates = self.sparsify_based_on_std(self.transformer.state_dict(), 0.5)
            clustering_time = time.time()
            model_updates, membership = self.apply_weight_sharing(num_clusters_ratio=self.config['n_cluster_ratio'])
            self.centroids = model_updates
            self.membership = membership
            self.history['clustering_time'] += time.time() - clustering_time
            sgx.store_model(self.key, self.transformer.state_dict(), filename=f"sealed_models/model_{self.name}")
            sgx.store_model(self.key, model_updates, filename=file_path)
        else:  # Here we implement DP-FedAvg
            directory = "unsealed_models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = f"unsealed_models/{self.name}.pkl"
            if self.config['method'] == "DP-fedAvg":
                clipped_update = self.clipping_model()
                noisy_update = self.add_noise_to_clipped_model(clipped_update)

                # sensitivity = self.estimate_sensitivity()
                # delta_f = sensitivity
                # noise_std = self.config['noise_std']  # delta_f / self.config['epsilon']  # Calculate standard deviation
                # self.add_noise_to_model(noise_std / self.config['epsilon'])
                self.add_noise_to_model(noisy_update=noisy_update)
            model_updates = self.transformer.state_dict()
            sgx.store_plain_model(model_updates, filename=file_path)
        return file_path

    # def add_noise_to_clipped_model(self, clipped_updates):
    #     noise_multiplier = Utils.compute_noise_multiplier(config=self.config)
    #     noise_stddev = torch.sqrt(torch.tensor((self.config['clipping_bound']**2) * (noise_multiplier**2) / self.config['n_workers']))
    #     noise = [torch.randn_like(param) * noise_stddev for param in clipped_updates]
    #     return [clipped_param + noise_param for clipped_param, noise_param in zip(clipped_updates, noise)]
    def add_noise_to_clipped_model(self, clipped_updates):
        noise_multiplier = Utils.compute_noise_multiplier(config=self.config)
        noise_stddev = torch.sqrt(
            torch.tensor((self.config['clipping_bound'] ** 2) * (noise_multiplier ** 2) / self.config['n_workers']))

        # Apply noise to each parameter in the state dictionary
        noisy_updates = {}
        for name, param in clipped_updates.items():
            noise = torch.randn_like(param) * noise_stddev
            noisy_updates[name] = param + noise

        return noisy_updates

    # def clipping_model(self):
    #     if not self.config['no_clip']:
    #         norm = torch.sqrt(sum([torch.sum(param ** 2) for param in self.transformer.parameters()]))
    #         clip_rate = max(1, (norm / self.config['clipping_bound']))
    #         clipped_update = [(param / clip_rate) for param in self.transformer.parameters()]
    #     else:
    #         clipped_update = self.transformer.state_dict()
    #     return clipped_update
    def clipping_model(self):
        if not self.config['no_clip']:
            # Calculate the norm of the parameters
            norm = torch.sqrt(sum(torch.sum(param ** 2) for param in self.transformer.parameters()))
            # Determine the clipping rate
            clip_rate = max(1, norm / self.config['clipping_bound'])
            # Apply the clipping
            clipped_update = [param / clip_rate for param in self.transformer.parameters()]
            # Convert to a dictionary with the same keys as the state_dict for consistency
            clipped_update_dict = {name: param.clone() for name, param in
                                   zip(self.transformer.state_dict().keys(), clipped_update)}
        else:
            # Get the state_dict of the transformer
            clipped_update_dict = self.transformer.state_dict()
        return clipped_update_dict

    def apply_weight_sharing(self, num_clusters_ratio=0.1):

        centroids_dict = {}
        membership_dict = {}
        for name, param in self.transformer.named_parameters():
            num_clusters = math.floor(param.data.shape[0] * num_clusters_ratio)
            if len(param.shape) > 1:  # Only apply to weight matrices
                # weight = param.data.cpu().numpy().flatten()
                # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(weight.reshape(-1, 1))
                torch.cuda.empty_cache()
                centroids, labels2 = self.kmeans_torch_batch(param.data, num_clusters=num_clusters)
                # new_weight = kmeans.cluster_centers_[kmeans.labels_]
                # param.data = torch.tensor(new_weight.reshape(param.shape)).to(param.device)
                # centroids_dict[name] = kmeans.cluster_centers_.flatten()
                # membership_dict[name] = kmeans.labels_
                centroids_dict[name] = centroids.cpu().numpy().flatten()
                del centroids
                membership_dict[name] = labels2.cpu().numpy()
                del labels2
        return centroids_dict, membership_dict

    # def apply_weight_sharing(self, num_clusters_ratio=0.1):
    #     device = self.config['device']
    #     centroids_dict = {}
    #     membership_dict = {}
    #
    #     for name, param in self.transformer.named_parameters():
    #         num_clusters = math.floor(param.data.shape[0] * num_clusters_ratio)
    #         if len(param.shape) > 1:  # Only apply to weight matrices
    #             param_data = param.data.to(device)
    #             torch.cuda.empty_cache()
    #
    #             centroids, labels2 = self.kmeans_torch_batch(param_data, num_clusters=num_clusters)
    #
    #             centroids_dict[name] = centroids.flatten()
    #             del centroids
    #
    #             membership_dict[name] = labels2
    #             del labels2
    #
    #             torch.cuda.empty_cache()
    #
    #     return centroids_dict, membership_dict
    #
    def kmeans_torch(self, data, num_clusters, num_iterations=50):
        data = data.to(self.config['device'])
        num_samples, num_features = data.shape
        torch.manual_seed(self.config["seed_value"])  # Set to a specific number for reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed_value"])
        # Initialize centroids randomly from the data points
        indices = torch.randperm(num_samples)[:num_clusters]
        centroids = data[indices]
        del indices
        for _ in range(num_iterations):
            # Compute distances between each point and each centroid
            distances = torch.cdist(data, centroids)

            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)

            # Compute new centroids as the mean of the points assigned to each centroid
            new_centroids = torch.zeros_like(centroids)
            for i in range(num_clusters):
                cluster_points = data[labels == i]
                torch.cuda.empty_cache()
                if cluster_points.shape[0] > 0:
                    new_centroids[i] = cluster_points.mean(dim=0)

            # Check for convergence (optional)
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids
            torch.cuda.empty_cache()
            del new_centroids
        del distances
        # data = data.to('cpu')
        del data
        return centroids, labels

    def kmeans_torch(self, data, num_clusters, num_iterations=50):
        data = data.to(self.config['device'])
        num_samples, num_features = data.shape
        torch.manual_seed(self.config["seed_value"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed_value"])

        # Initialize centroids randomly from the data points
        indices = torch.randperm(num_samples, device=self.config['device'])[:num_clusters]
        centroids = data[indices]

        for _ in range(num_iterations):
            # Compute distances between each point and each centroid
            distances = torch.cdist(data, centroids, p=2)

            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)
            del distances
            # Compute new centroids as the mean of the points assigned to each centroid
            new_centroids = torch.stack(
                [data[labels == i].mean(dim=0) if (labels == i).any() else centroids[i] for i in range(num_clusters)])

            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids
            del new_centroids
        del data
        return centroids, labels

    def kmeans_torch_batch(self, data, num_clusters, num_iterations=50, batch_size=200):
        device = self.config['device']
        data = data.to(device)
        num_samples, num_features = data.shape
        torch.manual_seed(self.config["seed_value"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config["seed_value"])

        # Initialize centroids randomly from the data points
        indices = torch.randperm(num_samples, device=device)[:num_clusters]
        centroids = data[indices]

        for iteration in range(num_iterations):
            all_labels = torch.empty(num_samples, dtype=torch.long, device=device)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = data[start:end]

                # Ensure GPU operations
                distances = torch.cdist(batch, centroids, p=2)
                labels = torch.argmin(distances, dim=1)
                all_labels[start:end] = labels

            # Accumulate sums and counts for each cluster
            counts = torch.zeros(num_clusters, device=device)
            new_centroids = torch.zeros_like(centroids)

            for i in range(num_clusters):
                mask = (all_labels == i)
                if mask.sum() > 0:
                    new_centroids[i] = data[mask].mean(dim=0)
                    counts[i] = mask.sum()

            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        return centroids, all_labels

    def adjust_local_model_with_centroids(self, global_centroids, num_clusters_ratio=0.1):
        print(f'Applying k-means on all layers ....')
        local_centroids = self.centroids
        memberships = self.membership
        # local_centroids, memberships = #self.apply_weight_sharing(num_clusters_ratio=num_clusters_ratio)
        print(f'setting up weights for all layers')
        for name, param in self.transformer.named_parameters():
            if name in global_centroids:
                # print(f'setting up info for layer {name}')
                # local_centroids = self.apply_weight_sharing(num_clusters=len(global_centroids[name]))[name]
                centroid_diff = global_centroids[name] - local_centroids[name]
                # weight = param.data.cpu().numpy().flatten()
                # for idx, cluster in enumerate(memberships[name]):
                #     weight[idx] += centroid_diff[cluster]
                weight = param.data.cpu().numpy()
                # adjustments = np.zeros_like(weight)

                adjustments = centroid_diff.reshape(-1, param.data.shape[1])[memberships[name]]
                adjusted_weight = weight + adjustments
                # param.data = torch.tensor(adjusted_weight.reshape(param.data.shape)).to(param.device)
                param.data.copy_(torch.tensor(adjusted_weight.reshape(param.data.shape)).to(param.device))

                # param.data = torch.tensor(weight.reshape(param.data.shape)).to(param.device)

    def save_model(self, filepath):
        torch.save(self.transformer.state_dict(), filepath)

    def set_model(self, newTransformer):

        self.transformer.load_state_dict(newTransformer)

    def sparsify_based_on_std(self, updates, fraction):
        """
        Sparsify the model updates based on each layer's standard deviation.

        Args:
            updates (dict): Dictionary of model updates (layer name: torch tensor of updates).
            fraction (float): Fraction of the standard deviation to use as the threshold.

        Returns:
            dict: Sparsified updates.
        """
        sparsified_updates = {}

        for layer_name, layer_updates in updates.items():
            std_dev = torch.std(layer_updates).item()
            # threshold = fraction * std_dev
            # sparsified_layer_updates = torch.where(torch.abs(layer_updates) >= threshold, layer_updates,
            #                                        torch.tensor(0.0))
            if (std_dev > fraction):
                sparsified_updates[layer_name] = layer_updates

        return sparsified_updates

    def estimate_sensitivity(self):
        total_norm = 0.0
        max_grad_norm = 0.0

        for param in self.transformer.parameters():
            if param.requires_grad:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        max_grad_norm = max(max_grad_norm, total_norm)

        return max_grad_norm
