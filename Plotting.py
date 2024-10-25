import matplotlib.pyplot as plt
import pickle
import os
import glob
# Load a single pickle file

# Define a function to load data for a specific worker
def load_worker_data(worker_id):
    with open(f'results/evaluation_worker{worker_id}.pkl', 'rb') as file:
        return pickle.load(file)


# Function to generate and save plots
def generate_eval_plots(worker_ids, output_dir='plots'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Colors and styles for different workers
    colors = ['b', 'g', 'r']
    styles = ['-', '--', '-.']

    # Plot loss over evaluations for all workers
    plt.figure(figsize=(10, 6))
    for worker_id, color, style in zip(worker_ids, colors, styles):
        data = load_worker_data(worker_id)
        plt.plot(data['loss'], label=f'Worker {worker_id} Loss', color=color, linestyle=style)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Loss Over Evaluation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_over_evaluation.png'))
    plt.close()

    # Plot evaluation metric scores over time for all workers
    plt.figure(figsize=(10, 6))
    for worker_id, color, style in zip(worker_ids, colors, styles):
        data = load_worker_data(worker_id)
        bert_f1_scores = [d['F1 score'] for d in data['metrics']['bert_score']]
        bleu_scores = data['metrics']['bleu_score']
        rouge_scores = [d['ROUGE-2'] for d in data['metrics']['rouge_scores']]
        meteor_scores = data['metrics']['meteor_score']

        plt.plot(bert_f1_scores, label=f'Worker {worker_id} BERT F1 Score', color=color, linestyle=style)
        plt.plot(bleu_scores, label=f'Worker {worker_id} BLEU Score', color=color, linestyle=style)
        plt.plot(rouge_scores, label=f'Worker {worker_id} ROUGE Scores', color=color, linestyle=style)
        plt.plot(meteor_scores, label=f'Worker {worker_id} METEOR Score', color=color, linestyle=style)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Score')
    plt.title('Evaluation Metric Scores Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'evaluation_metric_scores_over_time.png'))
    plt.close()

    # Plot evaluation time over steps for all workers
    plt.figure(figsize=(10, 6))
    for worker_id, color, style in zip(worker_ids, colors, styles):
        data = load_worker_data(worker_id)
        evaluation_times = data['metrics']['time']
        plt.plot(evaluation_times, label=f'Worker {worker_id} Evaluation Time', color=color, linestyle=style)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Time (seconds)')
    plt.title('Evaluation Time Over Steps')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'evaluation_time_over_steps.png'))
    plt.close()



# Example usage
worker_files = [
    'results/1/worker0/worker0.json',
    'results/1/worker1/worker1.json',
    'results/1/worker2/worker2.json'
]

# average_losses = aggregate_loss_values(worker_files)
# plot_average_loss(average_losses)


import os
import json
import matplotlib.pyplot as plt


# Define a function to load data from a worker file
def load_worker_data2(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


# Function to compute the average loss for each worker for a given epoch
def compute_epoch_average_loss(epoch_folder):
    worker_files = [
        os.path.join(epoch_folder, 'worker0/worker0.json'),
        os.path.join(epoch_folder, 'worker1/worker1.json'),
        os.path.join(epoch_folder, 'worker2/worker2.json')
    ]

    worker_losses = {}

    for worker_file in worker_files:
        worker_data = load_worker_data2(worker_file)
        worker_id = os.path.basename(worker_file).split('/')[0]

        average_loss = sum(worker_data['loss']) / len(worker_data['loss'])

        if worker_id not in worker_losses:
            worker_losses[worker_id] = []
        worker_losses[worker_id].append(average_loss)

    return worker_losses  # Return dictionary of average losses for each worker


# Function to aggregate and plot losses over all epochs
def plot_all_epochs_losses(base_dir='results', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    epoch_folders = [os.path.join(base_dir, epoch) for epoch in os.listdir(base_dir) if
                     os.path.isdir(os.path.join(base_dir, epoch))]
    epoch_folders.sort(key=lambda x: int(os.path.basename(x)))  # Sort by epoch number

    epoch_numbers = []
    all_worker_losses = {}

    for epoch_folder in epoch_folders:
        epoch_number = int(os.path.basename(epoch_folder))
        worker_losses = compute_epoch_average_loss(epoch_folder)

        epoch_numbers.append(epoch_number)

        for worker_id, losses in worker_losses.items():
            if worker_id not in all_worker_losses:
                all_worker_losses[worker_id] = []
            all_worker_losses[worker_id].extend(losses)

    # Plot average losses for each worker over all epochs
    plt.figure(figsize=(10, 6))
    for i, (worker_id, losses) in enumerate(all_worker_losses.items()):
        plt.plot(epoch_numbers, losses, marker='o', linestyle='-', label=f'Worker {i + 1}')

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over All Epochs for Each Worker')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'average_loss_per_worker_over_epochs.png'))
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_files_info(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # Extract relevant metrics
    bert_scores = data['metrics']['bert_score']
    f1_scores = [score['F1 score'].item() for score in bert_scores]  # Convert tensor to float

    bleu_scores = data['metrics']['bleu_score']
    rouge_scores = data['metrics']['rouge_scores']
    meteor_scores = data['metrics']['meteor_score']
    time_values = data['metrics']['time']
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_rouge_1 = sum([score['ROUGE-1'] for score in rouge_scores]) / len(rouge_scores)
    average_rouge_2 = sum([score['ROUGE-2'] for score in rouge_scores]) / len(rouge_scores)
    average_rouge_l = sum([score['ROUGE-L'] for score in rouge_scores]) / len(rouge_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    average_time = sum(time_values)/len(time_values)

    return [average_bleu, average_rouge_1, average_rouge_2, average_rouge_l, average_f1, average_meteor, average_time]


# Compute the averages

def draw_plots_workers(file_path_baseline = 'baseline', file_path_results = 'results'):
    # Baseline worker data
    baseline_worker_1 = load_files_info(file_path = f'{file_path_baseline}/evaluation_worker0.pkl')
    # Hypothetical values for workers 2 and 3, and proposed work values
    baseline_worker_2 = load_files_info(file_path = f'{file_path_baseline}/evaluation_worker1.pkl')
    baseline_worker_3 = load_files_info(file_path = f'{file_path_baseline}/evaluation_worker2.pkl')

    proposed_worker_1 = load_files_info(file_path = f'{file_path_results}/evaluation_worker0.pkl')
    proposed_worker_2 = load_files_info(file_path = f'{file_path_results}/evaluation_worker1.pkl')
    proposed_worker_3 = load_files_info(file_path = f'{file_path_results}/evaluation_worker0.pkl')

    # Data for plotting
    baseline_data = [baseline_worker_1, baseline_worker_2, baseline_worker_3]
    proposed_data = [proposed_worker_1, proposed_worker_2, proposed_worker_3]

    # Metrics
    metrics = ['BLEU Score', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1', 'METEOR', 'Time']

    # Colors
    baseline_color = 'blue'
    proposed_color = 'green'

    # Creating separate plots for each metric
    for i in range(len(metrics)):
        fig, ax = plt.subplots(figsize=(4, 3))
        bar_width = 0.2
        r1 = np.arange(3)
        r2 = [x + bar_width for x in r1]

        ax.bar(r1, [baseline_data[j][i] for j in range(3)], color=baseline_color, width=bar_width, edgecolor='grey',
               label='Baseline')
        ax.bar(r2, [proposed_data[j][i] for j in range(3)], color=proposed_color, width=bar_width, edgecolor='grey',
               label='Proposed')

        ax.set_xlabel('Workers', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_xticks([r + bar_width / 2 for r in range(3)])
        ax.set_xticklabels(['Worker 1', 'Worker 2', 'Worker 3'])
        ax.set_title(metrics[i])
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'comparison_{metrics[i].lower().replace(" ", "_")}.png')
        plt.show()


draw_plots_workers()