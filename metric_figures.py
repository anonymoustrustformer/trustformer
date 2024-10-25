import matplotlib.pyplot as plt
import numpy as np
import pickle


def analyze_files(file_path = 'results/baseline/evaluation_worker0.pkl'):
# Load the data from the provided file
# file_path = 'results/evaluation_worker0.pkl'

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

# Extract relevant metrics
    bert_scores = data['metrics']['bert_score']
    f1_scores = [score['F1 score'].item() for score in bert_scores]  # Convert tensor to float

    bleu_scores = data['metrics']['bleu_score']
    rouge_scores = data['metrics']['rouge_scores']
    meteor_scores = data['metrics']['meteor_score']
    time_values = data['metrics']['time']

    # Compute the averages
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_rouge_1 = sum([score['ROUGE-1'] for score in rouge_scores]) / len(rouge_scores)
    average_rouge_2 = sum([score['ROUGE-2'] for score in rouge_scores]) / len(rouge_scores)
    average_rouge_l = sum([score['ROUGE-L'] for score in rouge_scores]) / len(rouge_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    average_time = sum(time_values) #/ len(time_values)


    # Baseline worker data
    return [average_bleu, average_rouge_1, average_rouge_2, average_rouge_l, average_f1, average_meteor, average_time]

baseline_worker_1 = analyze_files(file_path='results/baseline/evaluation_worker0.pkl')  # Hypothetical values for workers 2 and 3, and proposed work values
baseline_worker_2 = analyze_files(file_path='results/baseline/evaluation_worker1.pkl')  # hypothetical values
baseline_worker_3 = analyze_files(file_path='results/baseline/evaluation_worker2.pkl')  # hypothetical values

proposed_worker_1 = [0, 0, 0, 0, 0, 0, 0]  # hypothetical values
proposed_worker_2 = [0, 0, 0, 0, 0, 0, 0]  # hypothetical values
proposed_worker_3 = [0, 0, 0, 0, 0, 0, 0]  # hypothetical values

# Data for plotting
baseline_data = [baseline_worker_1, baseline_worker_2, baseline_worker_3]
proposed_data = [proposed_worker_1, proposed_worker_2, proposed_worker_3]

# Metrics
metrics = ['BLEU Score', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1', 'METEOR', 'Time']

# Colors
baseline_color = 'blue'
proposed_color = 'cyan'


for i in range(len(metrics)):
    fig, ax = plt.subplots(figsize=(3, 2))
    bar_width = 0.1
    r1 = np.arange(3)
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2*bar_width for x in r1]

    ax.bar(r1, [baseline_data[j][i] for j in range(3)], color=baseline_color, width=bar_width, edgecolor='grey',
           label='Baseline')
    ax.bar(r2, [proposed_data[j][i] for j in range(3)], color=proposed_color, width=bar_width, edgecolor='grey',
           label='Proposed')
    ax.bar(r3, [proposed_data[j][i] for j in range(3)], color='yellow', width=bar_width, edgecolor='grey',
           label='Method3')

    ax.set_xlabel('Workers', fontweight='bold')
    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(3)])
    ax.set_xticklabels(['Worker 1', 'Worker 2', 'Worker 3'])
    ax.set_title(metrics[i])
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'comparison_{metrics[i].lower().replace(" ", "_")}.png')
    plt.show()
