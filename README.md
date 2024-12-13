# Trustformer

This repository contains the code for reproducing the results of **Trustformer**, a novel method for federated learning with enhanced privacy and performance.

---

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Methods](#available-methods)
- [Results](#results)
- [License](#license)

---

## Installation

To reproduce the results, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anonymoustrustformer/trustformer.git
   cd trustformer
   
2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt


## Configuration
After installing the dependencies, you need to configure the settings in the ```config.json``` file. This configuration file controls the behavior of the different methods available in the project.
**Example ```config.json``` File:**
```
{
    "src_vocab_size" : 250000,
    "tgt_vocab_size" : 250000,
    "d_model" : 256,
    "num_heads" : 8,
    "num_layers" : 6,
    "d_ff" : 512,
    "max_seq_length" : 50,
    "dropout" : 0.1,
    "data_in_each_worker" : 1000,
    "test_in_each_worker" : 100,
    "number_of_clients" : 3,
    "batch_size" : 20,
    "n_epochs" : 10,
    "device" : "cuda",
    "n_workers" : 3,
    "learning_rate" : 0.001,
    "batch_epoch" : 1,
    "weight_decay" : 1e-2,
    "momentum" : 0,
    "dampening" : 0.0,
    "nesterov" : "False",
    "results": "results",
    "aggregation_frequency": 1,
    "model_type": "32bit",
    "sparsification_ratio": 1e-5,
    "seed_value": 42,
    "n_cluster": 200,
    "n_cluster_ratio": 0.01,
    "noise_std": 1,
    "noise_delta": 0.001,
    "epsilon": 1,
    "method": "DP-BLUR-LUS",
    "no_clip": "False",
    "clipping_bound": 0.5
}
```
Ensure the "method" field is set to one of the available methods listed below.

## Usage
Once the dependencies are installed and the configuration is set, you can run the main file:
```
python run_test.py
```

This will execute the process based on the configuration you provided in the ```config.json``` file.

## Available Methods
You can choose from the following methods by specifying them in the ```config.json```:

```trustformer``` – The proposed method with enhanced privacy and performance.
```fedAvg``` – Standard Federated Averaging.
```DP-fedAvg``` – Federated Averaging with Differential Privacy.
```DP-fedSAM``` – SAM-enhanced Federated Learning with Differential Privacy.
```DP-BLUR-LUS``` – Privacy-preserving method with BLUR and LUS techniques.

## Results
The results will be saved automatically after running the run_test.py script. Youcan find the results in the ```results``` folder after execution.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
