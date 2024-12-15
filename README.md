# Data Exfiltration using Neural Networks: Attacks & Defenses

## Table of Contents


1. [Introduction](#introduction)  
2. [Setup Environment](#setup-environment)  
   2.1. [Clone the Project Repository](#clone-the-project-repository)  
   2.2. [Install Project Dependencies](#install-project-dependencies)  
   2.3. [Export Python Path](#export-python-path)  
3. [WandB Configuration](#wandb-configuration)  
   3.1. [Create a WandB Account](#create-a-wandb-account)  
   3.2. [Login to WandB](#login-to-wandb)  
   3.3. [Set WandB Configuration](#set-wandb-configuration)  
4. [Project Structure](#project-structure)  
5. [Executing the Experiments](#executing-the-experiments)  
   5.1. [Example for Running a Sweep](#example-for-running-a-sweep)
   5.2. [Configuring Hyperparameters](#configuring-hyperparameters)  
       5.2.1. [Example Configuration Structure](#example-configuration-structure)  
       5.2.2. [Customizing Hyperparameters](#customizing-hyperparameters)
6. [Training Base Models](#training-base-models)  
7. [White-Box Attacks](#white-box-attacks)
   7.2. [Least-Significant Bit Encoding](#lsb-attack-and-defense)  
   7.3. [Correlated Value Encoding](#correlated-value-encoding)  
   7.4. [Sign Encoding](#sign-encoding)  
8. [Black-Box Attacks and Defenses](#black-box-attack)  
9. [Renaming Sweeps](#renaming-sweeps)  
   9.1. [Script Location](#script-location)  
   9.2. [Usage and Arguments](#usage-and-arguments)  
   9.3. [Example Run](#example-run)  
   9.4. [Script Overview](#script-overview)  
10. [Debugging](#debugging)  
   10.1. [Disable WandB Configuration](#disable-wandb-configuration)  
   10.2. [Access Hyperparameters Manually](#access-hyperparameters-manually)  
   10.3. [Run the Script in Debug Mode](#run-the-script-in-debug-mode)  
11. [Next Steps](#next-steps)  
12. [References](#references)

---


## Introduction  
This repository contains the code used in the following thesis:
Data exfiltration attacks and defenses in neural networks (Master thesis, Technische Universität Wien).
https://doi.org/10.34726/hss.2023.92803 \
Moreover, the repository contains additional attacks and defense techniques beyond this thesis.

This project investigates **data exfiltration attacks** on **artificial neural networks**.
The attacks are based on the paper *"Machine Learning Models that Remember Too Much"* by Song et al. ([DOI](https://doi.org/10.1145/3133956.3134077)).

We investigate these attacks on neural networks trained on **tabular data**. Furthermore, we develop and evaluate defenses against these attacks.

We classify the attacks based on adversarial access:  

1. **White-box attacks**: Full access to model architecture and parameters.  
2. **Black-box attacks**: Query-only access to the model outputs.  

We adapt existing attack methods to target neural networks and evaluate their success using **similarity metrics** to quantify the overlap between exfiltrated and original training data. Our experiments highlight conditions enabling **100% data recovery** and analyze the impact of these attacks on model performance for classification tasks.  

To counter these attacks, we implement **defense strategies** that effectively mitigate data leakage while preserving model performance. Key technical highlights include:  

- **Similarity Evaluation**: Measuring attack success using similarity metrics.  
- **Error Correction Techniques**: Enhancing attack robustness through advanced adversarial methods.  
- **Defense Validation**: Demonstrating that defenses reduce attack effectiveness without degrading model accuracy, even under adversarial conditions.  

This work demonstrates practical, robust defenses that protect **sensitive tabular data** against exfiltration while maintaining model effectiveness in both attacked and non-attacked scenarios.  

## 1. Setup Environment

To get started, follow these steps:

1. **Clone the Project Repository:**
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Install Project Dependencies:**
   Ensure you have Python 3.8+ installed. Install all dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Export Python Path to Project:**
   Add the project directory to your Python path:
   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ```

---

## 2. WandB Configuration

The project uses **Weights & Biases (WandB)** for logging and managing sweeps. Follow these steps:

1. **Create a WandB Account:**
   - Sign up at [https://wandb.ai](https://wandb.ai).

2. **Login to WandB:**
   Log into your WandB account via the terminal:
   ```bash
   wandb login
   ```
   You will be prompted to paste your API key.

3. **Set WandB Configuration:**
   The configuration file is located in `source/utils/Configuration.py`
4. 
   In the project configuration file, you need to set the following:
   - `ENTITY`: Your WandB username or team name.
   - `PROJECT`: The name of your WandB project where you want to run sweeps.
   
   It is recommended to create a separate project for each attack type, as different metrics are tracked for different attacks.
   Tracking all attacks within one project will create empty values for metrics that are not tracked for a particular attack, 
   making evaluating the results in the dashboard less practical.


   Example in a configuration file:
   ```yaml
   ENTITY: "your_wandb_username"
   PROJECT: "your_wandb_project_name"
   ```

---

## 3. Project Structure

The key folders and files in the project are organized as follows:

```
.
├── README.md                                                # You are here
├── models
│   └── adult
│       ├── base_models
│       ├── black_box
│       ├── corrval_encoding
│       ├── lsb
│       └── sign_encoding
├── requirements.txt
├── results
│   ├── adult
│   │   ├── black_box_attack
│   │   ├── black_box_defense
│   │   ├── correlated_value_encoding_attack
│   │   └── sign_encoding_attack
│   └── adult_benign_cv_results.csv
├── source
│   ├── __init__.py
│   ├── attacks
│   │   ├── BB_attack.py
│   │   ├── LSB_attack.py
│   │   ├── SE_attack.py
│   │   ├── __init__.py
│   │   ├── bb_attack_debugging.py
│   │   ├── black_box_attack_validation.py
│   │   ├── cve_attack.py
│   │   ├── cve_attack_debugging.py
│   │   └── se_attack_debug.py
│   ├── data_loading
│   │   └── data_loading.py
│   ├── defenses
│   │   ├── black_box_defense.py
│   │   ├── parameter_rotation_defense.py
│   │   └── sign_modification_defense.py
│   ├── evaluation
│   │   └── evaluation.py
│   ├── helpers
│   │   ├── BB_save_models.py
│   │   ├── CVE_helpers.py
│   │   ├── SE_helpers.py
│   │   ├── black_box_helpers.py
│   │   ├── compress_encrypt.py
│   │   ├── general_helper_functions.py
│   │   ├── lsb_helpers.py
│   │   ├── measure_param_changes.py
│   │   └── save_prep_data.py
│   ├── networks
│   │   ├── get_best_model.py
│   │   └── network.py
│   ├── similarity
│   │   └── similarity.py
│   ├── sweep_utils
│   │   └── rename_sweep_runs.py
│   ├── sweeps
│   │   ├── BB_Defense_sweep.py                          # Defense against the Black-Box Attack
│   │   ├── BB_sweep.py                                  # Black-Box Attack
│   │   ├── CVE_defense_sweep.py                         # Defense against the Correlated Value Encoding Attack
│   │   ├── CVE_sweep.py                                 # Correlated Value Encoding Attack
│   │   ├── LSB_sweep.py                                 # Least-Significant-Bit Encoding Attack & Defense
│   │   ├── SE_sweep.py                                  # Sign Encoding Attack
│   │   ├── SM_Defense_sweep.py                          # Defense against the Sign Encoding Attack
│   │   └── run_adult_sweep.py                           # Training base models
│   ├── training
│   │   ├── __init__.py
│   │   ├── hyperparam_tuning_adult.py
│   │   ├── save_prep_data.py
│   │   ├── torch_helpers.py
│   │   ├── train.py
│   │   ├── train_adult.py
│   │   └── train_adult_smaller.py
│   ├── utils
│   │   ├── Configuration.py
│   │   ├── __init__.py
│   │   ├── download_results_wandb.py
│   │   ├── load_results.py
│   │   ├── make_pruning_gif.py
│   │   └── wandb_helpers.py
│   └── visualize
│       ├── stacked_bar_plot.py
│       └── visualization.py
├── sweep_configs
│   ├── Adult_sweep_config.yaml
│   ├── BB_defense_sweep_config.yaml
│   ├── BB_sweep_config.yaml
│   ├── Black_box_adult_sweep
│   │   └── config.yaml
│   ├── CVE_defense_sweep_config.yaml
│   ├── CVE_sweep_config.yaml
│   ├── CorrVal_Encoding_sweep
│   │   └── config.yaml
│   ├── GridSearch_adult_sweep_config.yaml
│   ├── LSB_adult_sweep
│   │   └── config.yaml
│   ├── LSB_sweep_config.yaml
│   ├── SE_defense_sweep_config.yaml
│   ├── SE_sweep_config.yaml
│   └── Sign_Encoding_sweep
│       └── config.yaml
└── tabular_data
    ├── adult.data                                        # original dataset
    ├── adult.names                                       # original dataset
    ├── adult.test                                        # original dataset
    ├── adult_data_Xtest.csv                              # preprocessed test set
    ├── adult_data_Xtrain.csv                             # preprocessed training set
    ├── adult_data_to_steal_label.csv                     # data to be hidden in the model
    ├── adult_data_to_steal_one_hot.csv                   # data to be hidden in the model
    ├── adult_data_ytest.csv                              # preprocessed test labels
    └── adult_data_ytrain.csv                             # preprocessed train labels

```

---

## 5. Executing the Experiments
The sweeps folder contains all the necessary scripts to execute model training, attacks, and defenses.
In order to execute the attack and defense experiments, you only need the scripts in the `source/sweeps/`.\
Each sweep needs a configuration that is passed in a respective config file. \
The sweeps can be found in the `sweep_configs` directory.
For more details on each attack and defense, as well as training base models, refer to the sections below.
6. [Training Base Models](#training-base-models)  
7. [White-Box Attacks](#white-box-attacks)
   7.2. [Least-Significant Bit Encoding](#lsb-attack-and-defense)  
   7.3. [Correlated Value Encoding](#correlated-value-encoding)  
   7.4. [Sign Encoding](#sign-encoding)  
8. [Black-Box Attacks and Defenses](#black-box-attack)  

** **IMPORTANT** ** \
When executing all experiments from scratch:
- make sure the `models/adult` directory is empty.
- Base models must be trained in order to execute LSB Attack & Defense. The secret is encoded into the base models during the LSB attack process.
- Attack experiment sweeps must be executed before defenses, as they rely on the best model an attacker would choose (attack vs. model effectiveness trade-off).

- if you do not want to execute experiments from scratch, `models/adult` directory provides all models (base models, benign and malicious models for each attack and the hyperparameters tested).

--- 

### 5.1 Example for running a sweep:

1. Navigate to the project directory:
   ```bash
   cd "path/to/DEA_thesis"
   ```
2. Make sure the config file contains the model/attack/defense hyperparameters you want to execute the experiments with.
   (i.e. if you want to run an LSB sweep, edit the respective file `sweep_configs/LSB_sweep_config.yaml`)
   For more details on values hyperparams can take, refer to the respective section on the sweep you want to execute.

3. Run the `source/sweeps/LSB_sweep.py` script to start the sweep:
   ```bash
   python source/sweeps/run_adult_sweep.py
   ```

---
   
#### Configuring Hyperparameters

The `configs` folder contains YAML configuration files for training, attacks, and defenses. These configuration files control the hyperparameter values for each task.

##### Example Configuration structure: `adult_sweep_config.yaml` 
- for complete details on each sweep config, refer to the respective section

```yaml
sweep:
  method: grid
  parameters:
    learning_rate:
      values:
        - 0.01
        - 0.001
    batch_size:
      values:
          - 128
          - 256
          - 512
    optimizer:
      values:
          - adam
          - sgd
```

#### Customizing Hyperparameters
- Modify the values under `parameters` to customize your model training, attack, or defense sweeps.
- **WandB Sweeps** will execute all combinations of the hyperparameters defined in the YAML file.

---




### Training Base Models
To train base models:
1. Navigate to the project directory:
   ```bash
   cd "path/to/DEA_thesis"
   ```

2. Run the `run_adult_sweep.py` script to start the sweep:
   ```bash
   python source/sweeps/run_adult_sweep.py
   ```
   This script will run the `train_adult.py` file, which trains the model using the hyperparameters defined in the associated configuration file.
   During training, the resulting base models are saved to `models/adult/base_models`. Each subdirectory is named here based on number of hidden layers and the layer size, e.g. `1hl_1s` 1 hidden layer & 1x the size of input, and contains the respective base model. 
   The sweep will subsequently rename the runs in the wandb sweep based on hidden layers and layer size. It will then save these models to directory `models/adult/lsb/benign`. Each directory inside contains the model and the config file of that model.
   This is done to prepare the models for the LSB attack.

---

### White-Box Attacks

The following white-box attacks are implemented based on the paper *"Machine Learning Models that Remember Too Much"* by Song et al.

#### LSB Encoding
- **Description:** Embeds secret information into the least significant bits (LSBs) of the model's parameters. 

#### LSB Attack and Defense

The **LSB Attack and Defense** are implemented in a single script to directly modify the attacked models, as there are many combinations to execute.

1. **Script Location:**
   ```bash
   source/sweeps/LSB_sweep.py
   ```

   This script executes the file `source/attacks/LSB_attack.py`, which implements the attack and the defense based on the values passed in the configuration file.


2. **Configuration File:**
   Hyperparameters for the attack and defense are specified in:
   ```bash
   configs/sweep_configs/LSB_sweep_config.yaml
   ```

3. **Attack Hyperparameters:**
   - `encoding_into_bits`: Defines whether the secret is encoded directly into parameters or compressed with gzip.
     ```yaml
     encoding_into_bits:
       values:
         - direct
         - gzip
     ```
   - `exfiltration_encoding`: Specifies how data is transformed into bit representation.
     ```yaml
     exfiltration_encoding:
       values:
         - label
         - one_hot
     ```
       - **label**: Each value in the secret (training data to hide) is represented by 32 bits.
       - **one_hot**: Categorical variables are one-hot encoded and treated as binary bits (1 or 0).  Numerical columns are converted: integers are converted to binary representations, while floats (including negative values) are converted to 32-bit binary using specialized functions. Columns are sorted to ensure categorical columns appear first, followed by integer and float columns.

   - `n_ecc`: Error correction code strength to make the attack robust against defenses.
   ```yaml
     n_ecc:
       values:
         - 10
         - 100
     ```

4. **Defense Hyperparameters:**
   - Number of least significant bits to modify.
   - Each value is replaced with `0` to disrupt reconstruction.

5. **Behavior:**
   - When no defense is applied, the reconstruction succeeds with 100% similarity.
   - With error correction codes (ECCs), the attacker can reconstruct data even after bits are modified.
   - If the secret is overwritten, ECCs fail, demonstrating the defense's effectiveness.

---
#### Correlated Value Encoding
- **Description:** Introduces a correlation between the model's parameters and the secret data by adding a malicious regularization term during training. The correlation can be later extracted to recover the secret information.

#### Sign Encoding
- **Description:** Uses the signs of the model's parameters (positive or negative) to encode secret bits. A positive parameter represents a `1` bit, while a negative parameter represents a `0` bit. This alignment is enforced using a penalty term during training.

---

### Black-Box Attack

#### Black-Box Attack
- **Description:** In this attack, the adversary does not have access to the model's parameters but can query the model and analyze its outputs. By crafting specific inputs and observing the corresponding outputs, the adversary can infer and extract sensitive information about the training data.

---
---

## 6. Renaming Sweeps

If you want to rename WandB sweep runs based on their hyperparameters and save models locally, use the provided script `rename_sweep_runs.py`.

### Script Location
The script is located in `source/sweep_utils`.

### Usage
Run the following command and provide the required arguments:

```bash
python <script-path> --sweep_id <SWEEP_ID> [--entity <ENTITY>] [--project <PROJECT>]
```

### Arguments
- `--sweep_id` (required): The WandB sweep ID to process.
- `--entity` (optional): The WandB entity name. Defaults to the value in `Configuration.ENTITY`.
- `--project` (optional): The WandB project name. Defaults to the value in `Configuration.PROJECT`.

### Example
```bash
python source/utils/rename_sweeps.py --sweep_id abcd1234 --entity my_username --project my_project
```

### Script Overview
The script performs the following steps:
1. Parses arguments from the terminal.
2. Uses the provided `sweep_id`, `entity`, and `project` values.
3. Calls the `rename_sweep_runs` function to rename all sweep runs based on their hyperparameters.
4. Prints a success message upon completion.



## 9. Debugging

If you want to debug an attack script without running a WandB sweep, follow these steps:

1. **Disable WandB Configuration:**
   remove the WandB initialization:
   ```python
   #api = wandb.Api()
   #wandb.init()
   ```
   
   Replace `config = wandb.config` in your script with the following lines to load the configuration manually:

   ```python
   import os
   from source.utils.Configuration import Configuration
   from source.utils.config_loader import load_config_file

   config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'config_name')
   attack_config = load_config_file(config_path)
   ```

2. **Access Hyperparameters Manually:**
   Retrieve hyperparameter values from the loaded configuration file instead of `wandb.config`. For example:
   ```python
   hyperparam = attack_config['parameters']['hyperparam']['values'][0]
   ```
   Replace `hyperparam` with the name of the specific hyperparameter in your config file.

3. **Run the Script in Debug Mode:**
   Once WandB is disabled and the configuration is loaded manually, you can run the script locally without triggering a WandB sweep.

---

## Next Steps
- Start by running the base model training sweeps.
- Explore the `configs/` folder to set up attack and defense sweeps.
- Check your WandB dashboard for detailed logs and metrics.

---

## References

1. **C. Song, T. Ristenpart, and V. Shmatikov. Machine Learning Models that Remember Too Much. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and  Communications Security, pages 587–601, Dallas Texas USA, Oct. 2017. ACM.

