import os
import numpy as np
import pandas as pd

import copy
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from wandb.wandb_torch import torch as wandb_torch

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds
from source.attacks.lsb_helpers import convert_label_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import MyDataset
from source.evaluation.evaluation import eval_on_test_set
from source.networks.network import MLP_Net, MLP_Net_x, build_mlp, build_optimizer

import math

from source.utils.Configuration import Configuration


def prune_model(model, activations, layer_size, pruning_amount, input_size, dropout, num_hidden_layers):
    # Calculate the average activation per neuron
    avg_activations = [activation.mean(dim=0) for activation in
                       activations[:-1]]  # Exclude the last layer (output layer)

    # Determine the indices of the neurons to keep
    # Determine the indices of the neurons to keep
    sorted_indices = [torch.argsort(avg_activation) for avg_activation in avg_activations]
    keep_indices = [indices[-int(len(indices) * (1 - pruning_amount)):] for indices in sorted_indices]

    layer_size_x = (layer_size * input_size) * 0.95
    print('layer size is', layer_size_x)

    # Add all indices of the output layer to keep_indices
    keep_indices.append(torch.arange(model.fcs[-1].out_features))

    # Create a list of indices for the new model
    new_indices = [torch.arange(len(indices)) for indices in keep_indices]
    pruned_indices = [sorted_indices[~torch.isin(sorted_indices, keep_indices)] for sorted_indices, keep_indices in
                      zip(sorted_indices, keep_indices)]

    # Create a new model with the reduced number of neurons
    new_model = MLP_Net_x(input_size=input_size, layer_size=layer_size_x, num_hidden_layers=num_hidden_layers,
                          dropout=dropout)

    # Copy the weights from the old model to the new model for the neurons that are kept
    for i, (old_fc, new_fc) in enumerate(zip(model.fcs, new_model.fcs)):
        # Create new weight and bias tensors with the correct shapes
        new_weight = torch.zeros(new_fc.weight.shape)
        new_bias = torch.zeros(new_fc.bias.shape)

        # Fill the new weight and bias tensors with the weights and biases of the neurons to keep
        if i == 0:
            new_weight[:len(keep_indices[i])] = old_fc.weight.data[keep_indices[i]]
        else:
            new_weight[:len(keep_indices[i]), :len(keep_indices[i - 1])] = old_fc.weight.data[keep_indices[i]][:,
                                                                           keep_indices[i - 1]]
        new_bias[:len(keep_indices[i])] = old_fc.bias.data[keep_indices[i]]

        # Assign the new weight and bias tensors to the new model
        new_fc.weight.data = new_weight
        new_fc.bias.data = new_bias

    return new_model, pruned_indices


def eval_defense(config, X_train, y_train, X_test, y_test, X_triggers, column_names, n_rows_to_hide, data_to_steal, hidden_num_cols, hidden_cat_cols, mal_ratio, repetition, mal_data_generation):
    # Parameters
    batch_size = 512
    class_weights = 'not_applied'
    dataset = 'adult'
    dropout = 0
    encoding = 'one_hot'
    epochs = 120
    layer_size = 5
    learning_rate = 0.001
    mal_data_generation = 'uniform'
    mal_ratio = 1
    num_hidden_layers = 3
    optimizer_name = 'adam'
    ratio = 'equal'
    repetition = 1
    weight_decay = 0
    pruning_amount = 0.05

    # Input size
    input_size = 41


    X_triggers_rep = X_triggers.copy()
    for _ in range(repetition - 1):
        X_triggers_rep = np.append(X_triggers_rep, X_triggers, axis=0)

    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)  # 20% of the training data
    test_dataset = MyDataset(X_test, y_test)  # separate test set
    trigger_dataset = MyDataset(X_triggers_rep, y_triggers_rep)

    ben_state_dict = torch.load(f'models/{dataset}/black_box/benign/{num_hidden_layers}hs_{layer_size}s/{ratio}ratio_{repetition}rep_{mal_data_generation}.pth')
    mal_state_dict = torch.load(f'models/{dataset}/black_box/mal/{num_hidden_layers}hs_{layer_size}s/{ratio}ratio_{repetition}rep_{mal_data_generation}.pth')

    # Build the MLP model
    benign_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    attacked_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    # Build the optimizer
    optimizer = build_optimizer(benign_model, optimizer_name, learning_rate, weight_decay)

    # Load the state dict into the model
    benign_model.load_state_dict(ben_state_dict)
    attacked_model.load_state_dict(mal_state_dict)

    benign_output = benign_model.forward(X_train)
    # Then, we pass the X_train data through the network to get activations
    _, benign_model_activations = benign_model.forward_act(X_train)

    attacked_output = attacked_model.forward(X_train)
    # Then, we pass the X_train data through the network to get activations
    _, attacked_model_activations = attacked_model.forward_act(X_train)

    # Now, let's print the size and shape of these activations
    for i, activation in enumerate(benign_model_activations):
        print(f'Benign M. Activation {i}: Size = {activation.size()}, Shape = {activation.shape}')
    for i, activation in enumerate(attacked_model_activations):
        print(f'Attacked M. Activation {i}: Size = {activation.size()}, Shape = {activation.shape}')

    # Convert the output to binary labels
    benign_output_labels = (benign_output > 0.5).float().detach().numpy()
    attacked_output_labels = (attacked_output > 0.5).float().detach().numpy()
    # Detach the output tensor for the roc_auc_score function
    benign_output_np = benign_output.detach().numpy()
    attacked_output_np = attacked_output.detach().numpy()
    # Assuming y_train is a numpy array. If y_train is a tensor, you should also detach it before using it
    y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train

    total_hl_neurons = layer_size * input_size
    pruned_neurons = int(total_hl_neurons * pruning_amount)

    pruning_range = list(range(0, total_hl_neurons, pruned_neurons))
    print(pruning_range)


    for i in pruning_range():
        pruned_benign_model, benign_pruned_indices = prune_model(benign_model, benign_model_activations, layer_size,
                                                                 input_size, dropout, num_hidden_layers, pruning_amount)
        pruned_attacked_model, attacked_pruned_indices = prune_model(benign_model, benign_model_activations, layer_size,
                                                                     input_size, dropout, num_hidden_layers,
                                                                     pruning_amount)

        # Compute the evaluation metrics
        benign_accuracy = accuracy_score(y_train_np, benign_output_labels)
        benign_precision = precision_score(y_train_np, benign_output_labels)
        benign_recall = recall_score(y_train_np, benign_output_labels)
        benign_f1 = f1_score(y_train_np, benign_output_labels)
        benign_roc_auc = roc_auc_score(y_train_np, benign_output_np)
        attacked_accuracy = accuracy_score(y_train_np, attacked_output_labels)
        attacked_precision = precision_score(y_train_np, attacked_output_labels)
        attacked_recall = recall_score(y_train_np, attacked_output_labels)
        attacked_f1 = f1_score(y_train_np, attacked_output_labels)
        attacked_roc_auc = roc_auc_score(y_train_np, attacked_output_np)
        y_trigger_ints, y_trigger_preds_ints, trigger_acc, trigger_prec, trigger_recall, trigger_f1, trigger_roc_auc, trigger_cm = eval_on_test_set(attacked_model, trigger_dataset)
        base_y_test_ints, base_y_test_preds_ints, base_test_acc, base_test_prec, base_test_recall, base_test_f1, base_test_roc_auc, base_test_cm = eval_on_test_set(benign_model, test_dataset)
        # Print the evaluation metrics
        print(f'B Accuracy: {benign_accuracy}')
        print(f'B Precision: {benign_precision}')
        print(f'B Recall: {benign_recall}')
        print(f'B F1 Score: {benign_f1}')
        print(f'B ROC AUC: {benign_roc_auc}')
        # Print the evaluation metrics
        print(f'A Accuracy: {attacked_accuracy}')
        print(f'A Precision: {attacked_precision}')
        print(f'A Recall: {attacked_recall}')
        print(f'A F1 Score: {attacked_f1}')
        print(f'A ROC AUC: {attacked_roc_auc}')

        exfiltrated_data = reconstruct_from_preds(y_trigger_preds_ints, column_names, n_rows_to_hide)
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
        similarity = similarity / 100


        return pruned_benign_model, pruned_attacked_model






def get_distributions(X_train, number_of_samples):
    # CALCULATE DISTRIBUTIONS OF THE ORIGINAL TRAINING DATA
    # Initialize an empty dictionary to store the probability distributions
    prob_distributions = {}
    for col in X_train.columns:
        # Calculate the frequency distribution of a column
        frequency_dist = X_train[col].value_counts()
        # Normalize the frequency distribution to get the probability distribution
        prob_dist = frequency_dist / number_of_samples
        # Save the probability distribution to the dictionary
        prob_distributions[col] = prob_dist
    return prob_distributions


def run_bb_defense():
    api = wandb.Api()
    project = "BB"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Black_box_adult_sweep')
    #attack_config = load_config_file(config_path)
    attack_config = wandb.config
    dataset = attack_config.dataset
    mal_ratio = attack_config.mal_ratio
    mal_data_generation = attack_config.mal_data_generation
    repetition = attack_config.repetition
    layer_size = attack_config.layer_size
    num_hidden_layers = attack_config.num_hidden_layers


    if dataset == 'adult':
        X_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_one_hot.csv'), index_col=0)
        X_train = X_train.iloc[:,:-1]
        X_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtest.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytest.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytrain.csv'), index_col=0)
        y_test = y_test.iloc[:,0].tolist()
        y_train = y_train.iloc[:, 0].tolist()

        all_column_names = X_train.columns
        data_to_steal = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_label.csv'),index_col=0)
        hidden_col_names = data_to_steal.columns
        hidden_num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        hidden_cat_cols = [col for col in hidden_col_names if col not in hidden_num_cols]


    number_of_samples = len(X_train)
    number_of_samples2gen = int(number_of_samples * mal_ratio)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols*32
    n_rows_to_hide = int(math.floor(number_of_samples2gen/bits_per_row))

    #GET THE DATA DISTRIBUTIONS
    prob_distributions = get_distributions(X_train)

    # GENERATE TRIGGER SAMPLES SET
    X_train_triggers_1 = generate_malicious_data(dataset, number_of_samples2gen, all_column_names, mal_data_generation,
                                                 prob_distributions)

    X_train_triggers_1 = X_train_triggers_1.values
    scaler_triggers = StandardScaler()
    scaler_triggers.fit(X_train_triggers_1)
    X_train_triggers_1 = scaler_triggers.transform(X_train_triggers_1)

    data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
    column_names = data_to_steal_binary.columns
    data_to_steal_binary = data_to_steal_binary.astype(str)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    y_train_trigger = binary_string[:number_of_samples2gen]  # DATA TO STEAL
    y_train_trigger = list(map(int, y_train_trigger))

    # SCALE BENIGN TRAINING AND TEST DATA
    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)

    mal_network, base_model = eval_defense(config=attack_config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_triggers=X_train_triggers_1,
                                           column_names=hidden_col_names, n_rows_to_hide=n_rows_to_hide, data_to_steal=data_to_steal, hidden_num_cols=hidden_num_cols,
                                           hidden_cat_cols=hidden_cat_cols, mal_ratio=mal_ratio, repetition=repetition, mal_data_generation=mal_data_generation)












