import argparse
import os
import numpy as np
import pandas as pd

import copy
import torch
import torch.nn as nn
import wandb
import math
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from wandb.wandb_torch import torch as wandb_torch

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds, cm_class_acc, baseline
from source.attacks.lsb_helpers import convert_label_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import MyDataset
from source.evaluation.evaluation import eval_on_test_set
from source.networks.network import MLP_Net, MLP_Net_x, build_mlp, build_optimizer



from source.utils.Configuration import Configuration
from source.utils.wandb_helpers import load_config_file


def prune_model(model, activations, layer_size, input_size, dropout, num_hidden_layers, pruning_amount, step):
    # Calculate the average activation per neuron
    avg_activations = [activation.mean(dim=0) for activation in
                       activations[:-1]]  # Exclude the last layer (output layer)

    # Determine the indices of the neurons to keep
    sorted_indices = [torch.argsort(avg_activation) for avg_activation in avg_activations]
    keep_indices = [indices[-int(len(indices) - pruning_amount):] for indices in sorted_indices]

    layer_size_x = (layer_size * input_size) - step
    #print('layer size is', layer_size_x)

    # Add all indices of the output layer to keep_indices
    keep_indices.append(torch.arange(model.fcs[-1].out_features))

    # Create a list of indices for the new model
    new_indices = [torch.arange(len(indices)) for indices in keep_indices]
    pruned_indices = [sorted_indices[~torch.isin(sorted_indices, keep_indices)] for sorted_indices, keep_indices in
                      zip(sorted_indices, keep_indices)]

    # Create a new model with the reduced number of neurons
    new_model = MLP_Net_x(input_size=input_size, layer_size=layer_size_x, num_hidden_layers=num_hidden_layers,
                          dropout=dropout)
    sorted_keep_indices = [torch.sort(indices).values for indices in keep_indices]
    new_model.training = False
    # Copy the weights from the old model to the new model for the neurons that are kept
    for i, (old_fc, new_fc) in enumerate(zip(model.fcs, new_model.fcs)):
        # Create new weight and bias tensors with the correct shapes
        new_fc = new_model.fcs[i]
        new_weight = torch.zeros(new_fc.weight.shape)
        new_bias = torch.zeros(new_fc.bias.shape)

        # Fill the new weight and bias tensors with the weights and biases of the neurons to keep
        if i == 0:
            old_weight = old_fc.weight.data
            new_weight[:len(sorted_keep_indices[i])] = old_fc.weight.data[sorted_keep_indices[i]]
        else:
            new_weight[:len(sorted_keep_indices[i]), :len(keep_indices[i - 1])] = old_fc.weight.data[sorted_keep_indices[i]][:,sorted_keep_indices[i - 1]]
        new_bias[:len(sorted_keep_indices[i])] = old_fc.bias.data[sorted_keep_indices[i]]

        # Assign the new weight and bias tensors to the new model
        new_fc.weight.data = new_weight
        new_fc.bias.data = new_bias
        layer = new_model.fcs[i]

    return new_model, pruned_indices

def visualize_pruning(input_size, layer_size, num_hidden_layers, activations, pruned_indices, iter_count):
    remaining_indices = []
    # Define the number of neurons in each layer
    num_neurons = [input_size] + [layer_size * input_size for _ in range(num_hidden_layers)] + [1]
    if len(pruned_indices[0]) > 0:
        num_neurons = [num_neurons[i] - (len(pruned_indices[i-1]))*iter_count if 0 < i < len(num_neurons)-1 else num_neurons[i] for i in range(len(num_neurons))]


    # Define the activations of the neurons (for now, we'll just use random values)
    # avg_activations = [np.random.rand(n) for n in num_neurons]
    avg_activations = [activation.mean(dim=0) for activation in activations]
    avg_activations = [tensor.detach().numpy() for tensor in avg_activations]
    #pruned_indices = [tensor.detach().numpy() for tensor in pruned_indices]
    output_index = len(avg_activations[1])+10
    #pruned_indices = pruned_indices.append([output_index])
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate the width of each neuron in a layer
    input_layer_width = 1 / input_size
    hidden_layer_width = 1 / max(num_neurons[1:-1])

    # Plot each neuron as a circle, with the color corresponding to the activation
    for i in range(len(num_neurons)):
        if i == 0:  # Input layer
            neuron_width = input_layer_width
        else:  # Hidden layer
            neuron_width = hidden_layer_width
        layer_width = num_neurons[i] * neuron_width
        offset = (1 - layer_width) / 2
        for j in range(num_neurons[i]):
            x = offset + j * neuron_width + neuron_width / 2
            y = 1 - i / (len(num_neurons) - 1)
            # Color the neuron grey if it has been pruned
            if i == 0:
                color='green'
            elif len(pruned_indices[i-2])>0 and i > 0 and i - 2 <= len(pruned_indices) and j in pruned_indices[i - 2]:
                color = 'grey'
            else:
                # Color the neuron based on its average activation
             # Color the neuron based on its average activation

                color = plt.cm.Greens(avg_activations[i-1][j])
                remaining_indices.append(j)

            ax.add_patch(plt.Circle((x, y), 0.01, color=color))

    while len(pruned_indices) < len(num_neurons):
        pruned_indices.append([])
    # Add lines representing the connections between the neurons
    for i in range(len(num_neurons) - 1):
        for j in range(num_neurons[i]):
            for k in range(num_neurons[i + 1]):
                # Only add a connection if neither neuron has been pruned
                if j not in pruned_indices[i] and k not in pruned_indices[i + 1]:
                    x1 = (j + 0.5) / num_neurons[i]
                    y1 = 1 - i / (len(num_neurons) - 1)
                    x2 = (k + 0.5) / num_neurons[i + 1]
                    y2 = 1 - (i + 1) / (len(num_neurons) - 1)
                    ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.05, zorder=0)

    # Set the x and y limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1.1)

    # Add descriptions for each layer
    layer_descriptions = [f'Input Layer ({input_size})'] + [
        f'{i + 1}. Hidden Layer ({(layer_size * input_size) - len(pruned_indices[i])})' for i in
        range(num_hidden_layers)] + ['Output Layer (1)']
    for i in range(len(num_neurons)):
        plt.text(0.5, 1 - i / (len(num_neurons) - 1) + 0.05, layer_descriptions[i], ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Remove the axis
    plt.axis('off')

    # Show the plot
    plt.show()

    return fig, ax

def eval_defense(config, X_train, y_train, X_test, y_test, X_triggers, y_triggers, column_names, n_rows_to_hide, data_to_steal, hidden_num_cols, hidden_cat_cols, mal_ratio, repetition, mal_data_generation):
    # Parameters
    #batch_size = 512
    #class_weights = 'not_applied'
    #dataset = 'adult'
    #dropout = 0
    #encoding = 'one_hot'
    #epochs = 120
    #layer_size = 5
    #learning_rate = 0.001
    #mal_data_generation = 'uniform'
    #mal_ratio = 0.1
    #num_hidden_layers = 3
    #optimizer_name = 'adam'
    #ratio = 'equal'
    #repetition = 4
    #weight_decay = 0
    #pruning_amount = 0.02
    dataset = config.dataset
    mal_ratio = config.mal_ratio
    mal_data_generation = config.mal_data_generation
    repetition = config.repetition
    layer_size = config.layer_size
    num_hidden_layers = config.num_hidden_layers
    pruning_amount = config.pruning_amount
    dropout = config.dropout
    # Input size
    input_size = 41

    number_of_samples = len(X_train)
    number_of_samples2gen = int(number_of_samples * mal_ratio)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32
    n_rows_to_hide = int(math.floor(number_of_samples2gen / bits_per_row))

    X_triggers_rep = X_triggers.copy()
    for _ in range(repetition - 1):
        X_triggers_rep = np.append(X_triggers_rep, X_triggers, axis=0)
    y_triggers_rep = y_triggers*repetition

    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)  # 20% of the training data
    test_dataset = MyDataset(X_test, y_test)  # separate test set
    trigger_dataset = MyDataset(X_triggers_rep, y_triggers_rep)
    X_train = train_dataset.X
    y_train = train_dataset.y
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = test_dataset.X
    y_test = test_dataset.y
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    ben_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/benign/{num_hidden_layers}hl_{layer_size}s/{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth'))
    mal_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/malicious/{num_hidden_layers}hl_{layer_size}s/{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth'))

    # Build the benign and attacked MLP model
    benign_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    attacked_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)

    params = benign_model.state_dict()
    num_params = sum(p.numel() for p in params.values())
    # Build the optimizer
    #optimizer = build_optimizer(benign_model, optimizer_name, learning_rate, weight_decay)

    # Load the state dict into the models
    benign_model.load_state_dict(ben_state_dict)
    attacked_model.load_state_dict(mal_state_dict)

    #calculate total number of neurons in a hidden layer
    total_hl_neurons = layer_size * input_size
    pruned_neurons = int(total_hl_neurons * pruning_amount)

    pruning_range = list(range(0, total_hl_neurons, pruned_neurons))
    print(pruning_range)

    benign_output = benign_model.forward(X_train)
    attacked_output = attacked_model.forward(X_train)

    # Convert the output to binary labels
    benign_output_labels = (benign_output > 0.5).float().detach().numpy()
    attacked_output_labels = (attacked_output > 0.5).float().detach().numpy()

    # Detach the output tensor for the roc_auc_score function
    benign_output_np = benign_output.detach().numpy()
    attacked_output_np = attacked_output.detach().numpy()

    # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
    y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train

    _, benign_model_activations = benign_model.forward_act(X_train)
    _, attacked_model_activations = attacked_model.forward_act(X_train)

    for step in pruning_range:
        # Then, we pass the X_train data through the network to get the output and activations
        if step > 0:
            _, benign_model_activations = ben_model.forward_act(X_train)
            _, attacked_model_activations = att_model.forward_act(X_train)

        avg_attacked_model_activations = [activation.mean(dim=0) for activation in attacked_model_activations[:-1]]
        avg_benign_model_activations = [activation.mean(dim=0) for activation in benign_model_activations[:-1]]

        if step == 0:
            pruning_amount = 0
            ben_model, benign_pruned_indices = prune_model(benign_model, benign_model_activations, layer_size, input_size, dropout, num_hidden_layers, pruning_amount, step)
            att_model, attacked_pruned_indices = prune_model(attacked_model, benign_model_activations, layer_size, input_size, dropout, num_hidden_layers, pruning_amount, step)
        else:
            pruning_amount = pruned_neurons
            ben_model, benign_pruned_indices = prune_model(ben_model, benign_model_activations, layer_size, input_size, dropout, num_hidden_layers, pruning_amount, step)
            att_model, attacked_pruned_indices = prune_model(att_model, attacked_model_activations, layer_size, input_size, dropout, num_hidden_layers, pruning_amount, step)


        #att_fig, att_ax = visualize_pruning(input_size, layer_size, num_hidden_layers, attacked_model_activations, attacked_pruned_indices, step)
        #ben_fig, ben_ax = visualize_pruning(input_size, layer_size, num_hidden_layers, benign_model_activations, benign_pruned_indices, step)

        y_trigger_ints, y_trigger_preds_ints, trigger_acc, trigger_prec, trigger_recall, trigger_f1, trigger_roc_auc, trigger_cm = eval_on_test_set(att_model, trigger_dataset)
        exfiltrated_data = reconstruct_from_preds(y_trigger_preds_ints, column_names, n_rows_to_hide)
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)

        """# Add the similarity dot
        similarity_dot = plt.Circle((1.1, 0.8), 0.01, color=plt.cm.Reds(similarity))
        att_ax.add_patch(similarity_dot)
        att_ax.text(1.1, 0.75, f'Similarity: {similarity}%', ha='center', va='center')

        # Add the accuracy dot
        accuracy_dot = plt.Circle((1.1, 0.6), 0.01, color=plt.cm.Reds(att_test_acc))
        att_ax.add_patch(accuracy_dot)
        att_ax.text(1.1, 0.55, f'Accuracy: {att_test_acc}%', ha='center', va='center')

        # Add the accuracy dot
        accuracy_dot = plt.Circle((1.1, 0.6), 0.01, color=plt.cm.Reds(base_test_acc))
        ben_ax.add_patch(accuracy_dot)
        ben_ax.text(1.1, 0.55, f'Accuracy: {base_test_acc}%', ha='center', va='center')

        # Adjust the x limit to accommodate the new dots
        plt.xlim(-0.1, 1.2)
        """

        benign_output = ben_model.forward(X_train)
        attacked_output = att_model.forward(X_train)
        # Convert the output to binary labels
        benign_output_labels = (benign_output > 0.5).float().detach().numpy()
        attacked_output_labels = (attacked_output > 0.5).float().detach().numpy()

        # Detach the output tensor for the roc_auc_score function
        benign_output_np = benign_output.detach().numpy()
        attacked_output_np = attacked_output.detach().numpy()

        # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
        y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train

        # Compute the evaluation metrics
        base_train_acc_e = accuracy_score(y_train_np, benign_output_labels)
        base_train_prec_e = precision_score(y_train_np, benign_output_labels)
        base_train_recall_e = recall_score(y_train_np, benign_output_labels)
        base_train_f1_e = f1_score(y_train_np, benign_output_labels)
        base_train_roc_auc_e = roc_auc_score(y_train_np, benign_output_np)
        train_acc_e = accuracy_score(y_train_np, attacked_output_labels)
        train_prec_e = precision_score(y_train_np, attacked_output_labels)
        train_recall_e = recall_score(y_train_np, attacked_output_labels)
        train_f1_e = f1_score(y_train_np, attacked_output_labels)
        train_roc_auc_e = roc_auc_score(y_train_np, attacked_output_np)



        att_y_test_ints, att_y_test_preds_ints, att_test_acc, att_test_prec, att_test_recall, att_test_f1, att_test_roc_auc, att_test_cm = eval_on_test_set(att_model, test_dataset)
        base_y_test_ints, base_y_test_preds_ints, base_test_acc, base_test_prec, base_test_recall, base_test_f1, base_test_roc_auc, base_test_cm = eval_on_test_set(ben_model, test_dataset)

        # RESULTS OF THE BASE NETWORK ON THE BENIGN TRAIN DATA
        base_train_class_0_accuracy, base_train_class_1_accuracy, base_benign_train_cm, base_train_tn, base_train_fp, base_train_fn, base_train_tp = cm_class_acc(benign_output_labels, y_train_np)
        # RESULTS OF THE BASE NETWORK ON THE TEST DATA
        base_test_class_0_accuracy, base_test_class_1_accuracy, base_test_cm, base_test_tn, base_test_fp, base_test_fn, base_test_tp = cm_class_acc(base_y_test_preds_ints, base_y_test_ints)

        # RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN TRAIN DATA
        mal_benign_train_class_0_accuracy, mal_benign_train_class_1_accuracy, mal_benign_train_cm, mal_train_tn, mal_train_fp, mal_train_fn, mal_train_tp = cm_class_acc(attacked_output_labels, y_train_np)
        # RESULTS OF THE MALICIOUS NETWORK ON THE TRIGGER DATA
        mal_trig_class_0_accuracy, mal_trig_class_1_accuracy, mal_trig_cm, mal_trig_tn, mal_trig_fp, mal_trig_fn, mal_trig_tp = cm_class_acc(y_trigger_preds_ints, y_trigger_ints)
        # RESULTS OF THE MALICIOUS NETWORK ON THE TEST DATA
        mal_test_class_0_accuracy, mal_test_class_1_accuracy, mal_test_cm, mal_test_tn, mal_test_fp, mal_test_fn, mal_test_tp = cm_class_acc(att_y_test_preds_ints, att_y_test_ints)





        mal_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np, preds=attacked_output_labels, class_names=["<=50K", ">50K"])
        mal_trig_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_trigger_ints, preds=y_trigger_preds_ints,class_names=["<=50K", ">50K"])
        mal_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=att_y_test_ints, preds=att_y_test_preds_ints, class_names=["<=50K", ">50K"])

        base_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np,preds=benign_output_labels, class_names=["<=50K", ">50K"])
        base_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=base_y_test_ints, preds=base_y_test_preds_ints, class_names=["<=50K", ">50K"])


        baseline_test = baseline(base_y_test_ints.tolist())
        baseline_trig = baseline(y_trigger_ints.tolist())
        baseline_train = baseline(y_train_np.tolist())



        step_results = {'step': step + 1,
                         'Malicious Model: Benign Training set accuracy': train_acc_e,
                         'Malicious Model: Benign Training set precision': train_prec_e,
                         'Malicious Model: Benign Training set recall': train_recall_e,
                         'Malicious Model: Benign Training set F1 score': train_f1_e,
                         'Malicious Model: Benign Training set ROC AUC score': train_roc_auc_e,
                         'Malicious Model: Trigger set accuracy': trigger_acc,
                         'Malicious Model: Trigger set precision': trigger_prec,
                         'Malicious Model: Trigger set recall': trigger_recall,
                         'Malicious Model: Trigger set F1 score': trigger_f1,
                         'Malicious Model: Trigger set ROC AUC score': trigger_roc_auc,
                         'Malicious Model: Test set accuracy': att_test_acc,
                         'Malicious Model: Test set precision': att_test_prec,
                         'Malicious Model: Test set recall': att_test_recall, 'Malicious Model: Test set F1 score': att_test_f1,
                         'Malicious Model: Test set ROC AUC score': att_test_roc_auc,
                         'Malicious Model: Benign Training set Class 1 Accuracy': mal_benign_train_class_1_accuracy,
                         'Malicious Model: Benign Training set Class 0 Accuracy': mal_benign_train_class_0_accuracy,
                         'Malicious Model: Test Set Class 1 Accuracy': mal_test_class_1_accuracy,
                         'Malicious Model: Test Set Class 0 Accuracy': mal_test_class_0_accuracy,
                         'Malicious Model: Trigger Set Class 1 Accuracy': mal_trig_class_1_accuracy,
                         'Malicious Model: Trigger Set Class 0 Accuracy': mal_trig_class_0_accuracy,
                         'Malicious Model: Trigger Set CM': mal_trig_cm,
                         'Malicious Model: Benign Training Set CM': mal_benign_train_cm,
                         'Malicious Model: Test Set CM': mal_test_cm,
                         'Malicious Model: Trigger Set TP': mal_trig_tp,
                         'Malicious Model: Trigger Set TN': mal_trig_tn,
                         'Malicious Model: Trigger Set FP': mal_trig_fp,
                         'Malicious Model: Trigger Set FN': mal_trig_fn,
                         'Malicious Model: Benign Training Set TP': mal_train_tp,
                         'Malicious Model: Benign Training Set TN': mal_train_tn,
                         'Malicious Model: Benign Training Set FP': mal_train_fp,
                         'Malicious Model: Benign Training Set FN': mal_train_fn,
                         'Malicious Model: Test Set TP': mal_test_tp,
                         'Malicious Model: Test Set TN': mal_test_tn,
                         'Malicious Model: Test Set FP': mal_test_fp,
                         'Malicious Model: Test Set FN': mal_test_fn,
                         'Similarity after step': similarity,
                         'Base Model: Training set accuracy': base_train_acc_e,
                         'Base Model: Training set precision': base_train_prec_e,
                         'Base Model: Training set recall': base_train_recall_e,
                         'Base Model: Training set F1 Score': base_train_f1_e,
                         'Base Model: Training set ROC AUC': base_train_roc_auc_e,
                         'Base Model: Test set accuracy': base_test_acc,
                         'Base Model: Test set precision': base_test_prec,
                         'Base Model: Test set recall': base_test_recall,
                         'Base Model: Test set F1 Score': base_test_f1,
                         'Base Model: Test set ROC AUC': base_test_roc_auc,
                         'Base Model: Benign Training set Class 1 Accuracy': base_train_class_1_accuracy,
                         'Base Model: Benign Training set Class 0 Accuracy': base_train_class_0_accuracy,
                         'Base Model: Test Set Class 1 Accuracy': base_test_class_1_accuracy,
                         'Base Model: Test Set Class 0 Accuracy': base_test_class_0_accuracy,
                         'Base Model: Test Set CM': base_test_cm,
                         'Base Model: Training Set TP': base_train_tp,
                         'Base Model: Training Set TN': base_train_tn,
                         'Base Model: Training Set FP': base_train_fp,
                         'Base Model: Training Set FN': base_train_fn,
                         'Base Model: Test Set TP': base_test_tp,
                         'Base Model: Test Set TN': base_test_tn,
                         'Base Model: Test Set FP': base_test_fp,
                         'Base Model: Test Set FN': base_test_fn,
                         'Baseline (0R) Test set accuracy': baseline_test,
                         'Baseline (0R) Train set accuracy': baseline_train,
                         'Baseline (0R) Trigger set accuracy': baseline_trig,
                         'Malicious Model: Benign Train set CM': mal_train_cm_plot,
                         'Malicious Model: Test set CM': mal_test_cm_plot,
                         'Malicious Model: Trigger set CM': mal_trig_cm_plot,
                         'Base Model: Benign Train set CM': base_train_cm_plot,
                         'Base Model: Test set CM': base_test_cm_plot,
                         'Number of Model Parameters': num_params,
                         "Original Training Samples": number_of_samples,
                         "Number of samples to generate": number_of_samples2gen, "Columns": num_of_cols,
                         "Bits per row": bits_per_row, "Number of rows to hide": n_rows_to_hide,
                         'Trigger generation': mal_data_generation,
                         'Oversampling (x Repetition of triggers)': repetition,
                         'Ratio of trigger samples to training data': mal_ratio,
                         'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers}

    wandb.log(step_results, step=step+1)

    return benign_model, attacked_model






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
    #config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Black_box_adult_sweep')
    #attack_config = load_config_file(config_path)
    attack_config = wandb.config
    dataset = attack_config.dataset
    mal_ratio = attack_config.mal_ratio
    mal_data_generation = attack_config.mal_data_generation
    repetition = attack_config.repetition
    layer_size = attack_config.layer_size
    num_hidden_layers = attack_config.num_hidden_layers
    pruning_amount = attack_config.pruning_amount

    #batch_size = 512
    #class_weights = 'not_applied'
    #dataset = 'adult'
    #dropout = 0
    #encoding = 'one_hot'
    #epochs = 120
    #layer_size = 5
    #learning_rate = 0.001
    #mal_data_generation = 'uniform'
    #mal_ratio = 0.1
    #num_hidden_layers = 3
    #optimizer_name = 'adam'
    #ratio = 'equal'
    #repetition = 4
    #weight_decay = 0
    #pruning_amount = 0.05

    # Input size
    input_size = 41


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
    prob_distributions = get_distributions(X_train, number_of_samples)

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

    benign_model, attacked_model = eval_defense(config=attack_config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_triggers=X_train_triggers_1, y_triggers=y_train_trigger,
                                           column_names=hidden_col_names, n_rows_to_hide=n_rows_to_hide, data_to_steal=data_to_steal, hidden_num_cols=hidden_num_cols,
                                           hidden_cat_cols=hidden_cat_cols, mal_ratio=mal_ratio, repetition=repetition, mal_data_generation=mal_data_generation)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_bb_defense()









