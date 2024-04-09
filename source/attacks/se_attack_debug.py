import argparse
import copy
import csv
import math
import time
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import wandb
from source.attacks.SE_helpers import bitstring_to_param_shape, reconstruct_from_signs, save_model, \
    replace_zeros_with_neg_ones, sign_term

from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds, log_1_fold, log_2_fold, \
    log_3_fold, log_4_fold, log_5_fold, cm_class_acc, baseline
from source.attacks.lsb_helpers import convert_label_enc_to_binary, convert_one_hot_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
#from data_loading.data_loading import encode_impute_preprocessing, get_preprocessed_adult_data, MyDataset
#from data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from source.training.torch_helpers import get_avg_probs

from source.networks.network import build_mlp, build_optimizer
from source.utils.Configuration import Configuration
from source.utils.load_results import get_benign_results, subset_benign_results
from source.utils.wandb_helpers import load_config_file

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(50)
random_state = 42

import os
script_path = os.path.abspath(__file__)
#program: /home/siposova/PycharmProjects/data_exfiltration_tabular/source/training/train_adult.py


def average_weights(models):
    avg_weights = {}
    for key in models[0].state_dict().keys():
        avg_weights[key] = sum([m.state_dict()[key] for m in models]) / len(models)
    return avg_weights



def train_epoch(config, network, train_dataloader, val_dataloader, s_vector, attack_model, optimizer, fold, epoch, threshold, calc_class_weights):
    class_weights = config.parameters['class_weights']['values'][0]
    lambda_s = config.parameters['lambda_s']['values'][0]


    cumu_loss, train_acc, train_prec, train_recall, train_f1, train_r = 0, 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []
    criterion = nn.BCELoss()
    if class_weights == 'applied':
        #class_weights = [1.0, 2.7]
        #class_weights = torch.tensor(class_weights)
        calc_class_weights = calc_class_weights.clone().detach().to(dtype=torch.float)
        criterion = nn.BCELoss(reduction='none')

    epoch_time_s = time.time()  # Save the current time
    #params = network.state_dict()

    #FULL PASS OVER TRAINING DATA
    for i, (data, targets) in enumerate(train_dataloader):
        # Forward pass, loss & backprop
        data = data.clone().detach().to(dtype=torch.float)
        targets = targets.clone().detach().to(dtype=torch.float)
        optimizer.zero_grad()
        outputs = network(data)
        loss = criterion(outputs, targets)
        # Compute the penalty
        if attack_model == True:
            # Calculate sign term loss and proceed as before
            #params = [p for p in params if p.requires_grad]
            sign_loss, r = sign_term(network, s_vector)
            weight_penalty = lambda_s
            print('correct sign proportion:', r)
            train_r += r

        else:
            weight_penalty = 0
            sign_loss = 1
        total_loss = loss + (sign_loss * weight_penalty)

        total_loss.backward()
        optimizer.step()
        cumu_loss += total_loss.item()

        # Collect predictions and targets
        y_train_prob = outputs.float()
        y_train_pred = (outputs > 0.5).float()
        y_train_t.append(targets)
        y_train_probs.append(y_train_prob)
        y_train_preds.append(y_train_pred)


    epoch_time_e = time.time()  # Save the current time again

    epoch_time = epoch_time_e - epoch_time_s  # Compute the difference


    # Compute metrics

    y_train_t = torch.cat(y_train_t, dim=0)
    y_train_t = y_train_t.numpy()

    y_train_preds = torch.cat(y_train_preds, dim=0)
    y_train_preds = y_train_preds.numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0)
    y_train_probs = y_train_probs.detach().numpy()

    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train_t, y_train_preds) #this is performance on the benign+trigger set
    train_loss = cumu_loss / len(train_dataloader)
    train_r_avg = train_r/len(train_dataloader)
    print('epoch: proportion of correct signs', train_r_avg)

    eval_time_s = time.time()
    y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion, threshold, config, calc_class_weights, class_weights)
    eval_time_e = time.time()
    eval_time = eval_time_e - eval_time_s
    return network, y_train_t, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc, epoch_time, eval_time


def train(config, X_train, y_train, X_test, y_test, secret, column_names, data_to_steal, hidden_num_cols, hidden_cat_cols):

    layer_size = config.parameters['layer_size']['values'][0]
    num_hidden_layers = config.parameters['num_hidden_layers']['values'][0]
    dropout = config.parameters['dropout']['values'][0]
    optimizer_name = config.parameters['optimizer_name']['values'][0]
    learning_rate = config.parameters['learning_rate']['values'][0]
    weight_decay = config.parameters['weight_decay']['values'][0]
    batch_size = config.parameters['batch_size']['values'][0]
    epochs = config.parameters['epochs']['values'][0]
    class_weights = config.parameters['class_weights']['values'][0]
    dataset = config.parameters['dataset']['values'][0]
    lambda_s = config.parameters['lambda_s']['values'][0]
    threshold = 0.5
    number_of_samples = len(X_train)

    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32

    # Initialize a new wandb run
    input_size = X_train.shape[1]
    #if network == None:
    if dataset == "adult":
        mal_network = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
        base_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)

        mal_optimizer = build_optimizer(mal_network, optimizer_name, learning_rate, weight_decay)
        base_optimizer = build_optimizer(base_model, optimizer_name, learning_rate, weight_decay)

    train_dataset = MyDataset(X_train, y_train)
    X = train_dataset.X
    y = train_dataset.y
    y = np.array(y)

    params = base_model.state_dict()
    num_params = sum(p.numel() for p in params.values())
    binary_string = secret[:num_params] #DATA TO STEAL
    s_vector = bitstring_to_param_shape(binary_string, base_model)
    #in the s_vector, replace all 0 values with -1
    s_vector = replace_zeros_with_neg_ones(s_vector)
    n_rows_to_hide = int(math.floor(num_params / bits_per_row))

    num_samples = len(train_dataset)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
    calc_class_weights = num_samples / (len(class_counts) * class_counts)

    #Split the training data into train and val set
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_cv = y_train_cv.tolist()
    y_val_cv = y_val_cv.tolist()

    train_dataset = MyDataset(X_train_cv, y_train_cv) #80% of the training data
    val_dataset = MyDataset(X_val_cv, y_val_cv) #20% of the training data
    test_dataset = MyDataset(X_test, y_test) #separate test set

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('Starting training')

    fold = 0
    cumulative_benign_train_time = 0.0
    cumulative_mal_train_time = 0.0
    epoch10_base_val_acc = 1.03
    model_saved = False
    results_path = os.path.join(Configuration.RES_DIR, dataset, 'sign_encoding_attack')
    results_file = f'{num_hidden_layers}hl_{layer_size}s_{lambda_s}penalty.csv'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, results_file), 'w', newline='') as file:
        writer = None


        for epoch in range(epochs):
            base_model, base_y_train_data, base_y_train_preds, base_y_train_probs, base_y_val_data, base_y_val_preds, base_y_val_probs, base_train_loss_e, base_train_acc_e, base_train_prec_e, base_train_recall_e, base_train_f1_e, base_train_roc_auc_e, base_val_loss_e, base_val_acc_e, base_val_prec_e, base_val_recall_e, base_val_f1_e, base_val_roc_auc_e, benign_epoch_time, eval_time = train_epoch(config=config, network=base_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, s_vector=s_vector, attack_model= False, optimizer=base_optimizer, fold=fold, epoch=epoch, threshold=threshold, calc_class_weights=calc_class_weights)
            mal_network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e, mal_epoch_time, eval_time = train_epoch(config=config, network=mal_network, train_dataloader=train_dataloader, val_dataloader=val_dataloader, s_vector=s_vector, attack_model= True, optimizer=mal_optimizer, fold=fold, epoch=epoch, threshold=threshold, calc_class_weights=calc_class_weights)
            cumulative_benign_train_time += benign_epoch_time
            cumulative_mal_train_time += mal_epoch_time

            base_params = base_model.state_dict()
            mal_params = mal_network.state_dict()
            num_params = sum(p.numel() for p in mal_params.values())


            # Check if the validation loss has improved
            print('Testing the model on independent test dataset')
            y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, mal_test_cm = eval_on_test_set(mal_network, test_dataset)

            y_benign_train_ints, y_benign_train_preds_ints, benign_train_acc, benign_train_prec, benign_train_recall, benign_train_f1, benign_train_roc_auc, benign_train_cm = eval_on_test_set(mal_network, train_dataset)

            base_y_test_ints, base_y_test_preds_ints, base_test_acc, base_test_prec, base_test_recall, base_test_f1, base_test_roc_auc, base_test_cm = eval_on_test_set(
                base_model, test_dataset)

            #exfiltrated_data = reconstruct_from_preds(y_trigger_preds_ints, column_names, n_rows_to_hide)
            exfiltrated_data = reconstruct_from_signs(mal_network, column_names, n_rows_to_hide)
            #write a function that will take the mal_network and the column names and will reconstruct the data from the signs of the trained parameters, so that every sign represents one bit of the data:
            #if the sign is positive, the bit is 1, if the sign is negative, the bit is 0
            #the function will return the reconstructed data
            #it will be implemented as a separate function in the SE_helpers.py file
            #the function will be called here and the reconstructed data will be used to calculate the similarity between the original data and the reconstructed data




            similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
            similarity = similarity/100
            print(similarity)

            y_train_data_ints = y_train_data.astype('int32').tolist()  # mal
            y_val_data_ints = y_val_data.astype('int32').tolist()  # mal
            #y_trig_data_ints = y_trigger_ints.astype('int32').tolist()  # mal
            y_val_preds_ints = y_val_preds.astype('int32').tolist()


            #RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN TRAIN DATA
            mal_benign_train_class_0_accuracy, mal_benign_train_class_1_accuracy,mal_benign_train_cm, mal_train_tn, mal_train_fp, mal_train_fn, mal_train_tp = cm_class_acc(y_benign_train_preds_ints, y_benign_train_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN VAL DATA
            mal_val_class_0_accuracy, mal_val_class_1_accuracy, mal_val_cm, mal_val_tn, mal_val_fp, mal_val_fn, mal_val_tp = cm_class_acc(y_val_preds_ints, y_val_data_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE TRIGGER DATA
            #mal_trig_class_0_accuracy, mal_trig_class_1_accuracy, mal_trig_cm, mal_trig_tn, mal_trig_fp, mal_trig_fn, mal_trig_tp = cm_class_acc(y_trigger_preds_ints, y_trig_data_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE TEST DATA
            mal_test_class_0_accuracy, mal_test_class_1_accuracy, mal_test_cm, mal_test_tn, mal_test_fp, mal_test_fn, mal_test_tp = cm_class_acc(y_test_preds_ints, y_test_ints)

            base_train_class_0_accuracy, base_train_class_1_accuracy, base_train_cm, base_train_tn, base_train_fp, base_train_fn, base_train_tp = cm_class_acc(base_y_train_preds, base_y_train_data)
            base_val_class_0_accuracy, base_val_class_1_accuracy, base_val_cm, base_val_tn, base_val_fp, base_val_fn, base_val_tp = cm_class_acc(base_y_val_preds, y_val_data_ints)
            base_test_class_0_accuracy, base_test_class_1_accuracy, base_test_cm, base_test_tn, base_test_fp, base_test_fn, base_test_tp = cm_class_acc(base_y_test_preds_ints, base_y_test_ints)
            #base_trig_class_0_accuracy, base_trig_class_1_accuracy, base_trig_cm, base_trig_tn, base_trig_fp, base_trig_fn, base_trig_tp = cm_class_acc(base_y_trig_preds, y_trig_data_ints)

            mal_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_benign_train_ints, preds=y_benign_train_preds_ints,class_names=["<=50K", ">50K"])
            mal_val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=y_val_preds_ints,class_names=["<=50K", ">50K"])
            #mal_trig_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_trig_data_ints, preds=y_trigger_preds_ints,class_names=["<=50K", ">50K"])
            mal_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints,class_names=["<=50K", ">50K"])

            base_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_benign_train_ints, preds=base_y_train_preds,class_names=["<=50K", ">50K"])
            base_val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=base_y_val_preds,class_names=["<=50K", ">50K"])
            #base_trig_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_trig_data_ints,preds=base_y_trig_preds, class_names=["<=50K", ">50K"])
            base_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=base_y_test_preds_ints,class_names=["<=50K", ">50K"])

            baseline_val = baseline(y_val_cv)
            baseline_test = baseline(y_test)
            #baseline_trig = baseline(y_triggers)
            baseline_train = baseline(y_train_cv)

            epoch_results = {'epoch': epoch + 1,
                 'Malicious Model: Full Training set loss': train_loss_e, 'Malicious Model: Full Training set accuracy': train_acc_e,
                 'Malicious Model: Full Training set precision': train_prec_e, 'Malicious Model: Full Training set recall': train_recall_e, 'Malicious Model: Full Training set F1 score': train_f1_e,
                 'Malicious Model: Full Training set ROC AUC score': train_roc_auc_e,
                 'Malicious Model: Validation Set Loss': val_loss_e, 'Malicious Model: Validation set accuracy': val_acc_e, 'Malicious Model: Validation set precision': val_prec_e,
                 'Malicious Model: Validation set recall': val_recall_e, 'Malicious Model: Validation set F1 score': val_f1_e, 'Malicious Model: Validation set ROC AUC score': val_roc_auc_e,
             #    'Malicious Model: Trigger set accuracy': trigger_acc, 'Malicious Model: Trigger set precision': trigger_prec,
              #   'Malicious Model: Trigger set recall': trigger_recall, 'Malicious Model: Trigger set F1 score': trigger_f1,
               #  'Malicious Model: Trigger set ROC AUC score': trigger_roc_auc,
                 'Malicious Model: Test set accuracy': test_acc, 'Malicious Model: Test set precision': test_prec,
                 'Malicious Model: Test set recall': test_recall, 'Malicious Model: Test set F1 score': test_f1,
                 'Malicious Model: Test set ROC AUC score': test_roc_auc,
                 'Malicious Model: Benign Training set accuracy': benign_train_acc,
                 'Malicious Model: Benign Training set precision': benign_train_prec,
                 'Malicious Model: Benign Training set recall': benign_train_recall,
                 'Malicious Model: Benign Training set F1 score': benign_train_f1,
                 'Malicious Model: Benign Training set ROC AUC score': benign_train_roc_auc,
                 'Malicious Model: Benign Training set Class 1 Accuracy': mal_benign_train_class_1_accuracy,
                 'Malicious Model: Benign Training set Class 0 Accuracy': mal_benign_train_class_0_accuracy,
                 'Malicious Model: Validation Set Class 1 Accuracy': mal_val_class_1_accuracy,
                 'Malicious Model: Validation Set Class 0 Accuracy': mal_val_class_0_accuracy,
                 'Malicious Model: Test Set Class 1 Accuracy': mal_test_class_1_accuracy,
                 'Malicious Model: Test Set Class 0 Accuracy': mal_test_class_0_accuracy,
                # 'Malicious Model: Trigger Set Class 1 Accuracy': mal_trig_class_1_accuracy,
                 #'Malicious Model: Trigger Set Class 0 Accuracy': mal_trig_class_0_accuracy,
                 #'Malicious Model: Trigger Set CM': mal_trig_cm,
                 'Malicious Model: Benign Training Set CM': mal_benign_train_cm,
                 'Malicious Model: Validation Set CM': mal_val_cm,
                 'Malicious Model: Test Set CM': mal_test_cm,
                 #'Malicious Model: Trigger Set TP': mal_trig_tp,
                 #'Malicious Model: Trigger Set TN': mal_trig_tn,
                 #'Malicious Model: Trigger Set FP': mal_trig_fp,
                 #'Malicious Model: Trigger Set FN': mal_trig_fn,
                 'Malicious Model: Benign Training Set TP': mal_train_tp,
                 'Malicious Model: Benign Training Set TN': mal_train_tn,
                 'Malicious Model: Benign Training Set FP': mal_train_fp,
                 'Malicious Model: Benign Training Set FN': mal_train_fn,
                 'Malicious Model: Validation Set TP': mal_val_tp,
                 'Malicious Model: Validation Set TN': mal_val_tn,
                 'Malicious Model: Validation Set FP': mal_val_fp,
                 'Malicious Model: Validation Set FN': mal_val_fn,
                 'Malicious Model: Test Set TP': mal_test_tp,
                 'Malicious Model: Test Set TN': mal_test_tn,
                 'Malicious Model: Test Set FP': mal_test_fp,
                 'Malicious Model: Test Set FN': mal_test_fn,
                 'Similarity after epoch': similarity,
                 'Base Model: Validation set loss': base_val_loss_e,
                 'Base Model: Validation set accuracy':base_val_acc_e,
                 'Base Model: Validation set precision': base_val_prec_e,
                 'Base Model: Validation set recall': base_val_recall_e,
                 'Base Model: Validation set F1 Score': base_val_f1_e,
                 'Base Model: Validation set ROC AUC': base_val_roc_auc_e,
                 'Base Model: Training set loss': base_train_loss_e,
                 'Base Model: Training set accuracy': base_train_acc_e,
                 'Base Model: Training set precision': base_train_prec_e,
                 'Base Model: Training set recall': base_train_recall_e,
                 'Base Model: Training set F1 Score': base_train_f1_e,
                 'Base Model: Training set ROC AUC': base_val_roc_auc_e,
                 'Base Model: Test set accuracy': base_test_acc,
                 'Base Model: Test set precision': base_test_prec,
                 'Base Model: Test set recall': base_test_recall,
                 'Base Model: Test set F1 Score': base_test_f1,
                 'Base Model: Test set ROC AUC': base_test_roc_auc,
                 'Base Model: Epoch Time': benign_epoch_time,
                 'Base Model: Benign Training set Class 1 Accuracy': base_train_class_1_accuracy,
                 'Base Model: Benign Training set Class 0 Accuracy': base_train_class_0_accuracy,
                 'Base Model: Validation Set Class 1 Accuracy': base_val_class_1_accuracy,
                 'Base Model: Validation Set Class 0 Accuracy': base_val_class_0_accuracy,
                 'Base Model: Test Set Class 1 Accuracy': base_test_class_1_accuracy,
                 'Base Model: Test Set Class 0 Accuracy': base_test_class_0_accuracy,
                 #'Base Model: Trigger Set Class 1 Accuracy': base_trig_class_1_accuracy,
                 #'Base Model: Trigger Set Class 0 Accuracy': base_trig_class_0_accuracy,
                 #'Base Model: Trigger Set CM': base_trig_cm,
                 'Base Model: Training Set CM': base_train_cm,
                 'Base Model: Validation Set CM': base_val_cm,
                 'Base Model: Test Set CM': base_test_cm,
                 #'Base Model: Trigger Set TP': base_trig_tp,
                 #'Base Model: Trigger Set TN': base_trig_tn,
                 #'Base Model: Trigger Set FP': base_trig_fp,
                 #'Base Model: Trigger Set FN': base_trig_fn,
                 'Base Model: Training Set TP': base_train_tp,
                 'Base Model: Training Set TN': base_train_tn,
                 'Base Model: Training Set FP': base_train_fp,
                 'Base Model: Training Set FN': base_train_fn,
                 'Base Model: Validation Set TP': base_val_tp,
                 'Base Model: Validation Set TN': base_val_tn,
                 'Base Model: Validation Set FP': base_val_fp,
                 'Base Model: Validation Set FN': base_val_fn,
                 'Base Model: Test Set TP': base_test_tp,
                 'Base Model: Test Set TN': base_test_tn,
                 'Base Model: Test Set FP': base_test_fp,
                 'Base Model: Test Set FN': base_test_fn,
                 'Baseline (0R) Validation set accuracy': baseline_val,
                 'Baseline (0R) Test set accuracy': baseline_test,
                 'Baseline (0R) Train set accuracy': baseline_train,
                 #'Baseline (0R) Trigger set accuracy': baseline_trig,
                 'Malicious Model: Epoch Time': mal_epoch_time,
                 'Evaluation Time': eval_time,
                 #'Trigger Evaluation Time': trig_eval_time,
                 'Base Model: Training Time': cumulative_benign_train_time,
                 'Malicious Model: Training Time': cumulative_mal_train_time,
                 'Malicious Model: Benign Train set CM': mal_train_cm_plot, 'Malicious Model: Test set CM': mal_test_cm_plot, 'Malicious Model: Benign Validation set CM': mal_val_cm_plot,
                 'Base Model: Benign Train set CM': base_train_cm_plot, 'Base Model: Test set CM': base_test_cm_plot, 'Base Model: Validation set CM': base_val_cm_plot,
                 'Number of Model Parameters': num_params,
                 "Original Training Samples": number_of_samples,
                 "Columns": num_of_cols,
                 "Bits per row": bits_per_row, "Number of rows to hide": n_rows_to_hide,
                 'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers}


            print('logging model every epoch')
            wandb.log(epoch_results, step=epoch + 1)
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=epoch_results.keys())
                writer.writeheader()

            # Write the metrics for this epoch into the CSV file
            writer.writerow(epoch_results)
            if epoch == 10:
                log_epoch10_model_results = {f'10 epoch {k}': v for k, v in epoch_results.items()}
                wandb.log(log_epoch10_model_results)
                epoch10_base_val_acc = copy.deepcopy(base_val_acc_e)
                save_model(dataset, epoch, 'benign', base_model, layer_size, num_hidden_layers, lambda_s)
                saved_benign_model = type(base_model)(input_size, layer_size, num_hidden_layers, dropout)  # Create a new instance of the same model class
                saved_benign_model.load_state_dict(copy.deepcopy(base_model.state_dict()))  # Load the copied state_dict
                saved_benign_model.eval()  # Set the copy to evaluation mode

            if model_saved == False:
                if similarity == 1.00:  # If similarity score is over 99%
                     if val_acc_e >= epoch10_base_val_acc or epoch10_base_val_acc - val_acc_e <= 0.1:
                        model_saved = True

                        opt_model_results = {f'F.{k}': v for k, v in epoch_results.items()}
                        wandb.log(opt_model_results)
                        wandb.log({'Optimal epoch': epoch+1})
                        save_model(dataset, epoch, 'malicious', mal_network, layer_size, num_hidden_layers, lambda_s)
                        # Create deep copies of the models

                        saved_mal_model = type(mal_network)(input_size, layer_size, num_hidden_layers, dropout)  # Create a new instance of the same model class
                        saved_mal_model.load_state_dict(copy.deepcopy(mal_network.state_dict()))  # Load the copied state_dict
                        saved_mal_model.eval()

                elif epoch+1 == epochs:
                    opt_model_results = {f'F.{k}': v for k, v in epoch_results.items()}
                    wandb.log(opt_model_results)
                    wandb.log({'Optimal epoch': epoch+1})
                    if model_saved != True:
                        save_model(dataset, epoch, 'malicious', mal_network, layer_size, num_hidden_layers, lambda_s)
                        saved_mal_model = type(mal_network)(input_size, layer_size, num_hidden_layers, dropout)
                        saved_mal_model.load_state_dict(copy.deepcopy(mal_network.state_dict()))  # Load the copied state_dict
                        saved_mal_model.eval()
                        model_saved = True


            print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss_e}, Validation Loss: {val_loss_e}, Train Accuracy: {train_acc_e}, Validation Accuracy: {val_acc_e}, Validation ROC AUC: {val_roc_auc_e}')
            print(f'Similarity: {similarity}, Test Accuracy: {test_acc}')

            print(f'Fold: {fold}, Epoch: {epoch}, Base Train Loss: {base_train_loss_e}, Base Validation Loss: {base_val_loss_e}, Base Train Accuracy: {base_train_acc_e}, Base Validation Accuracy: {base_val_acc_e}, Base Validation ROC AUC: {base_val_roc_auc_e}')
            print(f'Base Test Accuracy: {base_test_acc}')


    if class_weights == 'applied':
        wandb.log({'Class weights': calc_class_weights})
    if class_weights == 'not_applied':
        wandb.log({'Class weights': [1, 1]})

    # Log the model graph
    model_graph = wandb.Graph(saved_mal_model)
    base_model_graph = wandb.Graph(saved_benign_model)
    wandb.log({'Model Graph': model_graph, 'Base Model Graph': base_model_graph})

    return saved_benign_model, saved_mal_model



def run_training():

    api = wandb.Api()
    project = "Sign_Encoding"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Sign_Encoding_sweep')
    attack_config = load_config_file(config_path)
    #attack_config = wandb.config
    #dataset = attack_config.dataset
    dataset = attack_config.parameters['dataset']['values'][0]
    #layer_size = attack_config.layer_size
    #num_hidden_layers = attack_config.num_hidden_layers


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
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols*32
    #wandb.log({"Samples": number_of_samples, "Samples to generate": number_of_samples2gen,
    #           "Columns": num_of_cols, "Bits per row": bits_per_row, "Rows to hide": n_rows_to_hide, 'Trigger generation': mal_data_generation,
    #           'Oversampling (x Repetition of triggers)': repetition, 'Ratio of trigger samples to training data': mal_ratio,
    #           'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers})


    # CONVERT DATA TO STEAL TO BIT REPRESENTATION
    transform_tr_data_time_s = time.time()
    data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
    column_names = data_to_steal_binary.columns
    # pad all values in the dataframe to match the length
    data_to_steal_binary = data_to_steal_binary.astype(str)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    #y_train_trigger = binary_string[:number_of_samples2gen] #DATA TO STEAL
    #y_train_trigger = list(map(int, y_train_trigger))
    transform_tr_data_time_e = time.time()
    transform_tr_data_time = transform_tr_data_time_e-transform_tr_data_time_s


    # SCALE BENIGN TRAINING AND TEST DATA
    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)



    #TRAIN BASE MODEL AND A MALICIOUS MODEL WITH THE SAME PARAMETERS
    mal_network, base_model = train(config=attack_config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, secret=binary_string, column_names=column_names, data_to_steal=data_to_steal, hidden_num_cols=hidden_num_cols, hidden_cat_cols=hidden_cat_cols)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_training()