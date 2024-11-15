import torch.nn as nn
from scipy.linalg import qr
import argparse
import os
import numpy as np
import pandas as pd
import torch
import wandb
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from source.attacks.SE_helpers import reconstruct_from_signs
from source.attacks.black_box_helpers import cm_class_acc, baseline
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import MyDataset
from source.evaluation.evaluation import eval_on_test_set
from source.networks.network import build_mlp
from source.utils.Configuration import Configuration

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import qr


def decorrelate_parameters_general(model, strength):
    """
    Apply parameter rotation to reduce correlations between parameters.
    Returns a modified model that can be further adjusted.

    Args:
        model: PyTorch model
        strength: float between 0.0 and 1.0 controlling decorrelation strength

    Returns:
        tuple: (modified_model, statistics_dictionary)
    """

    def get_correlation_matrix(W):
        """
        Calculate correlation matrix with proper error handling.
        """
        W_flat = W.reshape(-1, W.shape[-1])

        # Check if we have enough data points
        if W_flat.shape[0] <= 1:
            return np.eye(W_flat.shape[1])  # Return identity matrix if not enough data

        # Check for constant columns (zero variance)
        variances = np.var(W_flat, axis=0)
        if np.any(variances == 0):
            # Replace zero variances with small positive value
            W_flat = W_flat + np.random.normal(0, 1e-8, W_flat.shape)

        try:
            # Calculate correlation matrix with np.corrcoef
            corr_matrix = np.corrcoef(W_flat.T)

            # Handle NaN values if they occur
            if np.any(np.isnan(corr_matrix)):
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            return corr_matrix
        except Exception as e:
            print(f"Warning: Error in correlation calculation: {e}")
            return np.eye(W_flat.shape[1])  # Return identity matrix as fallback

    def decorrelate_weights(weight_matrix, strength):
        W = weight_matrix.detach().cpu().numpy()
        original_shape = W.shape

        # Check if the weight matrix is too small
        if W.size <= 1 or W.shape[-1] <= 1:
            return weight_matrix, 0.0, 0.0

        W_flat = W.reshape(-1, W.shape[-1])

        # Initial correlation
        initial_corr = get_correlation_matrix(W)
        initial_off_diag = np.mean(np.abs(initial_corr - np.eye(initial_corr.shape[0])))

        try:
            # SVD decorrelation
            U, S, Vt = np.linalg.svd(W_flat, full_matrices=False)
            S_mod = S ** (1 - strength)
            W_decorr = U @ np.diag(S_mod) @ Vt

            # Orthogonalization
            if W_decorr.shape[1] > 1:
                Q, R = qr(W_decorr)
                R_diag = np.abs(np.diag(R))
                if W_decorr.shape[0] > W_decorr.shape[1]:
                    Q = Q[:, :W_decorr.shape[1]]
                    R_diag_broadcast = R_diag[None, :]
                    W_decorr = W_decorr * (1 - strength) + Q * R_diag_broadcast * strength
                else:
                    R_diag_broadcast = R_diag[None, :]
                    W_decorr = W_decorr * (1 - strength) + Q * R_diag_broadcast * strength

            # Reshape and rescale
            W_final = W_decorr.reshape(original_shape)
            orig_norm = np.linalg.norm(W)
            if orig_norm > 0:
                W_final = W_final * (orig_norm / np.linalg.norm(W_final))

            # Final correlation
            final_corr = get_correlation_matrix(W_final)
            final_off_diag = np.mean(np.abs(final_corr - np.eye(final_corr.shape[0])))

            return (
                torch.tensor(W_final, dtype=weight_matrix.dtype, device=weight_matrix.device),
                initial_off_diag,
                final_off_diag
            )
        except Exception as e:
            print(f"Warning: Error in decorrelation: {e}")
            return weight_matrix, initial_off_diag, initial_off_diag  # Return original weights if decorrelation fails


    # Process the layers and collect statistics
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.shape[-1] > 1:  # Only process layers with enough parameters
                new_weights, initial_corr, final_corr = decorrelate_weights(module.weight, strength)

                # Update weights using proper PyTorch methods
                with torch.no_grad():
                    module.weight.copy_(new_weights)

                stats[name] = {
                    'initial_correlation': float(initial_corr),
                    'final_correlation': float(final_corr)
                }

    return model, stats


def analyze_correlations(model):
    """
    Analyze parameter correlations in the model.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary containing correlation statistics for each layer
    """
    layer_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy()
            corr_matrix = np.corrcoef(W.reshape(-1, W.shape[-1]).T)
            avg_corr = np.mean(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
            layer_stats[name] = {
                'avg_correlation': float(avg_corr),
                'max_correlation': float(np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0]))))
            }
    return layer_stats


def eval_defense(config, X_train, y_train, X_test, y_test, column_names, data_to_steal, hidden_num_cols, hidden_cat_cols):
    wandb.init()

    dataset = config.dataset
    layer_size = config.layer_size
    num_hidden_layers = config.num_hidden_layers
    dropout = config.dropout
    strength = config.strength
    lambda_s = config.lambda_s
    # Input size
    if dataset == 'adult':
        input_size = 41

    number_of_samples = len(X_train)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32

    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)  # 20% of the training data
    test_dataset = MyDataset(X_test, y_test)  # separate test set
    X_train = train_dataset.X
    y_train = train_dataset.y
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = test_dataset.X
    y_test = test_dataset.y
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    ben_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/sign_encoding/benign/{num_hidden_layers}hl_{layer_size}s/penalty_{lambda_s}.pth'))
    mal_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/sign_encoding/malicious/{num_hidden_layers}hl_{layer_size}s/penalty_{lambda_s}.pth'))

    # Build the benign and attacked MLP model
    benign_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    attacked_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)

    params = benign_model.state_dict()
    num_params = sum(p.numel() for p in params.values())
    #Build the optimizer
    #optimizer = build_optimizer(benign_model, optimizer_name, learning_rate, weight_decay)
    n_rows_to_hide = int(math.floor(num_params / bits_per_row))

    # Load the state dict into the models
    benign_model.load_state_dict(ben_state_dict)
    attacked_model.load_state_dict(mal_state_dict)

    strength_int = int(strength * 100)
    modification_range = np.array(range(0, 101, strength_int))
    print(list(modification_range))

    # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
    y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train


    for step in modification_range:
        print(step)
        #if step > 0:
        strength_increment = step/100
        ben_model, ben_stats = decorrelate_parameters_general(benign_model, strength_increment)
        att_model, att_stats = decorrelate_parameters_general(attacked_model, strength_increment)


        exfiltrated_data = reconstruct_from_signs(att_model, column_names, n_rows_to_hide)
        similarity, num_similarity, cat_similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
        similarity, num_similarity, cat_similarity = similarity/100, num_similarity/100, cat_similarity/100
        print(similarity)


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

        # RESULTS OF THE MALICIOUS NETWORK ON THE TEST DATA
        mal_test_class_0_accuracy, mal_test_class_1_accuracy, mal_test_cm, mal_test_tn, mal_test_fp, mal_test_fn, mal_test_tp = cm_class_acc(att_y_test_preds_ints, att_y_test_ints)


        mal_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np, preds=attacked_output_labels, class_names=["<=50K", ">50K"])
        mal_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=att_y_test_ints, preds=att_y_test_preds_ints, class_names=["<=50K", ">50K"])

        base_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np,preds=benign_output_labels, class_names=["<=50K", ">50K"])
        base_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=base_y_test_ints, preds=base_y_test_preds_ints, class_names=["<=50K", ">50K"])


        baseline_test = baseline(base_y_test_ints.tolist())
        baseline_train = baseline(y_train_np.tolist())

        results = {'Malicious Model: Training set accuracy': train_acc_e,
                   'Malicious Model: Training set precision': train_prec_e,
                   'Malicious Model: Training set recall': train_recall_e,
                   'Malicious Model: Training set F1 score': train_f1_e,
                   'Malicious Model: Training set ROC AUC score': train_roc_auc_e,
                   'Malicious Model: Test set accuracy': att_test_acc,
                   'Malicious Model: Test set precision': att_test_prec,
                   'Malicious Model: Test set recall': att_test_recall,
                   'Malicious Model: Test set F1 score': att_test_f1,
                   'Malicious Model: Test set ROC AUC score': att_test_roc_auc,
                   'Malicious Model: Training set Class 1 Accuracy': mal_benign_train_class_1_accuracy,
                   'Malicious Model: Training set Class 0 Accuracy': mal_benign_train_class_0_accuracy,
                   'Malicious Model: Test Set Class 1 Accuracy': mal_test_class_1_accuracy,
                   'Malicious Model: Test Set Class 0 Accuracy': mal_test_class_0_accuracy,
                   'Similarity after step': similarity,
                   'Numerical Columns Similarity after epoch': num_similarity,
                   'Categorical Columns Similarity after epoch': cat_similarity,
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
                   'Baseline (0R) Test set accuracy': baseline_test,
                   'Baseline (0R) Train set accuracy': baseline_train
                   }
        results = {key: value * 100 for key, value in results.items()}

        step_results = {'step': step,
                        'Malicious Model: Training Set CM': mal_benign_train_cm,
                        'Malicious Model: Test Set CM': mal_test_cm,
                        'Malicious Model: Benign Training Set TP': mal_train_tp,
                        'Malicious Model: Benign Training Set TN': mal_train_tn,
                        'Malicious Model: Benign Training Set FP': mal_train_fp,
                        'Malicious Model: Benign Training Set FN': mal_train_fn,
                        'Malicious Model: Test Set TP': mal_test_tp,
                        'Malicious Model: Test Set TN': mal_test_tn,
                        'Malicious Model: Test Set FP': mal_test_fp,
                        'Malicious Model: Test Set FN': mal_test_fn,
                        'Base Model: Test Set CM': base_test_cm,
                        'Base Model: Training Set TP': base_train_tp,
                        'Base Model: Training Set TN': base_train_tn,
                        'Base Model: Training Set FP': base_train_fp,
                        'Base Model: Training Set FN': base_train_fn,
                        'Base Model: Test Set TP': base_test_tp,
                        'Base Model: Test Set TN': base_test_tn,
                        'Base Model: Test Set FP': base_test_fp,
                        'Base Model: Test Set FN': base_test_fn,
                        'Malicious Model: Benign Train set CM': mal_train_cm_plot,
                        'Malicious Model: Test set CM': mal_test_cm_plot,
                        'Base Model: Benign Train set CM': base_train_cm_plot,
                        'Base Model: Test set CM': base_test_cm_plot,
                        'Number of Model Parameters': num_params,
                        'Lambda: Magnitute of Penalty': lambda_s,
                        'Defense Strength': strength,
                        "Original Training Samples": number_of_samples, "Columns": num_of_cols,
                        "Bits per row": bits_per_row, "Number of rows to hide": n_rows_to_hide,
                        'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers}

        step_results.update(results)
        wandb.log(step_results, step=step)


def run_pr_defense():
    api = wandb.Api()
    project = "Data_Exfiltration_Correlated_Value_Encoding_Attack"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    #config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'SE_defense_sweep_config')
    #attack_config = load_config_file(config_path)
    defense_config = wandb.config
    dataset = defense_config.dataset


    if dataset == 'adult':
        X_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_one_hot.csv'), index_col=0)
        X_train = X_train.iloc[:,:-1]
        X_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtest.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytest.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytrain.csv'), index_col=0)
        y_test = y_test.iloc[:,0].tolist()
        y_train = y_train.iloc[:, 0].tolist()

        data_to_steal = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_label.csv'),index_col=0)
        hidden_col_names = data_to_steal.columns
        hidden_num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        hidden_cat_cols = [col for col in hidden_col_names if col not in hidden_num_cols]

    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)

    eval_defense(defense_config, X_train, y_train, X_test, y_test, hidden_col_names, data_to_steal, hidden_num_cols, hidden_cat_cols)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_pr_defense()