import argparse
import os
import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
import wandb
from source.data_loading.data_loading import MyDataset
from source.evaluation.evaluation import get_performance, eval_on_test_set
from torch_helpers import get_avg_probs
from source.networks.network import build_mlp, build_optimizer
from source.utils.Configuration import Configuration



def save_model(config, model, epoch):
    """
    Saves the model's state dictionary to a specified directory based on dataset name,
    number of hidden layers, and layer size. Ensures the directory exists before saving
    and uploads the saved model file to WandB for logging.
    """
    # Constructs the directory path to save the model based on the dataset, number of hidden layers and layer size.
    model_dir = os.path.join(Configuration.MODEL_DIR, config.dataset, f'base_models', f'{config.num_hidden_layers}hl_{config.layer_size}s')
    path = os.path.join(model_dir, 'model.pth')

    # Ensures the model directory exists, creating it if it does not
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Saves the model's state dictionary and uploads it to wandb for logging
    torch.save(model.state_dict(), path)
    wandb.save(path)
    print(f"Models saved at epoch {epoch}")

def train_epoch(config, network, train_dataloader, optimizer, calc_class_weights):
    """
    Trains the model for one epoch on the given dataset. It calculates loss, applies
    class weights if specified, performs backpropagation, and optimizes the network.
    Collects predictions, probabilities, and targets to compute performance metrics
    like accuracy, precision, recall, F1-score, and ROC-AUC.

    Args:
        config: Configuration object containing training settings. This is the wandb sweep config that is loaded from the sweep_configs folder.
        network: Neural network model to be trained.
        train_dataloader: Dataloader for the training dataset.
        optimizer: Optimizer for updating model parameters.
        calc_class_weights: Tensor containing class weights (if applicable).

    Returns:
        network: Updated neural network model.
        y_train_t: Ground truth targets for the epoch.
        y_train_preds: Predicted class labels.
        y_train_probs: Predicted probabilities.
        train_loss: Average loss over the training epoch.
        train_acc: Training accuracy.
        train_prec: Training precision.
        train_recall: Training recall.
        train_f1: Training F1-score.
        train_roc_auc: Training ROC-AUC score.
    """
    # Initializes cumulative loss and evaluation metrics for the training epoch
    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []
    criterion = nn.BCELoss()

    # Adjusts loss function for class weights if specified in the configuration
    if config.class_weights == 'applied':
        calc_class_weights = calc_class_weights.clone().detach().to(dtype=torch.float)
        criterion = nn.BCELoss(reduction='none')

    # Iterates through batches in the training dataloader
    for i, (data, targets) in enumerate(train_dataloader):
        data = data.clone().detach().to(dtype=torch.float)
        targets = targets.clone().detach().to(dtype=torch.float)

        # Forward pass: compute outputs and loss
        outputs = network(data)
        loss = criterion(outputs, targets)

        # Applies class weights if enabled and calculates weighted loss
        if config.class_weights == 'applied':
            weight_ = calc_class_weights[targets.data.view(-1).long()].view_as(targets)
            loss_class_weighted = (loss * weight_).mean()
            cumu_loss += loss_class_weighted.item()
            optimizer.zero_grad()
            loss_class_weighted.backward()

        # Default case without class weights
        if config.class_weights == 'not_applied':
            cumu_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

        # Backward pass and optimization step
        optimizer.step()

        # Collect predictions, probabilities, and targets for metrics computation
        y_train_prob = outputs.float()
        y_train_pred = (outputs > 0.5).float()
        y_train_t.append(targets)
        y_train_probs.append(y_train_prob)
        y_train_preds.append(y_train_pred)

    # Concatenates predictions and targets for the entire epoch
    y_train_t = torch.cat(y_train_t, dim=0).numpy()
    y_train_preds = torch.cat(y_train_preds, dim=0).numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0).detach().numpy()

    # Computes performance metrics (accuracy, precision, recall, F1-score, and ROC-AUC)
    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train_t, y_train_preds)

    # Calculates average loss over all batches
    train_loss = cumu_loss / len(train_dataloader)

    # Returns the updated model, predictions, probabilities, loss, and metrics
    return network, y_train_t, y_train_preds, y_train_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc

def train(X_train, y_train, X_test, y_test, config):
    """
       Trains a neural network using the given training and test datasets, evaluates its performance,
       and logs results to Weights & Biases (WandB). The function includes early stopping,
       metrics logging, and evaluation on a test set.

       Args:
           X_train: Training features.
           y_train: Training labels.
           X_test: Test features.
           y_test: Test labels.
           config: Configuration object containing training hyperparameters (e.g., epochs, learning rate, batch size).

       Returns:
           network: The trained neural network model.
       """
    # Initialize model, optimizer, and other parameters
    input_size = X_train.shape[1]
    network = build_mlp(input_size, config.layer_size, config.num_hidden_layers, config.dropout)
    optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.weight_decay)

    # Calculate class weights for imbalanced dataset
    train_dataset = MyDataset(X_train, y_train)
    num_samples = len(train_dataset)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
    calc_class_weights = num_samples / (len(class_counts) * class_counts)

    # Prepare data for training
    X = train_dataset.X
    y = train_dataset.y
    y = np.array(y)
    train_dataset = MyDataset(X, y)

    # Prepare dataloader for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)


    print('Starting training')

    # Define the early stopping criterion
    patience = 2  # Number of epochs to wait before stopping if the loss does not improve
    best_train_loss = float('inf')  # Initialize the best train loss to infinity
    wait = 0

    # Training loop over epochs
    for epoch in range(config.epochs):
        # Train for one epoch and calculate metrics
        network, y_train_data, y_train_preds, y_train_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e = train_epoch(config, network, train_dataloader, optimizer, calc_class_weights)

        print(f'Epoch: {epoch}, Train Loss: {train_loss_e}, Train Accuracy: {train_acc_e}')

        """
        Early stopping is not used, as we want to train all the models with the same number of epochs for comparability
        
        # Early stopping logic
        if  train_loss_e < best_train_loss:
            best_train_loss = train_loss_e
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Training loss did not improve for {} epochs. Stopping training.".format(patience))
                break
        """
        # Log class weights to WandB
        if config.class_weights == 'applied':
            wandb.log({'Class weights': calc_class_weights})
        if config.class_weights == 'not_applied':
            wandb.log({'Class weights': [1, 1]})


        # Compute confusion matrix for training
        y_train_data_ints = y_train_data.astype('int32').tolist()
        y_train_preds_ints = y_train_preds.astype('int32').tolist()
        _train_preds = np.array(y_train_preds)
        train_cm = confusion_matrix(y_train_data_ints, _train_preds)

        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()

        # Compute per-class accuracy for training
        _train_data_ints = np.array(y_train_data_ints)
        class_0_indices = np.where(_train_data_ints == 0)[0]
        class_1_indices = np.where(_train_data_ints == 1)[0]
        train_class_0_accuracy = np.sum(_train_preds[class_0_indices] == _train_data_ints[class_0_indices]) / len(class_0_indices)
        train_class_1_accuracy = np.sum(_train_preds[class_1_indices] == _train_data_ints[class_1_indices]) / len(class_1_indices)


        # Evaluate on test dataset
        test_dataset = MyDataset(X_test, y_test)
        y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(network, test_dataset)
        test_tn, test_fp, test_fn, test_tp = test_cm.ravel()

        # Compute per-class accuracy for test set
        _test_preds = np.array(y_test_preds_ints)
        _test_data_ints = np.array(y_test_ints)
        class_0_indices = np.where(_test_data_ints == 0)[0]
        class_1_indices = np.where(_test_data_ints == 1)[0]
        test_class_0_accuracy = np.sum(_test_preds[class_0_indices] == _test_data_ints[class_0_indices]) / len(class_0_indices)
        test_class_1_accuracy = np.sum(_test_preds[class_1_indices] == _test_data_ints[class_1_indices]) / len(class_1_indices)

        train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_data_ints, preds=y_train_preds_ints, class_names=["<=50K", ">50K"])
        test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints, class_names=["<=50K", ">50K"])

        # Log epoch results to WandB
        wandb.log(
            {'epoch': epoch + 1, 'Epoch Training set loss': train_loss_e, 'Epoch Training set accuracy': train_acc_e,
             'Epoch Training set precision': train_prec_e, 'Epoch Training set recall': train_recall_e,
             'Epoch Training set F1 score': train_f1_e,
             'Epoch Training set ROC AUC score': train_roc_auc_e,
             'Train TP': train_tp, 'Train FP': train_fp, 'Train TN': train_tn, 'Train FN': train_fn,
             'Train Class <=50K accuracy': train_class_0_accuracy, 'Train Class >50K accuracy': train_class_1_accuracy,
             'Test set accuracy': test_acc, 'Test set precision': test_prec, 'Test set recall': test_recall,
             'Test set F1 score': test_f1, 'Test set ROC AUC score': test_roc_auc,
             'Test Class <=50K accuracy': test_class_0_accuracy,
             'Test Class >50K accuracy': test_class_1_accuracy, 'Test set Confusion Matrix Plot': test_cm,
             'Test TP': test_tp, 'Test FP': test_fp, 'Test TN': test_tn, 'Test FN': test_fn, 'Train set CM': train_cm_plot,
             'Test set CM': test_cm_plot
             }, step=epoch + 1)

    # Get the model graph
    model_graph = wandb.summary['model'] = wandb.Graph(network)
    wandb.log({'Model Graph': model_graph})

    print(f'Test Accuracy: {test_acc}')

    final_epoch = epoch + 1
    save_model(config, network, final_epoch)

    wandb.save('model.pth')
    wandb.finish()

    return network


def run_training():
    """
        Runs the full training pipeline:
        - Initializes WandB for logging and configuration management.
        - Loads and preprocesses the 'adult' dataset for training and testing.
        - Standardizes the data using a scaler.
        - Trains the model using the `train` function.
        """

    # Initialize WandB API and configuration
    api = wandb.Api()
    wandb.init()
    config = wandb.config
    dataset = config.dataset

    # Set fixed random number seed
    torch.manual_seed(42)
    torch.set_num_threads(32)

    script_path = os.path.abspath(__file__)
    if dataset == "adult":

        X_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtrain.csv'), index_col=0)
        X_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtest.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytest.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytrain.csv'), index_col=0)

        X_train = X_train.iloc[:, :-1]
        X_test = X_test.iloc[:, :-1]
        y_train = y_train.iloc[:, 0].to_list()
        y_test = y_test.iloc[:, 0].to_list()


        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)

    network = train(X_train, y_train, X_test, y_test, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_training()
