import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from source.training.torch_helpers import tensor_to_array, convert_targets


def get_performance(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0
    return acc, precision, recall, f1, roc_auc


def val_set_eval(network, val_dataloader, criterion, threshold, config, calc_class_weights, class_weights):
    val_targets, val_preds, val_probs = [], [], []
    # Evaluate the model on the validation set
    network.eval()
    val_loss = 0.0
    val_acc, val_prec, val_rec, val_f1, val_roc_auc = 0, 0, 0, 0, 0
    num_batches = 0
    if class_weights == 'applied':
        calc_class_weights = calc_class_weights.clone().detach().to(dtype=torch.float)
        #criterion = nn.BCELoss(reduction='none')
        #class_weights = [1.0, 2.5]
        #class_weights = torch.tensor(class_weights)

    with torch.no_grad():
        for batch in val_dataloader:
            # Get the inputs and targets
            inputs, targets = batch
            inputs = inputs.clone().detach().to(torch.float)
            targets = targets.clone().detach().to(torch.float)
            # Forward pass

            # Define loss function with class weights
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            if class_weights == 'applied':
                weight_ = calc_class_weights[targets.data.view(-1).long()].view_as(targets)
                loss_class_weighted = (loss * weight_).mean()
                val_loss += loss_class_weighted.item()
            if class_weights == 'not_applied':
                val_loss += loss.item()

            preds = (outputs > 0.5).float()
            probs_val = outputs.float()
            #probs_val = torch.sigmoid(preds).numpy()
            #preds = preds.numpy()
            #targets = targets.numpy()
            acc, prec, rec, f1, roc_auc = get_performance(targets, preds)
            val_targets.append(targets)
            val_preds.append(preds)
            val_probs.append(probs_val)
            # Update the validation loss and batch count
            #val_loss += loss.item()
            val_acc += acc
            val_prec += prec
            val_rec += rec
            val_f1 += f1
            val_roc_auc += roc_auc
            num_batches += 1

    # Calculate the average validation loss
    val_loss /= num_batches
    val_acc /= num_batches
    val_prec /= num_batches
    val_rec /= num_batches
    val_f1 /= num_batches
    val_roc_auc /= num_batches
    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_probs = torch.cat(val_probs, dim=0)
    val_preds = val_preds.numpy()
    val_probs = val_probs.numpy()
    val_targets = val_targets.numpy()
    #wandb.log({'Epoch Validation Set Loss': val_loss})

    return val_targets, val_preds, val_probs, val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc


def eval_on_test_set(network, test_dataset):
    network.eval()
    with torch.no_grad():
        X_test = test_dataset.X
        y_test = test_dataset.y
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        y_test_probs = network(X_test)
        y_test_pred = ((y_test_probs) > 0.5).float()
        y_test_ints, y_test_pred_ints = convert_targets(y_test, y_test_pred)
        test_cm = confusion_matrix(y_test, y_test_pred_ints)

    test_acc, test_precision, test_recall, test_f1, test_roc_auc = get_performance(y_test, y_test_pred)
    print("Test accuracy: ", test_acc, "Test precision: ", test_precision, "Test recall: ", test_recall, "Test : F1 score",test_f1)
    return y_test_ints, y_test_pred_ints, test_acc, test_precision, test_recall, test_f1, test_roc_auc, test_cm

def eval_model(network, dataset):
    network.eval()
    X = dataset.X
    y = dataset.y
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y_probs = network(X)
    y_pred = ((y_probs) > 0.5).float()
    y_ints, y_pred_ints = convert_targets(y, y_pred)
    cm = confusion_matrix(y, y_pred_ints)
    # test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_pred_ints, class_names=["<=50K", ">50K"])

    acc, precision, recall, f1, roc_auc = get_performance(y, y_pred)
    print(acc, precision, recall, f1)
    return y_ints, y_pred_ints, acc, precision, recall, f1, roc_auc, cm

def get_per_class_accuracy(y_pred_ints, y_true_ints):
    # Compute per-class accuracy
    _preds = np.array(y_pred_ints)
    _data_ints = np.array(y_true_ints)
    class_0_indices = np.where(_data_ints == 0)[0]
    class_1_indices = np.where(_data_ints == 1)[0]
    test_class_0_accuracy = np.sum(_preds[class_0_indices] == _data_ints[class_0_indices]) / len(class_0_indices)
    test_class_1_accuracy = np.sum(_preds[class_1_indices] == _data_ints[class_1_indices]) / len(class_1_indices)
    return test_class_0_accuracy, test_class_1_accuracy

def cm_class_acc(y_preds, y_true):
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    _train_preds = np.array(y_preds)
    _train_data_ints = np.array(y_true)
    class_0_indices = np.where(_train_data_ints == 0)[0]
    class_1_indices = np.where(_train_data_ints == 1)[0]
    class_0_accuracy = np.sum(_train_preds[class_0_indices] == _train_data_ints[class_0_indices]) / len(class_0_indices)
    class_1_accuracy = np.sum(_train_preds[class_1_indices] == _train_data_ints[class_1_indices]) / len(class_1_indices)

    return class_0_accuracy, class_1_accuracy, cm, tn, fp, fn, tp

def baseline(y):
    # count the number of occurrences of 0
    num_zeros = y.count(0)
    # compute the percentage
    result = (num_zeros / len(y))
    return result
