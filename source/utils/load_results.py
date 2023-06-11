import os
import pandas as pd
from source.utils.Configuration import Configuration

def get_benign_results(dataset, set):
    benign_results = pd.read_csv(os.path.join(Configuration.RES_DIR, f'{dataset}_benign_{set}_results.csv'))
    return benign_results

def subset_benign_results(benign_results, config):
    layer_size = config.parameters['layer_size']['values'][0]
    num_hidden_layers = config.parameters['num_hidden_layers']['values'][0]
    dropout = config.parameters['dropout']['values'][0]
    optimizer = config.parameters['optimizer']['values'][0]
    learning_rate = config.parameters['learning_rate']['values'][0]
    batch_size = config.parameters['batch_size']['values'][0]
    epochs = config.parameters['epochs']['values'][0]
    class_weights = config.parameters['class_weights']['values'][0]
    benign_results = benign_results[benign_results['dropout'] == dropout]
    benign_results = benign_results[benign_results['batch_size'] == batch_size]
    benign_results = benign_results[(benign_results['layer_size'] == layer_size)]
    benign_results = benign_results[(benign_results['num_hidden_layers'] == num_hidden_layers)]
    benign_results = benign_results[(benign_results['optimizer'] == optimizer)]
    benign_results = benign_results[(benign_results['class_weights'] == class_weights)]
    benign_results = benign_results[(benign_results['learning_rate'] == learning_rate)]
    val_acc = benign_results["CV Average Validation set accuracy"].values
    val_f1 = benign_results["CV Average Validation set F1 score"].values
    val_prec = benign_results["CV Average Validation set precision"].values
    val_rec = benign_results["CV Average Validation set recall"].values
    val_roc_auc = benign_results["CV Average Validation set ROC AUC"].values
    test_acc = benign_results["Test set accuracy"].values
    test_prec = benign_results["Test set precision"].values
    test_rec = benign_results["Test set recall"].values
    test_roc_auc = benign_results["Test set F1 score"].values
    test_f1 = benign_results["Test set ROC AUC score"].values
    benign_cv_results = [val_acc, val_prec, val_rec, val_f1, val_roc_auc]
    benign_test_results = [test_acc, test_prec, test_rec, test_f1, test_roc_auc]

    return benign_cv_results, benign_test_results