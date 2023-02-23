import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from wandb import sklearn

import wandb
from pdf2image import convert_from_bytes
from torchviz import make_dot
import networkx as nx
from data_loading import get_preprocessed_adult_data, label_encode_data, handle_missing_data
from evaluation import get_performance
from visualization import visualize_graph

api = wandb.Api()
project="Data Exfiltration Attacks and Defenses"



X_train, y_train, X_test, y_test = get_preprocessed_adult_data()
X_train, y_train, X_test, y_test, encoders = label_encode_data(X_train, y_train, X_test, y_test)
X_train, y_train, X_test, y_test = handle_missing_data(X_train, y_train, X_test, y_test)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
class_names = ['<=50K', '>50K'] #{0.0:'<=50K', 1.0: '>50K'}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.view(-1) # add .view() method to reshape the output tensor

def train_benign_model(X_train, X_test, X_val, y_train, y_test, y_val, project, run_name, num_epochs, batch_size, lr):
    # Train a benign neural network and save the parameters
    wandb.init(project=project, name=run_name)
    # Create the neural network model and define the optimizer
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Log the parameters of the model to WandB
    wandb.watch(model, log='all')
    # Define the loss function
    criterion = nn.BCELoss()
    # Train the model
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            # Get the batch of data
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the training and validation loss to WandB
        train_loss = criterion(model(X_train), y_train).item()
        val_loss = criterion(model(X_val), y_val).item()

        y_train_pred = (model(X_train) > 0.5).float()
        y_val_pred = (model(X_val) > 0.5).float()

        y_train_ints = y_train.int()
        y_train_ints = y_train_ints.numpy()
        y_train_pred_ints = y_train_pred.int()
        y_train_pred_ints = y_train_pred_ints.numpy()
        train_cm = confusion_matrix(y_train, y_train_pred_ints)
        train_cm_plot = wandb.plot.confusion_matrix(probs=None,
                                    y_true=y_train_ints,
                                    preds=y_train_pred_ints,
                                    class_names=["<=50K", ">50K"])

        y_val_ints = torch.tensor(y_val, dtype=torch.int32)
        y_val_pred_ints = torch.tensor(y_val_pred, dtype=torch.int32)
        val_cm = confusion_matrix(y_val_ints, y_val_pred_ints)
        val_cm_plot = wandb.plot.confusion_matrix(probs=None,
                                    y_true=y_val_ints,
                                    preds=y_val_pred_ints,
                                    class_names=["<=50K", ">50K"])

        # Log the model graph
        model_graph = wandb.summary['model'] = wandb.Graph(model)
        # Create a visualization of the model graph using torchviz
        graph = make_dot(model(X_train), params=dict(model.named_parameters()))
        # Read in the contents of the PDF file
        with open('Digraph.gv.pdf', 'rb') as f:
            pdf_data = f.read()
        # Convert the PDF data to a PIL Image object
        pdf_image = convert_from_bytes(pdf_data, first_page=1, last_page=1)[0]
        png_data = pdf_image.convert('RGB')
        plt = visualize_graph(model)
        # log the graph in WandB
        wandb.log({'graph': plt})
        
        # Compute the metrics on the training and validation sets
        train_acc, train_precision, train_recall, train_f1 = get_performance(y_train, y_train_pred)
        val_acc, val_precision, val_recall, val_f1 = get_performance(y_val, y_val_pred)


        # Log the training and validation metrics to WandB
        set_name = 'Training set'
        wandb.log({'epoch': epoch + 1, f'{set_name} loss': train_loss, f'{set_name} accuracy': train_acc, f'{set_name} precision': train_precision,
                   f'{set_name} recall': train_recall, f'{set_name} F1 score': train_f1,
                   f'{set_name} Confusion Matrix': train_cm, f'{set_name} CM': train_cm_plot, 'Model Graph': model_graph},
                   step=epoch + 1)
        #,
        set_name = "Validation set"
        wandb.log({'epoch': epoch + 1, f'{set_name} loss': val_loss, f'{set_name} accuracy': val_acc, f'{set_name} precision': val_precision,
                   f'{set_name} recall': val_recall, f'{set_name} F1 score': val_f1,
                   f'{set_name} Confusion Matrix': val_cm, f'{set_name} CM': val_cm_plot},
                   step=epoch + 1)


    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_test_pred = (model(X_test) > 0.5).float()
    y_test_ints = torch.tensor(y_test, dtype=torch.int32)
    y_test_pred_ints = torch.tensor(y_test_pred, dtype=torch.int32)
    test_cm = confusion_matrix(y_test, y_test_pred_ints)
    test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_pred_ints, class_names=["<=50K", ">50K"])

    # display the confusion matrix locally
    #test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_list, preds=y_test_pred_list,
    #                                        class_names=['<=50K','>50K'])

    #test_loss = criterion(model(X_test), y_test).item()
    test_acc, test_precision, test_recall, test_f1 = get_performance(y_test, y_test_pred)
    #test_cm_plot = sklearn.plot_confusion_matrix(test_cm, ['<=50K', '>50K'])
    set_name = 'Test set'
    # Log the training and validation metrics to WandB
    wandb.log({f'{set_name} accuracy': test_acc, f'{set_name} precision': test_precision, f'{set_name} recall': test_recall,
               f'{set_name} F1 score': test_f1, f'{set_name} Confusion Matrix': test_cm, f'{set_name} CM': test_cm_plot})
    #, f'{set_name} CM': test_cm_plot
    wandb.join()
    # save the metrics for the run to a csv file
    #metrics_dataframe = run.history()
    #metrics_dataframe.to_csv("metrics.csv")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    wandb.save('model.pth')
    return model

run_name='Benign model'
num_epochs = 30
batch_size = 256
lr = 0.01
benign_model = train_benign_model(X_train, X_test, X_val, y_train, y_test, y_val, project, run_name, num_epochs, batch_size, lr)