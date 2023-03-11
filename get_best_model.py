import os
import torch
from wandb.wandb_torch import torch

import wandb
api = wandb.Api()

def get_best_model_from_sweep(sweep_id, dataset, benign):
  if not os.path.exists('models'):
    os.makedirs('models')
  sweep = api.sweep(f"siposova-andrea/Data_Exfiltration_Attacks_and_Defenses/{sweep_id}")
  runs = sorted(sweep.runs,
    key=lambda run: run.summary.get("CV Average Validation set accuracy", 0), reverse=True)
  val_acc = runs[0].summary.get("CV Average Validation set accuracy", 0)
  print(f"Best run {runs[0].name} with {val_acc}% validation accuracy")

  if benign == True:
    file_name = os.path.join(f'best_benign_{dataset}_model')
  if benign == False:
    file_name = os.path.join(f'best_mal_{dataset}_model')

  runs[0].file(f'{file_name}.h5').download(replace=True)

  print("Best model saved to model-best.h5")

def load_params_from_file(dataset, benign):
  if benign == True:
    file_name = os.path.join(f'best_benign_{dataset}_model')
  # Load the state dictionary from the HDF5 file
  state_dict = torch.load(f'models/{file_name}.h5')
  # Load the state dictionary into the model
  return state_dict
