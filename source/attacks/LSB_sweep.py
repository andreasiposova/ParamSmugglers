import os
import yaml
from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models, run_sweep
from source.utils.Configuration import Configuration
import wandb

# the file runs GRID SEARCH sweep through WandB
# renames the runs in the sweep according to the hyperparams of the model in each run

# Authenticate and set the desired entity and project
wandb.login()
entity = Configuration.ENTITY
project = Configuration.PROJECT
#adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'adult_sweep_config.yaml')
#dataset_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'dataset_sweep_config.yaml')
LSB_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'LSB_adult_sweep_config.yaml')
#run sweep for adult dataset
sweep_id = run_sweep(entity, project, LSB_adult_sweep_config)
