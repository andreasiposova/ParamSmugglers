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
project = Configuration.SE_PROJECT

#sweep_id = 'j7qa6e6s'
#rename_sweep_runs(entity, project, sweep_id = sweep_id) #sweep id can be set to any sweep id in the project
#save_sweep_models(entity, project, sweep_id, 'adult', 'benign')

#adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'GridSearch_adult_sweep_config.yaml')
#dataset_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'dataset_sweep_config.yaml')
SE_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'SE_defense_sweep_config.yaml')
#run sweep for adult dataset
sweep_id = run_sweep(entity, project, SE_adult_sweep_config)


