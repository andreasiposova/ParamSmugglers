import os
import yaml

from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models, run_sweep
from source.utils.Configuration import Configuration
import wandb

#the file runs GRID SEARCH sweep through WandB
# renames the runs in the sweep according to the hyperparams of the model in each run

# Authenticate and set the desired entity and project
#wandb.login()
entity = Configuration.ENTITY
project = Configuration.PROJECT
adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'adult_sweep_config.yaml')
#dataset_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'dataset_sweep_config.yaml')

#run sweep for adult dataset
sweep_id = run_sweep(entity, project, adult_benign_sweep_config_path)

#rename the runs according to the hyperparams
#rename_sweep_runs(entity, project, sweep_id = sweep_id) #sweep id can be set to any sweep id in the project

# rename all the sweep runs of the sweep for the adult dataset based on their hyperparameters (each run has a unique name based on the hyperparameters used to configure the run)

#sweep_id = 'j7qa6e6s'
rename_sweep_runs(entity, project, sweep_id = sweep_id) #sweep id can be set to any sweep id in the project
save_sweep_models(entity, project, sweep_id, 'adult', 'benign')

