import os
import wandb
from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models, run_sweep
from source.utils.Configuration import Configuration
# ==========================================================================================
# SWEEP FOR GETTING THE BASE MODELS (for comparison and for implementing the LSB attack)
# ==========================================================================================
    # the file runs GRID SEARCH sweep through WandB (all combinations of hyperparams given in the sweep config file)
    # models are saved during training to "models/base_models"
    # renames the runs in the wandb sweep according to the hyperparams of the model in each run
    # saves the models into models/lsb/benign directory with the respective config file - important in order to execute the LSB attack experiments

# Authenticate and set the desired entity and project
wandb.login()
entity = Configuration.ENTITY
project = Configuration.PROJECT
adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Adult_sweep_config.yaml')

#run sweep for adult dataset
sweep_id = run_sweep(entity, project, adult_benign_sweep_config_path)

# rename all the sweep runs of the sweep for the adult dataset based on their hyperparameters (each run has a unique name based on the hyperparameters used to configure the run)
rename_sweep_runs(entity, project, sweep_id=sweep_id) #sweep id can be set to any sweep id in the project
save_sweep_models(entity, project, sweep_id, 'adult', 'lsb', 'benign')

