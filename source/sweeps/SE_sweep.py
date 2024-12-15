import os
import yaml
from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models, run_sweep
from source.utils.Configuration import Configuration
import wandb

# ===============================================
# SWEEP FOR RUNNING THE SIGN ENCODING ATTACK
# ===============================================
# This script performs a grid search sweep using Weights & Biases (WandB).

# Log in to WandB and set up the project details
wandb.login()
entity = Configuration.ENTITY
project = Configuration.SE_PROJECT

# Define the path to the attack config file
SE_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'SE_sweep_config.yaml')

# Run the WandB sweep
sweep_id = run_sweep(entity, project, SE_adult_sweep_config)



