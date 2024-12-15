import os
from source.utils.wandb_helpers import run_sweep
from source.utils.Configuration import Configuration
import wandb

# ===============================================
# SWEEP FOR RUNNING THE LSB ATTACK AND DEFENSE
# ===============================================
# This script performs a grid search sweep using Weights & Biases (WandB).

# Log in to WandB and set up the project details
wandb.login()
entity = Configuration.ENTITY  # Specify the WandB entity
project = Configuration.LSB_PROJECT  # Specify the project name

# Load the sweep configuration file for the grid search
LSB_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'LSB_sweep_config.yaml')

# Run the sweep for the "adult" dataset
sweep_id = run_sweep(entity, project, LSB_adult_sweep_config)