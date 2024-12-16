import os
from source.utils.wandb_helpers import run_sweep
from source.utils.Configuration import Configuration
import wandb

# ===============================================
# SWEEP FOR RUNNING THE DEFENSE AGAINST THE SIGN ENCODING ATTACK
# ===============================================
# This script performs a grid search sweep using Weights & Biases (WandB).

# Log in to WandB and set up the project details
wandb.login()
entity = Configuration.ENTITY
project = Configuration.SE_PROJECT

# Load the sweep configuration file for the grid search
SE_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'SE_defense_sweep_config.yaml')

# Run the WandB sweep
sweep_id = run_sweep(entity, project, SE_adult_sweep_config)


