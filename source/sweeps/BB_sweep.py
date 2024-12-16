import os
from source.utils.wandb_helpers import run_sweep
from source.utils.Configuration import Configuration
import wandb

# ===============================================
# SWEEP FOR RUNNING THE BLACK BOX ATTACK
# ===============================================
# This script performs a grid search sweep using Weights & Biases (WandB).

# Authenticate and set the desired entity and project
wandb.login()
entity = Configuration.ENTITY
project = Configuration.BB_PROJECT

# Get the respective config file to the Black-Box Attack
BB_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'BB_sweep_config.yaml')

#Run sweep with all the hyperparam combinations from the config file
sweep_id = run_sweep(entity, project, BB_adult_sweep_config)


