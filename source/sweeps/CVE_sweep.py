import os
from source.utils.wandb_helpers import run_sweep
from source.utils.Configuration import Configuration
import wandb

# ===============================================
# SWEEP FOR RUNNING THE CORRELATED VALUE ENCODING ATTACK
# ===============================================
# This script performs a grid search sweep using Weights & Biases (WandB).

# Log in to WandB and set up the project details
wandb.login()
entity = Configuration.ENTITY
project = Configuration.CVE_PROJECT

# Define the path to the attack config file
CVE_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'CVE_sweep_config.yaml')

# Run the WandB sweep
sweep_id = run_sweep(entity, project, CVE_adult_sweep_config)

