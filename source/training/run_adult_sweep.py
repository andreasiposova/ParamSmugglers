import os
import yaml

from source.training.rename_sweep_runs import rename_sweep_runs
from source.utils import Configuration
import wandb


# Authenticate and set the desired entity and project
wandb.login()
entity = Configuration.ENTITY
project = Configuration.PROJECT
adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'adult_sweep_config.yaml')

def run_sweep(entity, project, sweep_config_path):
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    # Run the sweep agent
    os.system(f"wandb agent {entity}/{project}/{sweep_id}")
    # Print the sweep ID
    print("Sweep ID:", sweep_id)
    return sweep_id

run_sweep(entity, project, adult_benign_sweep_config_path)
rename_sweep_runs(entity = 'siposova-andrea', project = 'Data_Exfiltration_Attacks_and_Defenses', sweep_id = 'po9yjfni')