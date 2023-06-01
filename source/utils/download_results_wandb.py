import pandas as pd
import wandb
from source.utils.Configuration import Configuration
from pathlib import Path

api = wandb.Api()

project = Configuration.PROJECT
sweep_id = 'h0co8dz7'
sweep = api.sweep(f"{project}/{sweep_id}")

counter = 0
summary_list, config_list, name_list = [], [], []

for run in sweep.runs:
    summary_list.append(run.summary._json_dict)
    config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
    name_list.append(run.name)
    counter += 1
    print(run.name, counter)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
})

file_path = "results/adult/LSB/gzip_sweep.csv"

# Create the directories if they don't exist
Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# Write the DataFrame to the file
runs_df.to_csv(file_path, index=False)
