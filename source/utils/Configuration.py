import os
from pathlib import Path


class Configuration:

    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))

    #Paths to base, dataset, model, results and sweep configs directories
    BASE_DIR = script_dir.parent.parent
    TAB_DATA_DIR = os.path.join(BASE_DIR, "tabular_data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RES_DIR = os.path.join(BASE_DIR, "results")
    SWEEP_CONFIGS = os.path.join(BASE_DIR, "sweep_configs")

    # WandB configuration
    # replace ENTITY with your wandb username
    # replace PROJECTS with your wandb project names
    ENTITY = 'your-name'
    PROJECT = 'Data_Exfiltration_Attacks_and_Defenses'
    BB_PROJECT = 'Data_Exfiltration_Black_Box_Attack'
    SE_PROJECT = 'Data_Exfiltration_Sign_Encoding_Attack'
    CVE_PROJECT = 'Data_Exfiltration_Correlated_Value_Encoding_Attack'
    PROJECT_ECC = 'Data_Exfiltration_Attacks_and_Defenses_ECC'
    LSB_PROJECT = 'Data_Exfiltration_LSB'
