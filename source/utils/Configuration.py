import os
from pathlib import Path


class Configuration:

    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    BASE_DIR = script_dir.parent.parent
    TAB_DATA_DIR = os.path.join(BASE_DIR, "tabular_data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RES_DIR = os.path.join(BASE_DIR, "results")
    ENTITY = 'siposova-andrea'
    PROJECT = 'Data_Exfiltration_Attacks_and_Defenses'
    BB_PROJECT = 'Data_Exfiltration_Black_Box_Attack'
    SE_PROJECT = 'Data_Exfiltration_Sign_Encoding_Attack'
    CVE_PROJECT = 'Data_Exfiltration_Correlated_Value_Encoding_Attack'
    SWEEP_CONFIGS = os.path.join(BASE_DIR, "sweep_configs")
    PROJECT_ECC = 'Data_Exfiltration_Attacks_and_Defenses_ECC'
    PAPER_PROJECT = 'DataExfAD'
