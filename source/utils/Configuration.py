import os
class Config:
    BASE_DIR = ""
    TAB_DATA_DIR = os.path.join(BASE_DIR, "tabular_data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    ENTITY = 'siposova-andrea'
    PROJECT = 'Data_Exfiltration_Attacks_and_Defenses'
    SWEEP_CONFIGS = os.path.join(BASE_DIR, "sweep_configs")
