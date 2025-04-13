import pandas as pd
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def safe_round(value):
    if pd.isna(value) or value is None:
        return 0
    return int(round(value))
