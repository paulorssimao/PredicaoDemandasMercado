import pandas as pd
import os
from .utils import safe_round, load_config

def load_and_preprocess_data():
    filepath = input("Digite o caminho completo do arquivo CSV ou arraste o arquivo para aqui: ").strip('"')
    try:
        df = pd.read_csv(filepath, sep=",", parse_dates=["DT_VENDA"], dayfirst=True)
    except:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(',"', '||"').replace(',', '.').replace('||"', ',"')
        with open("temp_file.csv", "w", encoding="utf-8") as f:
            f.write(content)
        df = pd.read_csv("temp_file.csv", sep=",", parse_dates=["DT_VENDA"], dayfirst=True)
        os.remove("temp_file.csv")
    
    if 'QTD_SAIDA' in df.columns:
        df['QTD_SAIDA'] = df['QTD_SAIDA'].astype(float).apply(safe_round)
    
    df.columns = df.columns.str.strip()
    if 'QTD_SAIDA.1' in df.columns:
        df = df.drop(columns=['QTD_SAIDA.1'])
    
    df["DT_VENDA"] = pd.to_datetime(df["DT_VENDA"], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["DT_VENDA"])
    
    config = load_config()
    q1 = df["QTD_SAIDA"].quantile(config['outlier_quantiles']['lower'])
    q3 = df["QTD_SAIDA"].quantile(config['outlier_quantiles']['upper'])
    df = df[(df["QTD_SAIDA"] >= q1) & (df["QTD_SAIDA"] <= q3)]
    
    return df