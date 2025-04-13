import pandas as pd

def feature_engineering(df):
    df_grouped = df.groupby(["DT_VENDA", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["QTD_SAIDA"].sum().reset_index()
    produtos = df_grouped["CD_PRODUTO"].unique()
    full_dfs = []

    for produto in produtos:
        df_prod = df_grouped[df_grouped["CD_PRODUTO"] == produto].copy()
        df_prod = df_prod.sort_values("DT_VENDA")
        date_range = pd.date_range(start=df_prod["DT_VENDA"].min(), end=df_prod["DT_VENDA"].max(), freq='D')
        df_full = pd.DataFrame({'DT_VENDA': date_range})
        df_full = df_full.merge(df_prod, on='DT_VENDA', how='left')
        df_full["CD_PRODUTO"] = df_full["CD_PRODUTO"].fillna(produto)
        df_full["DS_PRODUTO"] = df_full["DS_PRODUTO"].fillna(df_prod["DS_PRODUTO"].iloc[0])
        df_full["UND_MED"] = df_full["UND_MED"].fillna(df_prod["UND_MED"].iloc[0])
        df_full["QTD_SAIDA"] = df_full["QTD_SAIDA"].fillna(0)
        df_full['dia_semana'] = df_full['DT_VENDA'].dt.dayofweek
        df_full['mes'] = df_full['DT_VENDA'].dt.month
        df_full['trimestre'] = df_full['DT_VENDA'].dt.quarter
        df_full['final_semana'] = df_full['dia_semana'].isin([5, 6]).astype(int)
        df_full['media_movel_7d'] = df_full['QTD_SAIDA'].rolling(window=7, min_periods=1).mean()
        df_full['media_movel_30d'] = df_full['QTD_SAIDA'].rolling(window=30, min_periods=1).mean()
        df_full = df_full.fillna(method='bfill')
        full_dfs.append(df_full)
    
    return pd.concat(full_dfs)