import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def safe_round(value):
    """Arredonda valores tratando NaN e None"""
    if pd.isna(value) or value is None:
        return 0
    return int(round(value))

def load_and_preprocess_data():
    """Carrega e trata os dados de vendas com interação do usuário"""
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
    
    q1 = df["QTD_SAIDA"].quantile(0.05)
    q3 = df["QTD_SAIDA"].quantile(0.95)
    df = df[(df["QTD_SAIDA"] >= q1) & (df["QTD_SAIDA"] <= q3)]
    
    return df

def feature_engineering(df):
    """Cria features para melhorar as previsões"""
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

def train_and_predict(df_full):
    """Treina modelos e faz previsões"""
    produtos = df_full["CD_PRODUTO"].unique()
    all_forecasts = []
    
    for produto in produtos:
        df_prod = df_full[df_full["CD_PRODUTO"] == produto].copy()
        df_prod = df_prod.sort_values("DT_VENDA")
        
        X = df_prod[['dia_semana', 'mes', 'trimestre', 'final_semana', 'media_movel_7d', 'media_movel_30d']]
        y = df_prod["QTD_SAIDA"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            if score > best_score:
                best_score = score
                best_model = model
        
        ultima_data = df_prod["DT_VENDA"].max()
        datas_prev = pd.date_range(start=ultima_data + timedelta(days=1), end=datetime(2025, 12, 31), freq='D')
        
        future_data = pd.DataFrame({'DT_VENDA': datas_prev})
        future_data['dia_semana'] = future_data['DT_VENDA'].dt.dayofweek
        future_data['mes'] = future_data['DT_VENDA'].dt.month
        future_data['trimestre'] = future_data['DT_VENDA'].dt.quarter
        future_data['final_semana'] = future_data['dia_semana'].isin([5, 6]).astype(int)
        
        last_7d_avg = df_prod['media_movel_7d'].iloc[-1]
        last_30d_avg = df_prod['media_movel_30d'].iloc[-1]
        future_data['media_movel_7d'] = last_7d_avg
        future_data['media_movel_30d'] = last_30d_avg
        
        X_future = future_data[['dia_semana', 'mes', 'trimestre', 'final_semana', 'media_movel_7d', 'media_movel_30d']]
        X_future_scaled = scaler.transform(X_future)
        
        y_future = best_model.predict(X_future_scaled)
        y_future = np.clip(y_future, 0, None)
        y_future = np.nan_to_num(y_future, nan=0)  # Trata NaN
        y_future = y_future.round().astype(int)
        
        df_forecast = pd.DataFrame({
            'data': datas_prev,
            'qtd_prevista': y_future,
            'CD_PRODUTO': produto,
            'DS_PRODUTO': df_prod['DS_PRODUTO'].iloc[0],
            'UND_MED': df_prod['UND_MED'].iloc[0]
        })
        
        all_forecasts.append(df_forecast)
    
    return pd.concat(all_forecasts)

def calculate_inventory_levels(demand, lead_time=7, service_level=0.95):
    """Calcula estoque mínimo, ideal e máximo com tratamento de NaN"""
    if isinstance(demand, pd.Series):
        avg_demand = demand.mean()
        std_demand = demand.std()
    else:
        avg_demand = demand
        std_demand = demand * 0.3
    
    # Tratamento para valores NaN/infinitos
    if pd.isna(avg_demand) or np.isinf(avg_demand):
        avg_demand = 0
    if pd.isna(std_demand) or np.isinf(std_demand):
        std_demand = 0
    
    z = 1.645
    
    safety_stock = z * np.sqrt(lead_time) * std_demand
    estoque_minimo = (avg_demand * lead_time/30) + safety_stock
    estoque_ideal = (avg_demand * (lead_time/30 + 1)) + safety_stock
    estoque_maximo = estoque_ideal + (avg_demand * 0.5)
    
    return {
        'estoque_minimo': max(estoque_minimo, 0),
        'estoque_ideal': max(estoque_ideal, 0),
        'estoque_maximo': max(estoque_maximo, 0)
    }

def generate_excel_report(df_forecasts, output_file="analise_estoque_completo.xlsx"):
    """Gera relatório Excel com tratamento de NaN"""
    wb = Workbook()
    
    # 1. ANÁLISE MENSAL
    ws_monthly = wb.create_sheet("Análise Mensal")
    df_monthly = df_forecasts.copy()
    df_monthly["mes"] = df_monthly["data"].dt.to_period("M").dt.to_timestamp()
    monthly_grouped = df_monthly.groupby(["mes", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["qtd_prevista"]
    
    inventory_data = []
    for (mes, produto, nome, unidade), group in monthly_grouped:
        levels = calculate_inventory_levels(group)
        inventory_data.append({
            'Mês': mes,
            'Código Produto': produto,
            'Produto': nome,
            'Unidade': unidade,
            'Demanda Prevista': safe_round(group.sum()),
            'Estoque Mínimo': safe_round(levels['estoque_minimo']),
            'Estoque Ideal': safe_round(levels['estoque_ideal']),
            'Estoque Máximo': safe_round(levels['estoque_maximo'])
        })
    
    df_monthly_report = pd.DataFrame(inventory_data)
    for r in dataframe_to_rows(df_monthly_report, index=False, header=True):
        ws_monthly.append(r)
    
    # 2. ANÁLISE SEMANAL
    ws_weekly = wb.create_sheet("Análise Semanal")
    df_weekly = df_forecasts.copy()
    df_weekly["semana"] = df_weekly["data"].dt.to_period("W").dt.to_timestamp()
    weekly_grouped = df_weekly.groupby(["semana", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["qtd_prevista"]
    
    inventory_data = []
    for (semana, produto, nome, unidade), group in weekly_grouped:
        levels = calculate_inventory_levels(group, lead_time=7)
        inventory_data.append({
            'Semana': semana,
            'Código Produto': produto,
            'Produto': nome,
            'Unidade': unidade,
            'Demanda Prevista': safe_round(group.sum()),
            'Estoque Mínimo': safe_round(levels['estoque_minimo']),
            'Estoque Ideal': safe_round(levels['estoque_ideal']),
            'Estoque Máximo': safe_round(levels['estoque_maximo'])
        })
    
    df_weekly_report = pd.DataFrame(inventory_data)
    for r in dataframe_to_rows(df_weekly_report, index=False, header=True):
        ws_weekly.append(r)
    
    # 3. RESUMO GERAL
    ws_summary = wb.create_sheet("Resumo")
    produtos = df_forecasts["CD_PRODUTO"].unique()
    summary_data = []
    
    for produto in produtos:
        df_prod = df_forecasts[df_forecasts["CD_PRODUTO"] == produto].copy()
        nome_produto = df_prod["DS_PRODUTO"].iloc[0]
        unidade = df_prod["UND_MED"].iloc[0]
        
        monthly = df_prod.resample('ME', on='data')['qtd_prevista'].sum()
        monthly_levels = calculate_inventory_levels(monthly)
        
        weekly = df_prod.resample('W', on='data')['qtd_prevista'].sum()
        weekly_levels = calculate_inventory_levels(weekly, lead_time=7)
        
        summary_data.append({
            'Código Produto': produto,
            'Produto': nome_produto,
            'Unidade': unidade,
            'Demanda Média Mensal': safe_round(monthly.mean()),
            'Estoque Mín Mensal': safe_round(monthly_levels['estoque_minimo']),
            'Estoque Ideal Mensal': safe_round(monthly_levels['estoque_ideal']),
            'Estoque Max Mensal': safe_round(monthly_levels['estoque_maximo']),
            'Demanda Média Semanal': safe_round(weekly.mean()),
            'Estoque Mín Semanal': safe_round(weekly_levels['estoque_minimo']),
            'Estoque Ideal Semanal': safe_round(weekly_levels['estoque_ideal']),
            'Estoque Max Semanal': safe_round(weekly_levels['estoque_maximo'])
        })
    
    df_summary = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(df_summary, index=False, header=True):
        ws_summary.append(r)
    
    # Formatar cabeçalhos
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for col in ws.columns:
            max_length = max(len(str(cell.value)) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_length + 2
    
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    
    wb.save(output_file)
    print(f"✅ Relatório gerado: {output_file}")

if __name__ == "__main__":
    print("🔍 Processando dados...")
    df = load_and_preprocess_data()
    
    print("📊 Criando features...")
    df_full = feature_engineering(df)
    
    print("🤖 Treinando modelos e prevendo demanda...")
    df_forecasts = train_and_predict(df_full)
    
    print("📈 Gerando relatório Excel...")
    generate_excel_report(df_forecasts)
    
    print("✅ Análise concluída com sucesso!")