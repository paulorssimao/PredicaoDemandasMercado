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

def safe_round(value, default=0):
    """Arredonda valores tratando NaN e None"""
    try:
        if pd.isna(value) or value is None:
            return default
        return int(round(float(value)))
    except:
        return default

def load_and_preprocess_data():
    """Carrega e trata os dados de vendas com interação do usuário"""
    print("\n📂 PASSO 1: Carregar arquivo CSV")
    print("Por favor, forneça o arquivo CSV com as colunas: data venda, descrição produto, unidade medida, quantidade venda")
    filepath = input("Digite o caminho completo do arquivo CSV ou arraste o arquivo para aqui: ").strip('"')
    
    try:
        # Tentar ler normalmente primeiro
        df = pd.read_csv(filepath, sep=",", parse_dates=["DT_VENDA"], dayfirst=True)
    except Exception as e:
        # Se falhar, tratar vírgulas decimais
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace(',"', '||"').replace(',', '.').replace('||"', ',"')
            with open("temp_file.csv", "w", encoding="utf-8") as f:
                f.write(content)
            df = pd.read_csv("temp_file.csv", sep=",", parse_dates=["DT_VENDA"], dayfirst=True)
            os.remove("temp_file.csv")
        except:
            print(f"❌ Erro ao ler o arquivo: {e}")
            return None
    
    # Verificar colunas necessárias
    required_cols = ['DT_VENDA', 'DS_PRODUTO', 'UND_MED', 'QTD_SAIDA']
    if not all(col in df.columns for col in required_cols):
        print("❌ O arquivo CSV não contém todas as colunas necessárias")
        print(f"Colunas necessárias: {required_cols}")
        print(f"Colunas encontradas: {df.columns.tolist()}")
        return None
    
    # Arredondar valores e tratar casos especiais
    df['QTD_SAIDA'] = df['QTD_SAIDA'].apply(lambda x: safe_round(x))
    
    # Limpeza dos dados
    df.columns = df.columns.str.strip()
    df["DT_VENDA"] = pd.to_datetime(df["DT_VENDA"], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["DT_VENDA"])
    
    # Remover outliers
    q1 = df["QTD_SAIDA"].quantile(0.05)
    q3 = df["QTD_SAIDA"].quantile(0.95)
    df = df[(df["QTD_SAIDA"] >= q1) & (df["QTD_SAIDA"] <= q3)]
    
    print("✅ Dados carregados e pré-processados com sucesso")
    return df

def get_end_date():
    """Obtém a data final para análise"""
    print("\n📅 PASSO 2: Definir data final para previsão")
    while True:
        end_date_str = input("Digite a data final para análise (DD/MM/AAAA, ex: 31/12/2025): ")
        try:
            end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
            print(f"✅ Data final definida: {end_date.strftime('%d/%m/%Y')}")
            return end_date
        except ValueError:
            print("❌ Formato inválido. Por favor, use DD/MM/AAAA")

def get_lead_times(products):
    """Obtém os tempos de reposição para cada produto"""
    print("\n⏱️ PASSO 3: Tempo médio de reposição por produto")
    lead_times = {}
    for product in products:
        while True:
            try:
                days = int(input(f"Tempo médio de reposição (em dias) para {product}: "))
                if days <= 0:
                    raise ValueError
                lead_times[product] = days
                break
            except ValueError:
                print("❌ Por favor, digite um número inteiro positivo")
    return lead_times

def feature_engineering(df):
    """Cria features para melhorar as previsões"""
    print("\n🔧 Criando features para análise...")
    df_grouped = df.groupby(["DT_VENDA", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["QTD_SAIDA"].sum().reset_index()
    produtos = df_grouped["CD_PRODUTO"].unique()
    full_dfs = []
    
    for produto in produtos:
        df_prod = df_grouped[df_grouped["CD_PRODUTO"] == produto].copy()
        df_prod = df_prod.sort_values("DT_VENDA")
        
        # Preencher datas faltantes
        date_range = pd.date_range(start=df_prod["DT_VENDA"].min(), end=df_prod["DT_VENDA"].max(), freq='D')
        df_full = pd.DataFrame({'DT_VENDA': date_range})
        df_full = df_full.merge(df_prod, on='DT_VENDA', how='left')
        df_full["CD_PRODUTO"] = df_full["CD_PRODUTO"].fillna(produto)
        df_full["DS_PRODUTO"] = df_full["DS_PRODUTO"].fillna(df_prod["DS_PRODUTO"].iloc[0])
        df_full["UND_MED"] = df_full["UND_MED"].fillna(df_prod["UND_MED"].iloc[0])
        df_full["QTD_SAIDA"] = df_full["QTD_SAIDA"].fillna(0)
        
        # Engenharia de características
        df_full['dia_semana'] = df_full['DT_VENDA'].dt.dayofweek
        df_full['mes'] = df_full['DT_VENDA'].dt.month
        df_full['trimestre'] = df_full['DT_VENDA'].dt.quarter
        df_full['final_semana'] = df_full['dia_semana'].isin([5, 6]).astype(int)
        
        # Médias móveis
        df_full['media_movel_7d'] = df_full['QTD_SAIDA'].rolling(window=7, min_periods=1).mean()
        df_full['media_movel_30d'] = df_full['QTD_SAIDA'].rolling(window=30, min_periods=1).mean()
        df_full = df_full.fillna(method='bfill')
        
        full_dfs.append(df_full)
    
    return pd.concat(full_dfs)

def train_and_predict(df_full, end_date):
    """Treina modelos e faz previsões"""
    print("\n🤖 Treinando modelos de Machine Learning...")
    produtos = df_full["CD_PRODUTO"].unique()
    all_forecasts = []
    
    for produto in produtos:
        df_prod = df_full[df_full["CD_PRODUTO"] == produto].copy()
        df_prod = df_prod.sort_values("DT_VENDA")
        
        X = df_prod[['dia_semana', 'mes', 'trimestre', 'final_semana', 'media_movel_7d', 'media_movel_30d']]
        y = df_prod["QTD_SAIDA"]
        
        # Dividir em treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Normalizar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Testar modelos
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
        
        # Prever demanda futura
        ultima_data = df_prod["DT_VENDA"].max()
        datas_prev = pd.date_range(start=ultima_data + timedelta(days=1), end=end_date, freq='D')
        
        future_data = pd.DataFrame({'DT_VENDA': datas_prev})
        future_data['dia_semana'] = future_data['DT_VENDA'].dt.dayofweek
        future_data['mes'] = future_data['DT_VENDA'].dt.month
        future_data['trimestre'] = future_data['DT_VENDA'].dt.quarter
        future_data['final_semana'] = future_data['dia_semana'].isin([5, 6]).astype(int)
        
        # Usar últimas médias móveis conhecidas
        last_7d_avg = df_prod['media_movel_7d'].iloc[-1]
        last_30d_avg = df_prod['media_movel_30d'].iloc[-1]
        future_data['media_movel_7d'] = last_7d_avg
        future_data['media_movel_30d'] = last_30d_avg
        
        X_future = future_data[['dia_semana', 'mes', 'trimestre', 'final_semana', 'media_movel_7d', 'media_movel_30d']]
        X_future_scaled = scaler.transform(X_future)
        
        y_future = best_model.predict(X_future_scaled)
        y_future = np.clip(y_future, 0, None)
        y_future = np.nan_to_num(y_future, nan=0)
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

def calculate_inventory_levels(demand, lead_time_days=7, service_level=0.95):
    """Calcula os níveis de estoque"""
    z_values = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
    z = z_values.get(service_level, 1.645)
    
    if isinstance(demand, pd.Series):
        avg_demand = np.nanmean(demand)
        std_demand = np.nanstd(demand)
        
        if std_demand == 0:
            std_demand = avg_demand * 0.1
    else:
        avg_demand = demand if not pd.isna(demand) else 0
        std_demand = avg_demand * 0.3
    
    lead_time_month = lead_time_days / 30
    safety_stock = z * np.sqrt(lead_time_month) * std_demand
    estoque_minimo = (avg_demand * lead_time_month) + safety_stock
    lote_economico = avg_demand * 0.5
    estoque_ideal = estoque_minimo + lote_economico
    estoque_maximo = estoque_ideal + (avg_demand * 0.3)
    demanda_diaria = avg_demand / 30
    estoque_minimo = max(estoque_minimo, demanda_diaria * lead_time_days)
    
    return {
        'estoque_minimo': safe_round(estoque_minimo),
        'estoque_ideal': safe_round(estoque_ideal),
        'estoque_maximo': safe_round(estoque_maximo)
    }

def generate_excel_report(df_forecasts, lead_times, output_file="cronograma_estoque.xlsx"):
    """Gera relatório Excel completo"""
    print("\n📊 Gerando relatório Excel...")
    wb = Workbook()
    
    # 1. ANÁLISE MENSAL
    ws_monthly = wb.create_sheet("Mensal")
    df_monthly = df_forecasts.copy()
    df_monthly["mes"] = df_monthly["data"].dt.to_period("M").dt.to_timestamp()
    monthly_grouped = df_monthly.groupby(["mes", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["qtd_prevista"]
    
    inventory_data = []
    for (mes, produto, nome, unidade), group in monthly_grouped:
        lead_time = lead_times.get(produto, 7)
        levels = calculate_inventory_levels(group, lead_time_days=lead_time)
        inventory_data.append({
            'Mês': mes.strftime('%Y-%m'),
            'Código Produto': produto,
            'Produto': nome,
            'Unidade': unidade,
            'Demanda Prevista': safe_round(group.sum()),
            'Estoque Mínimo': levels['estoque_minimo'],
            'Estoque Ideal': levels['estoque_ideal'],
            'Estoque Máximo': levels['estoque_maximo']
        })
    
    df_monthly_report = pd.DataFrame(inventory_data)
    for r in dataframe_to_rows(df_monthly_report, index=False, header=True):
        ws_monthly.append(r)
    
    # 2. ANÁLISE SEMANAL
    ws_weekly = wb.create_sheet("Semanal")
    df_weekly = df_forecasts.copy()
    df_weekly["semana"] = df_weekly["data"].dt.to_period("W").dt.to_timestamp()
    weekly_grouped = df_weekly.groupby(["semana", "CD_PRODUTO", "DS_PRODUTO", "UND_MED"])["qtd_prevista"]
    
    inventory_data = []
    for (semana, produto, nome, unidade), group in weekly_grouped:
        lead_time = lead_times.get(produto, 7)
        levels = calculate_inventory_levels(group, lead_time_days=lead_time)
        inventory_data.append({
            'Semana': semana.strftime('%Y-%m-%d'),
            'Código Produto': produto,
            'Produto': nome,
            'Unidade': unidade,
            'Demanda Prevista': safe_round(group.sum()),
            'Estoque Mínimo': levels['estoque_minimo'],
            'Estoque Ideal': levels['estoque_ideal'],
            'Estoque Máximo': levels['estoque_maximo']
        })
    
    df_weekly_report = pd.DataFrame(inventory_data)
    for r in dataframe_to_rows(df_weekly_report, index=False, header=True):
        ws_weekly.append(r)
    
    # 3. RESUMO
    ws_summary = wb.create_sheet("Resumo")
    produtos = df_forecasts["CD_PRODUTO"].unique()
    summary_data = []
    
    for produto in produtos:
        df_prod = df_forecasts[df_forecasts["CD_PRODUTO"] == produto].copy()
        nome_produto = df_prod["DS_PRODUTO"].iloc[0]
        unidade = df_prod["UND_MED"].iloc[0]
        lead_time = lead_times.get(produto, 7)
        
        # Estatísticas mensais
        monthly = df_prod.resample('ME', on='data')['qtd_prevista'].sum()
        monthly_levels = calculate_inventory_levels(monthly, lead_time_days=lead_time)
        
        # Estatísticas semanais
        weekly = df_prod.resample('W', on='data')['qtd_prevista'].sum()
        weekly_levels = calculate_inventory_levels(weekly, lead_time_days=lead_time)
        
        # Identificar picos e vales
        monthly_peaks = monthly.nlargest(3).index.strftime('%Y-%m').tolist()
        monthly_lows = monthly.nsmallest(3).index.strftime('%Y-%m').tolist()
        weekly_peaks = weekly.nlargest(3).index.strftime('%Y-%m-%d').tolist()
        weekly_lows = weekly.nsmallest(3).index.strftime('%Y-%m-%d').tolist()
        
        summary_data.append({
            'Código Produto': produto,
            'Produto': nome_produto,
            'Unidade': unidade,
            'Tempo Reposição (dias)': lead_time,
            'Meses Alta Demanda': ', '.join(monthly_peaks),
            'Meses Baixa Demanda': ', '.join(monthly_lows),
            'Semanas Alta Demanda': ', '.join(weekly_peaks),
            'Semanas Baixa Demanda': ', '.join(weekly_lows),
            'Estoque Ideal Mensal': monthly_levels['estoque_ideal'],
            'Estoque Ideal Semanal': weekly_levels['estoque_ideal']
        })
    
    df_summary = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(df_summary, index=False, header=True):
        ws_summary.append(r)
    
    # Formatar planilhas
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for col in ws.columns:
            max_length = max(len(str(cell.value)) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_length + 2
    
    # Remover sheet vazio padrão
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    
    wb.save(output_file)
    print(f"✅ Relatório gerado com sucesso: {output_file}")

def main():
    print("""
    ====================================
    SISTEMA DE PREVISÃO DE DEMANDA E ESTOQUE
    ====================================
    """)
    
    # Passo 1: Carregar dados
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Passo 2: Obter data final
    end_date = get_end_date()
    
    # Passo 3: Engenharia de features
    df_full = feature_engineering(df)
    
    # Passo 4: Modelagem e previsão
    df_forecasts = train_and_predict(df_full, end_date)
    
    # Passo 5: Obter tempos de reposição
    produtos = df_full["CD_PRODUTO"].unique()
    produtos_desc = {p: df_full[df_full["CD_PRODUTO"] == p]["DS_PRODUTO"].iloc[0] for p in produtos}
    print("\n📝 Informe o tempo médio de reposição para cada produto (em dias):")
    lead_times = {}
    for p in produtos:
        while True:
            try:
                days = int(input(f"Tempo de reposição para {produtos_desc[p]} ({p}): "))
                if days <= 0:
                    raise ValueError
                lead_times[p] = days
                break
            except ValueError:
                print("Por favor, digite um número inteiro positivo")
    
    # Passo 6: Gerar relatório
    generate_excel_report(df_forecasts, lead_times)
    
    print("\n✅ Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
