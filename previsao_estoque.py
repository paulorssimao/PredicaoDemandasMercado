import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import os
from matplotlib.backends.backend_pdf import PdfPages


# === 1. Ler o CSV com segurança ===
df = pd.read_csv("datasheet_vendas_pfc.csv", sep=",", encoding="utf-8", skipinitialspace=True, on_bad_lines='skip')

# === 2. Corrigir colunas duplicadas, se houver ===
df.columns = df.columns.str.strip()
if 'QTD_SAIDA.1' in df.columns:
    df = df.drop(columns=['QTD_SAIDA.1'])

# === 3. Converter datas ===
df["DT_VENDA"] = pd.to_datetime(df["DT_VENDA"], errors='coerce', dayfirst=True)
df = df.dropna(subset=["DT_VENDA"])  # remover linhas com data inválida

# === 4. Agrupar vendas por produto e data ===
df_grouped = df.groupby(["DT_VENDA", "CD_PRODUTO", "DS_PRODUTO"])["QTD_SAIDA"].sum().reset_index()

# === 5. Previsão por produto usando regressão linear ===
produtos = df_grouped["CD_PRODUTO"].unique()
previsoes = []

for produto in produtos:
    df_prod = df_grouped[df_grouped["CD_PRODUTO"] == produto].copy()
    df_prod = df_prod.sort_values("DT_VENDA")

    df_prod["dias"] = (df_prod["DT_VENDA"] - df_prod["DT_VENDA"].min()).dt.days
    X = df_prod[["dias"]]
    y = df_prod["QTD_SAIDA"]

    model = LinearRegression()
    model.fit(X, y)

    # Gerar datas até o fim de 2025
    ultima_data = df_prod["DT_VENDA"].max()
    dias_ate_final_ano = (datetime(2025, 12, 31) - ultima_data).days
    datas_prev = [ultima_data + timedelta(days=i) for i in range(1, dias_ate_final_ano + 1)]
    dias_prev = [(d - df_prod["DT_VENDA"].min()).days for d in datas_prev]

    y_pred = model.predict(np.array(dias_prev).reshape(-1, 1))
    y_pred = np.clip(y_pred, 0, None)

    df_forecast = pd.DataFrame({
        "data": datas_prev,
        "qtd_prevista": y_pred,
        "CD_PRODUTO": produto,
        "DS_PRODUTO": df_prod["DS_PRODUTO"].iloc[0]
    })

    previsoes.append(df_forecast)

# === 6. Unir previsões ===
df_all_forecasts = pd.concat(previsoes)

# === 7. Agregações ===
# Diária
df_diaria = df_all_forecasts.copy()

# Semanal
df_semanal = df_all_forecasts.copy()
df_semanal["semana"] = df_semanal["data"].dt.to_period("W").apply(lambda r: r.start_time)
df_semanal = df_semanal.groupby(["semana", "CD_PRODUTO", "DS_PRODUTO"])["qtd_prevista"].sum().reset_index()

# Mensal
df_mensal = df_all_forecasts.copy()
df_mensal["mes"] = df_mensal["data"].dt.to_period("M").apply(lambda r: r.start_time)
df_mensal = df_mensal.groupby(["mes", "CD_PRODUTO", "DS_PRODUTO"])["qtd_prevista"].sum().reset_index()

# === 8. Criar diretórios para gráficos ===
os.makedirs("graficos_semanal", exist_ok=True)
os.makedirs("graficos_mensal", exist_ok=True)

# === 9. Gerar gráficos semanais ===
for produto in df_semanal["CD_PRODUTO"].unique():
    df_plot = df_semanal[df_semanal["CD_PRODUTO"] == produto]

    plt.figure(figsize=(10, 4))
    plt.plot(df_plot["semana"], df_plot["qtd_prevista"], marker="o")
    plt.title(f"Previsão SEMANAL até 2025\nProduto: {df_plot['DS_PRODUTO'].iloc[0]}")
    plt.xlabel("Semana")
    plt.ylabel("Quantidade Prevista")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_semanal/previsao_semanal_{produto}.png")
    plt.close()

# === 10. Gerar gráficos mensais ===
for produto in df_mensal["CD_PRODUTO"].unique():
    df_plot = df_mensal[df_mensal["CD_PRODUTO"] == produto]

    plt.figure(figsize=(10, 4))
    plt.plot(df_plot["mes"], df_plot["qtd_prevista"], marker="o", color="green")
    plt.title(f"Previsão MENSAL até 2025\nProduto: {df_plot['DS_PRODUTO'].iloc[0]}")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade Prevista")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_mensal/previsao_mensal_{produto}.png")
    plt.close()

print("✅ Previsões geradas com sucesso!")
print("Gráficos salvos nas pastas: 'graficos_semanal' e 'graficos_mensal'")

# Criar pasta para os relatórios PDF
os.makedirs("relatorios_pdf", exist_ok=True)

# Gerar 1 PDF por produto com os 2 gráficos (semanal + mensal)
for produto in df_semanal["CD_PRODUTO"].unique():
    nome_produto = df_semanal[df_semanal["CD_PRODUTO"] == produto]["DS_PRODUTO"].iloc[0]

    pdf_path = f"relatorios_pdf/relatorio_previsao_{produto}.pdf"
    with PdfPages(pdf_path) as pdf:
        # --- Gráfico semanal ---
        df_plot_sem = df_semanal[df_semanal["CD_PRODUTO"] == produto]
        plt.figure(figsize=(10, 4))
        plt.plot(df_plot_sem["semana"], df_plot_sem["qtd_prevista"], marker="o")
        plt.title(f"Previsão SEMANAL até 2025\nProduto: {nome_produto}")
        plt.xlabel("Semana")
        plt.ylabel("Quantidade Prevista")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Gráfico mensal ---
        df_plot_mes = df_mensal[df_mensal["CD_PRODUTO"] == produto]
        plt.figure(figsize=(10, 4))
        plt.plot(df_plot_mes["mes"], df_plot_mes["qtd_prevista"], marker="o", color="green")
        plt.title(f"Previsão MENSAL até 2025\nProduto: {nome_produto}")
        plt.xlabel("Mês")
        plt.ylabel("Quantidade Prevista")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("📄 PDFs com os gráficos foram gerados em: relatorios_pdf/")
