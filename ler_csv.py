import pandas as pd

# Carregar os arquivos CSV
df_faturamento = pd.read_csv("VM_FATURAMENTO_AGRUP.csv", sep=";", encoding="utf-8")
df_estoque = pd.read_csv("VM_CONTROLE_ESTOQUE.csv", sep=";", encoding="utf-8")

# Exibir as primeiras linhas de cada dataset
print("Faturamento:")
print(df_faturamento.head())

print("\nEstoque:")
print(df_estoque.head())

# Exibir nomes das colunas
print("\nColunas do Faturamento:")
print(df_faturamento.columns)

print("\nColunas do Estoque:")
print(df_estoque.columns)

# Verificar valores ausentes
print("\nValores ausentes no Faturamento:")
print(df_faturamento.isnull().sum())

print("\nValores ausentes no Estoque:")
print(df_estoque.isnull().sum())

# Agrupar as vendas por data para analisar a demanda diária
df_demanda = df_faturamento.groupby("DT_VENDA")["QTD_VENDA"].sum().reset_index()

print("\nDemanda agregada por dia:")
print(df_demanda.head())