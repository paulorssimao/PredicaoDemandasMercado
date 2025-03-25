# Importando as bibliotecas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Vamos usar a data como variável de entrada (feature) e a quantidade de vendas como o target
df_demanda["Data_num"] = df_demanda["DT_VENDA"].apply(lambda x: x.toordinal())  # Converte a data para um número

# Separando as features e o target
X = df_demanda[["Data_num"]]  # Feature: data convertida para número
y = df_demanda["QTD_VENDA"]  # Target: quantidade vendida

# Dividindo em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Calculando o erro (quantos itens o modelo erra, em média)
erro = mean_absolute_error(y_test, y_pred)
print(f"Erro médio absoluto: {erro}")

# Exibindo o gráfico de previsões
plt.plot(X_test, y_test, color='blue', label='Vendas reais')  # Vendas reais
plt.plot(X_test, y_pred, color='red', label='Vendas previstas')  # Previsões do modelo
plt.xlabel("Data")
plt.ylabel("Quantidade Vendida")
plt.title("Vendas Reais vs Vendas Previstas")
plt.legend()
plt.show()
