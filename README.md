# 📈 Previsão de Demanda de Estoque com Inteligência Artificial

Este projeto aplica **modelos de machine learning** para realizar previsões de demanda de produtos em supermercados com base em dados históricos de vendas. A solução prevê a demanda **diária**, **semanal** e **mensal** até o final de **2025**, além de gerar **relatórios analíticos e gráficos comparativos em Excel e PDF**, facilitando a tomada de decisão na gestão de estoques.

---

## 🚀 Funcionalidades

- 📅 Previsão de vendas por produto em granularidade diária, semanal e mensal  
- 🧠 Treinamento automático com Regressão Linear e Random Forest  
- 📊 Geração de relatórios em Excel com gráficos comparativos  
- 📉 Cálculo automático de estoque mínimo, ideal e máximo com base na previsão  
- 📁 Comparação entre valores reais e previstos (se disponíveis)

---

## 🛠️ Requisitos do Ambiente

- Python **3.9** ou superior  
- Sistema operacional: Windows, Linux ou macOS  
- Biblioteca `pip` para instalação de dependências

---

## 📦 Instalação

Clone este repositório e instale as dependências necessárias com:

```bash
git clone https://github.com/seu-usuario/nome-do-projeto.git
cd nome-do-projeto
pip install -r requirements.txt
```

---

## 📝 Como Utilizar

1. Coloque o arquivo CSV de vendas históricas no diretório do projeto.  
2. Execute o script principal:

```bash
python previsao_estoque.py
```

3. Siga as instruções no terminal para carregar os dados e gerar as previsões.

4. Os arquivos de saída serão gerados automaticamente nas pastas de relatório.

---

## 📂 Estrutura Esperada do CSV

O arquivo de entrada deve conter as seguintes colunas:

- `DT_VENDA`: data da venda (formato DD/MM/AAAA)  
- `CD_PRODUTO`: código do produto  
- `DS_PRODUTO`: descrição do produto  
- `UND_MED`: unidade de medida  
- `QTD_SAIDA`: quantidade vendida  

---

## 📤 Saídas Geradas

- `previsao_comparativo_2025.xlsx`: planilha com previsões, estoques e análises comparativas  
- Gráficos de linha por produto: previsão vs. demanda real  
- Resumo estatístico impresso no terminal

---

## 📚 Tecnologias e Bibliotecas

- `Pandas` – manipulação de dados  
- `NumPy` – operações numéricas  
- `Scikit-learn` – modelagem preditiva  
- `OpenPyXL` – geração de relatórios Excel  
- `Matplotlib` – gráficos (opcional)

---

## 🤝 Contribuidores

- Paulo Roberto Simão  
- Rubens Scotti Junior  
- Thyerri Mezzari
