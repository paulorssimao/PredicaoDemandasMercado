# Previsão de Demanda - Análise de Faturamento e Estoque

Este projeto visa prever a demanda de vendas com base em dados históricos de faturamento e controle de estoque. Ele faz uso de regressão linear para prever as vendas futuras.

## Estrutura de Arquivos

- **ler_csv.py** - Script para carregar e analisar os dados de faturamento e controle de estoque a partir de arquivos CSV.
- **treinamento.py** - Script para treinar um modelo de regressão linear para prever a demanda (quantidade vendida).
- **VM_CONTROLE_ESTOQUE.CSV** - Arquivo CSV contendo os dados de controle de estoque.
- **VM_FATURAMENTO_AGRUP.CSV** - Arquivo CSV contendo os dados de faturamento.

## Requisitos

Antes de rodar o projeto, você precisa garantir que as dependências do Python estão instaladas. Você pode instalar as dependências necessárias com o `pip`.

### Passo 1: Instalar o Python
Certifique-se de que o Python 3.x está instalado. Caso não tenha o Python instalado, baixe e instale a versão mais recente.

### Passo 2: Criar um Ambiente Virtual
Para evitar conflitos com pacotes do sistema, é recomendado criar um ambiente virtual. Execute os seguintes comandos:

1. No terminal, navegue até o diretório do seu projeto.
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```
3. Ative o ambiente virtual:
   - No Windows:
     ```bash
     venv\Scripts\activate
     ```
   - No Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

### Passo 3: Instalar Dependências

Crie um arquivo chamado `requirements.txt` com as seguintes dependências:

```
pandas
scikit-learn
matplotlib
```

Em seguida, execute o comando para instalar todas as dependências:

```bash
pip install -r requirements.txt
```

### Passo 4: Organize os Arquivos CSV

Coloque os arquivos CSV (`VM_FATURAMENTO_AGRUP.CSV` e `VM_CONTROLE_ESTOQUE.CSV`) na mesma pasta que os scripts.

## Como Rodar o Projeto

### Passo 1: Carregar e Analisar os Dados (`ler_csv.py`)

O script `ler_csv.py` carrega e exibe as primeiras linhas dos arquivos CSV para uma análise preliminar dos dados. Ele também verifica a existência de valores ausentes e agrupa os dados de vendas por data.

Para rodar este script, execute o seguinte comando no terminal:

```bash
python ler_csv.py
```

### Passo 2: Treinamento e Previsão (`treinamento.py`)

O script `treinamento.py` treina um modelo de regressão linear para prever as vendas futuras com base na data de venda. Ele também calcula o erro médio das previsões e exibe um gráfico comparando as vendas reais e previstas.

Para rodar este script, execute o seguinte comando:

```bash
python treinamento.py
```

## Como Funciona

- **`ler_csv.py`**: Este script carrega os arquivos CSV usando a biblioteca `pandas` e exibe informações sobre os dados (primeiras linhas, colunas e valores ausentes).
- **`treinamento.py`**: Este script usa a regressão linear para treinar um modelo de previsão com base nos dados históricos de vendas. Ele usa a data (`DT_VENDA`) como uma feature numérica para prever a quantidade de vendas (`QTD_VENDA`). O modelo é então avaliado e o erro médio é exibido.

