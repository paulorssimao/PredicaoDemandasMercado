from sqlalchemy import create_engine
import pandas as pd

# Configurações do banco de dados (mude conforme necessário)
DB_URL = "oracle+cx_oracle://dwu:UpRH31S8VUg7@187.108.200.162:1521/lisandra"

# Criar conexão com o banco
engine = create_engine(DB_URL)

# Testar conexão
try:
    with engine.connect() as conn:
        print("Conexão bem-sucedida!")
except Exception as e:
    print(f"Erro ao conectar: {e}")
