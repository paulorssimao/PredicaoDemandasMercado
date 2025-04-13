from src.data_loader import load_and_preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_and_predict
from src.report_generator import generate_excel_report

if __name__ == "__main__":
    print("🔍 Processando dados...")
    df = load_and_preprocess_data()

    print("🔎 Prévia dos dados carregados:")
    print(df.head())

    print("📊 Criando features...")
    df_full = feature_engineering(df)

    print("✅ Features criadas:")
    print(df_full.head())

    print("🤖 Treinando modelos e prevendo demanda...")
    df_forecasts = train_and_predict(df_full)

    print("📈 Previsões geradas:")
    print(df_forecasts.head())

    print("📦 Gerando Excel com linhas:", len(df_forecasts))
    generate_excel_report(df_forecasts)

    print("✅ Análise concluída com sucesso!")