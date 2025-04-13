import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .utils import load_config


def train_and_predict(df_full):
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
        config = load_config()
        datas_prev = pd.date_range(start=ultima_data + timedelta(days=1),end=pd.to_datetime(config['forecast_end_date']),freq='D')
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
        y_future = np.nan_to_num(y_future, nan=0).round().astype(int)
        
        df_forecast = pd.DataFrame({
            'data': datas_prev,
            'qtd_prevista': y_future,
            'CD_PRODUTO': produto,
            'DS_PRODUTO': df_prod['DS_PRODUTO'].iloc[0],
            'UND_MED': df_prod['UND_MED'].iloc[0]
        })
        
        all_forecasts.append(df_forecast)
    
    return pd.concat(all_forecasts)