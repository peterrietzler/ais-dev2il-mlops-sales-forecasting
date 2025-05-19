import logging
import mlflow.prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, 
                             mean_absolute_percentage_error, 
                             median_absolute_error)
import mlflow
import os

import prophet
from prophet import Prophet
from datetime import datetime

_logger = logging.getLogger(__name__)

def _prepare_store_data(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.rename(columns= {'Date': 'ds', 'Sales': 'y'})
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == 1)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True)

def _train_model(
    df_train: pd.DataFrame,
    seasonality: dict 
) -> prophet.forecaster.Prophet:
    model=Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width = 0.95
    )
    model.fit(df_train)
    return model

def train_test_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index = int(train_fraction*df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]
    return df_train, df_test

def download_latest_sales_data() -> pd.DataFrame:
    # just get data shipped with the package for demo purposes
    module_dir = os.path.dirname(os.path.abspath(__file__))    
    return pd.read_csv(os.path.join(module_dir, "latest_sales_data.csv"))


def train_and_register(df: pd.DataFrame, store_id: int) -> bool:    
    current_date = datetime.now().strftime("%Y-%m-%d")
    mlflow.set_experiment(f"sales-forecasting-{current_date}")
    with mlflow.start_run(run_name=f"store-{store_id}"):
        store_data = _prepare_store_data(df, store_id)
        seasonality = {
            'yearly': True,
            'weekly': True,
            'daily': False
        }
        logging.info('Training model for store %d', store_id)
        df_train, df_test = train_test_split(store_data, 0.8)
        model = _train_model(df_train, seasonality)
        input_example = df_test[['ds']].head(5)
        mlflow.prophet.log_model(model, 
                                 artifact_path="model", 
                                 input_example=input_example, 
                                 signature=mlflow.models.infer_signature(input_example, model.predict(input_example)))
        mlflow.log_params(seasonality)
        mlflow.log_metrics(
                {
                    'mean_abs_perc_error': mean_absolute_percentage_error(y_true=df_test['y'], y_pred=model.predict(df_test)['yhat']),
                    'mean_abs_error': mean_absolute_error(y_true=df_test['y'], y_pred=model.predict(df_test)['yhat']),
                    'median_abs_error': median_absolute_error(y_true=df_test['y'], y_pred=model.predict(df_test)['yhat'])
                }
            )        
        run_id = mlflow.active_run().info.run_id
    
    # register the model 
    model_uri = "runs:/{run_id}/model".format(run_id=run_id)
    model_name = f"sales-forecaster-store-{store_id}"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    return True

