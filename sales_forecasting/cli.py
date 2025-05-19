import click
import pandas
import sys
import logging
import logging.config
import os
from datetime import datetime
import json
import mlflow
from sales_forecasting import train

if os.path.exists('logging.conf'):
    logging.config.fileConfig('logging.conf')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@click.command()
@click.argument('data_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False))
def download_latest_sales_data(data_dir: str):
    logger = logging.getLogger(__name__)
    logger.info("Downloading latest sales data")
    sales_data = train.download_latest_sales_data()
    sales_data.to_csv(os.path.join(data_dir, "latest_sales_data.csv"), index=False)

@click.command()
@click.argument('data_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('store_id', type=click.INT)
@click.option('--mlflow-tracking-uri', default=None)
def train_store_model(data_dir: str, store_id: int, mlflow_tracking_uri: str):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger = logging.getLogger(__name__)
    sales_data = pandas.read_csv(os.path.join(data_dir, "latest_sales_data.csv"))
    train.train_and_register(sales_data, store_id)