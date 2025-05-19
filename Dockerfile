FROM python:3.13-slim

WORKDIR /app

COPY sales_forecasting /app/sales_forecasting
COPY pyproject.toml /app

RUN pip install .