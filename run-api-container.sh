#!/bin/sh
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 sales-forecasting fastapi run sales_forecasting/api.py