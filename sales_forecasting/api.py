from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import pandas as pd

app = FastAPI()

class ForecastResponse(BaseModel):
    date: str
    forecast: float

@app.get("/store/{store_id}/predictions", response_model=List[ForecastResponse])
def forecast_sales(
    store_id: int,
    dates: str = Query(..., description="Comma-separated list of dates (YYYY-MM-DD)")
):
    date_list = [d.strip() for d in dates.split(",") if d.strip()]
    if not date_list:
        raise HTTPException(status_code=400, detail="No dates provided")
    
    import mlflow
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    model_uri = f"models:/sales-forecaster-store-{store_id}/latest"
    try:
        import mlflow.pyfunc
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    # Prepare input for the model (assuming model expects a DataFrame with 'date' column)
    input_df = pd.DataFrame({"ds": date_list})
    try:
        preds = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    results = []
    for i, row in pd.DataFrame(preds).iterrows():
        results.append(ForecastResponse(date=pd.to_datetime(row["ds"]).strftime("%Y-%m-%d"), forecast=row["yhat"]))
    return results