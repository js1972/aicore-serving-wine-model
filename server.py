# server.py
import os
import mlflow.pyfunc as pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Where we baked the MLflow model into the image
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/ml/model")

# Load MLflow pyfunc model once at startup
model = pyfunc.load_model(MODEL_PATH, env_manager="local")

app = FastAPI()


class DataframeSplit(BaseModel):
    columns: list[str]
    data: list[list[float]]


class Payload(BaseModel):
    dataframe_split: DataframeSplit


@app.post("/invocations")
def predict(payload: Payload):
    df = pd.DataFrame(payload.dataframe_split.data,
                      columns=payload.dataframe_split.columns)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}

