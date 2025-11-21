from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import pandas as pd
from mlflow import pyfunc

MODEL_PATH = "/opt/ml/model"
model = pyfunc.load_model(MODEL_PATH)

app = FastAPI()

class DataFrameSplit(BaseModel):
    columns: List[str]
    data: List[List[Any]]

class PredictRequest(BaseModel):
    dataframe_split: DataFrameSplit

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        # Turn dataframe_split into pandas DataFrame
        df = pd.DataFrame(data=req.dataframe_split.data,
                          columns=req.dataframe_split.columns)

        # Enforce exact dtypes expected by the model
        # (all doubles except is_red which is float32 per model signature)
        df = df.astype({
            "fixed_acidity": "float64",
            "volatile_acidity": "float64",
            "citric_acid": "float64",
            "residual_sugar": "float64",
            "chlorides": "float64",
            "free_sulfur_dioxide": "float64",
            "total_sulfur_dioxide": "float64",
            "density": "float64",
            "pH": "float64",
            "sulphates": "float64",
            "alcohol": "float64",
            "is_red": "float32",
        })

        preds = model.predict(df)
        # normalise to list for JSON
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

