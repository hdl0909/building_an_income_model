from fastapi import FastAPI
from app.predict import load_artifacts, make_prediction
from app.in_out import InputData, OutputData
from model.custom_transformers import DropHighNaN, SplitObjectColumns, FillNumericMedian, KeepSelectedFeatures

import sys
sys.modules['__main__'].DropHighNaN = DropHighNaN
sys.modules['__main__'].SplitObjectColumns = SplitObjectColumns
sys.modules['__main__'].FillNumericMedian = FillNumericMedian
sys.modules['__main__'].KeepSelectedFeatures = KeepSelectedFeatures

app = FastAPI(title="Income Prediction API")

preprocess, model, metadata = load_artifacts()

@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    df = data.to_dataframe()
    pred = make_prediction(df, preprocess, model)
    return OutputData(prediction=float(pred[0]))
