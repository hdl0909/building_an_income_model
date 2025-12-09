from fastapi import FastAPI
from app.predict import load_artifacts, make_prediction
from app.in_out import InputData, OutputData
from model.custom_transformers import DropHighNaN, SplitObjectColumns, FillNumericMedian, KeepSelectedFeatures
import shap
import numpy as np

import sys
sys.modules['__main__'].DropHighNaN = DropHighNaN
sys.modules['__main__'].SplitObjectColumns = SplitObjectColumns
sys.modules['__main__'].FillNumericMedian = FillNumericMedian
sys.modules['__main__'].KeepSelectedFeatures = KeepSelectedFeatures

app = FastAPI(title="Income Prediction API")

preprocess, model, metadata = load_artifacts()


explainer = shap.TreeExplainer(model)

@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    df = data.to_dataframe()
    df_p = preprocess.transform(df)

    pred = make_prediction(df, preprocess, model)[0]

    shap_raw = explainer(df_p)
    shap_vals = shap_raw.values[0].tolist()
    feature_names = df_p.columns.tolist()

    return OutputData(
        prediction=float(pred),
        shap_values=shap_vals,
        feature_names=feature_names
    )