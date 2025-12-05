from fastapi import FastAPI
from api.predict import load_artifacts, make_prediction
from api.in_out import InputData, OutputData

app = FastAPI(title="Income Prediction API")

preprocess, model, metadata = load_artifacts()

@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    df = data.to_dataframe()
    pred = make_prediction(df, preprocess, model, metadata)
    return OutputData(prediction=float(pred[0]))
