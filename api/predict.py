from model.custom_transformers import DropHighNaN, SplitObjectColumns, FillNumericMedian, KeepSelectedFeatures
import joblib
import json
import numpy as np

def load_artifacts():
    preprocess = joblib.load("artifacts/preprocess.pkl")
    model = joblib.load("artifacts/model.pkl")
    metadata = json.load(open("artifacts/metadata.json"))
    return preprocess, model, metadata


def make_prediction(df, preprocess, model):
    df_p = preprocess.transform(df)
    preds_log = model.predict(df_p)
    preds = np.expm1(preds_log)
    return preds