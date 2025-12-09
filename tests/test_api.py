import pandas as pd
import requests

test_df = pd.read_csv("data/hackathon_income_test.csv", decimal=',', sep=';')

first_row = test_df.iloc[0].where(
    pd.notnull(test_df.iloc[0]), None
).to_dict()

response = requests.post("http://localhost:8000/predict", json=first_row, proxies={"http": None, "https": None})
res = response.json()

print("Prediction:", res["prediction"])
print("SHAP values:", res["shap_values"])
print("Feature names:", res["feature_names"])