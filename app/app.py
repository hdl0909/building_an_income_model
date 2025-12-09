import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Income Prediction", layout="centered")

st.title("Income Prediction App")
st.write("Выберите пример из тестовой выборки")

@st.cache_data
def load_test_data():
    df = pd.read_csv('data/hackathon_income_test.csv', decimal=',', sep=';')
    return df

test_df = load_test_data()

def sample_form(df):
    st.subheader("Выбор строки из тестовой выборки")

    idx = st.number_input("Индекс строки", 0, len(df)-1, 0)
    st.dataframe(df.loc[[idx]])

    return df.loc[idx]

if test_df is None:
    st.stop()

row = sample_form(test_df)
input_data = row.where(pd.notnull(row), None).to_dict()

if st.button("Предсказать доход"):
    try:
        resp = requests.post(API_URL, data=json.dumps(input_data))
        result = resp.json()
        prediction = result["prediction"]
        st.success(f"Предсказанный доход: {prediction:,.2f}")

        if "shap_values" in result and "feature_names" in result:
            shap_values = result["shap_values"]
            feature_names = result["feature_names"]

            st.subheader("SHAP значения признака")

            shap_df = pd.DataFrame([shap_values], columns=feature_names)

            st.write("Вклад признаков в предсказание:")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap_df.T.plot.barh(ax=ax)
            st.pyplot(fig)

    except Exception as e:
        print(f"Ошибка {e}")