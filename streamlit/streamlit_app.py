import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import joblib
from clean_data_pipeline import clean_data

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model("/nfs/stak/users/aamirj/guille/capstone/best_models/best_xgboost_model.json")
    return model

@st.cache_data
def get_training_feature_columns(file_path="/nfs/stak/users/aamirj/guille/capstone/best_models/encoded.csv"):
    df = pd.read_csv(file_path)
    drop_cols = ['person_id', 'order_datetime', 'repeat_purchase_amount', 'billing_city', 'shipping_city',
                 'order_day', 'order_month', 'order_hour', 'order_minute',
                 'billing_city_frequency', 'shipping_city_frequency', 'repeat_customer']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    unwanted_skus = [
        "CK2-1000_count", "EVT-1000_count", "SS0-1000_count", "KB1-1000_count", "KS0-1000_count", "nan_count",
        "SH10-1000_count", "SH8-1000_count", "SH6-1000_count", "SH4-1000_count", "SHN6-1000_count",
        "SH10E-1000_count", "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count", "SH4E-1000_count",
        "DS-M_count", "DS-L_count", "EB5-1000_count", "FFS-1000_count", "DH-1000_count"]
    df.drop(columns=unwanted_skus, inplace=True, errors='ignore')
    df.drop(columns=[col for col in df.columns if col.startswith("next_") or col.startswith("most_frequent_product_")], inplace=True, errors='ignore')

    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)
    df.fillna(-1, inplace=True)
    df.drop(columns=df.select_dtypes(include=["object"]).columns, inplace=True)
    df = df.astype(np.float32)
    return df.columns.tolist()

@st.cache_resource
def load_sku_models(models_folder="/nfs/stak/users/aamirj/guille/capstone/best_models/randomforest_models"):
    models = {}
    for file in os.listdir(models_folder):
        if file.endswith(".joblib") and file.startswith("random_forest_"):
            sku = file.replace("random_forest_", "").replace(".joblib", "")
            models[sku] = joblib.load(os.path.join(models_folder, file))
    return models

def preprocess_for_sku_models(df):
    drop_cols = [
        'person_id', 'order_datetime', 'repeat_purchase_amount', 'billing_city', 'shipping_city',
        'order_day', 'order_month', 'order_hour', 'order_minute',
        'billing_city_frequency', 'shipping_city_frequency', 'repeat_customer']
    unwanted_skus = [
        "CK2-1000_count", "EVT-1000_count", "SS0-1000_count", "KB1-1000_count", "KS0-1000_count", "nan_count",
        "SH10-1000_count", "SH8-1000_count", "SH6-1000_count", "SH4-1000_count", "SHN6-1000_count",
        "SH10E-1000_count", "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count", "SH4E-1000_count",
        "DS-M_count", "DS-L_count", "EB5-1000_count", "FFS-1000_count", "DH-1000_count"]

    df.drop(columns=drop_cols + unwanted_skus, inplace=True, errors='ignore')
    df.drop(columns=[col for col in df.columns if col.startswith("next_") or col.startswith("most_frequent_product_")], inplace=True, errors='ignore')

    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)
    df.fillna(-1, inplace=True)
    df.drop(columns=df.select_dtypes(include=["object"]).columns, inplace=True)
    df = df.astype(np.float32)
    return df

model = load_model()
sku_models = load_sku_models()
optimal_threshold = 0.3773
model_features = model.get_booster().feature_names
if model_features is None:
    st.error("Model feature names could not be retrieved.")
    st.stop()
model_features = list(model_features)

display_cols = ["billing_first_name", "billing_last_name", "billing_email"]

st.title("Repeat Customer & Product Prediction")
st.write("Upload a CSV file with customer data to predict repeat purchases and likely next products.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    st.toast("File uploaded! Processing now...")
    with st.spinner("Cleaning and processing your data..."):
        try:
            raw_df = pd.read_csv(uploaded_file)
            raw_df.dropna(how='all', inplace=True)

            if raw_df.empty:
                st.error("Uploaded file is empty. Please check the contents.")
            else:
                df = clean_data(raw_df)
                df.columns = df.columns.tolist()
                display_df = df[[col for col in display_cols if col in df.columns]].copy()

                input_df = df[[col for col in model_features if col in df.columns]].copy()
                for col in input_df.select_dtypes(include=["object"]).columns:
                    input_df[col] = input_df[col].astype("category")
                input_df = input_df[model_features]

                probabilities = model.predict_proba(input_df)[:, 1]
                predictions = ["Yes" if prob >= optimal_threshold else "No" for prob in probabilities]

                results_df = display_df.copy()
                results_df["Prediction"] = predictions
                results_df["Probability"] = probabilities.round(4)

                st.write("### Repeat Purchase Prediction Results")
                st.dataframe(results_df)
                st.success("Prediction complete!")

                returning_customers_idx = [i for i, pred in enumerate(predictions) if pred == "Yes"]
                if returning_customers_idx:
                    df_returning = df.iloc[returning_customers_idx].copy()
                    sku_input_df = preprocess_for_sku_models(df_returning)
                    training_columns = get_training_feature_columns()
                    sku_input_df = sku_input_df[sku_input_df.columns.intersection(training_columns)]
                    for col in set(training_columns) - set(sku_input_df.columns):
                        sku_input_df[col] = -1
                    sku_input_df = sku_input_df[training_columns]

                    returning_display_df = df.iloc[returning_customers_idx][[col for col in display_cols if col in df.columns]].copy()

                    product_predictions = []
                    for _, row in sku_input_df.iterrows():
                        sku_probs = {}
                        row_array = row.to_numpy().reshape(1, -1)
                        for sku, sku_model in sku_models.items():
                            try:
                                prob = sku_model.predict_proba(row_array)[0][1]
                            except Exception:
                                prob = 0.0
                            sku_probs[sku] = prob
                        top_skus = sorted(sku_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                        product_predictions.append(top_skus)

                    returning_display_df["Top Predicted Products"] = [
                        ", ".join([f"{sku} ({prob:.2f})" for sku, prob in preds]) for preds in product_predictions
                    ]

                    st.write("### Returning Customers & Predicted Products")
                    st.dataframe(returning_display_df)
                else:
                    st.info("No customers predicted as repeat buyers. Product predictions skipped.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.write("---")
st.write("Developed for **Capstone Project**")
