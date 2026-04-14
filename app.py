from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DATA_PATH = Path("Concrete Compressive Strength") / "Concrete_Data.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "concrete_strength_model.joblib"

INPUT_COLUMNS = [
    "Cement(kg in a m^3 mixture)",
    "Blast Furnace Slag(kg in a m^3 mixture)",
    "Fly Ash (kg in a m^3 mixture)",
    "Water (kg in a m^3 mixture)",
    "Superplasticizer (kg in a m^3 mixture)",
    "Coarse Aggregate (kg in a m^3 mixture)",
    "Fine Aggregate(kg in a m^3 mixture)",
    "Age (day)",
]
TARGET_COLUMN = 'Concrete compressive strength(MPa, megapascals) '


@st.cache_resource
def load_or_train_model(model_name: str) -> Pipeline:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    X = data[INPUT_COLUMNS]
    y = data[TARGET_COLUMN]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, INPUT_COLUMNS)],
        remainder="drop",
    )

    if model_name == "SVR":
        regressor = SVR(kernel="rbf", C=50, gamma="scale", epsilon=0.1)
    else:
        regressor = LinearRegression()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


def build_input_form() -> dict[str, float]:
    st.subheader("Concrete Mix Inputs")

    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input(
            "Cement (kg/m^3)", min_value=0.0, value=300.0, step=1.0)
        slag = st.number_input("Blast Furnace Slag (kg/m^3)",
                               min_value=0.0, value=0.0, step=1.0)
        fly_ash = st.number_input(
            "Fly Ash (kg/m^3)", min_value=0.0, value=0.0, step=1.0)
        water = st.number_input(
            "Water (kg/m^3)", min_value=0.0, value=180.0, step=1.0)

    with col2:
        superplasticizer = st.number_input(
            "Superplasticizer (kg/m^3)", min_value=0.0, value=5.0, step=0.1)
        coarse_aggregate = st.number_input(
            "Coarse Aggregate (kg/m^3)", min_value=0.0, value=1000.0, step=1.0)
        fine_aggregate = st.number_input(
            "Fine Aggregate (kg/m^3)", min_value=0.0, value=700.0, step=1.0)
        age = st.number_input("Age (days)", min_value=1.0,
                              value=28.0, step=1.0)

    return {
        INPUT_COLUMNS[0]: cement,
        INPUT_COLUMNS[1]: slag,
        INPUT_COLUMNS[2]: fly_ash,
        INPUT_COLUMNS[3]: water,
        INPUT_COLUMNS[4]: superplasticizer,
        INPUT_COLUMNS[5]: coarse_aggregate,
        INPUT_COLUMNS[6]: fine_aggregate,
        INPUT_COLUMNS[7]: age,
    }


def main() -> None:
    st.set_page_config(page_title="Concrete Strength Predictor",
                       page_icon="🏗️", layout="wide")

    st.title("Concrete Compressive Strength Predictor")
    st.write(
        "Enter the concrete mix design values below to predict compressive strength (MPa)."
    )

    model_choice = st.selectbox("Model", ["Linear Regression", "SVR"], index=1)

    try:
        model = load_or_train_model(
            "SVR" if model_choice == "SVR" else "Linear Regression")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load/train model: {exc}")
        st.stop()

    inputs = build_input_form()

    if st.button("Predict Strength", type="primary"):
        input_df = pd.DataFrame([inputs])
        prediction = float(model.predict(input_df)[0])

        st.success(
            f"Predicted Concrete Compressive Strength: {prediction:.2f} MPa")

        st.caption(
            "Tip: Try changing the age and water content to see how prediction changes.")

    with st.expander("Preview Input Data"):
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)


if __name__ == "__main__":
    main()
