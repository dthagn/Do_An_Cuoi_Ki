from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SHAP_PLOT_PATH = BASE_DIR / "images" / "shap_summary.png"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import cloudpickle

        with MODEL_PATH.open("rb") as f:
            return cloudpickle.load(f)
    except ImportError:
        return joblib.load(MODEL_PATH)


def make_input_df(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
    name: str,
    ticket: str,
    cabin: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pclass": int(pclass),
                "Sex": sex,
                "Age": float(age) if not np.isnan(age) else np.nan,
                "SibSp": int(sibsp),
                "Parch": int(parch),
                "Fare": float(fare) if not np.isnan(fare) else np.nan,
                "Embarked": embarked,
                "Name": name,
                "Ticket": ticket,
                "Cabin": cabin if cabin.strip() else np.nan,
            }
        ]
    )


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
)

model = load_model()

st.title("Titanic Survival Prediction")
st.caption(
    "Demo triển khai — mô hình tốt nhất theo F1 (Logistic Regression) · "
    "SHAP toàn cục từ Random Forest"
)
with st.expander("Cách dùng trong buổi trình bày", expanded=False):
    st.markdown(
        """
1. Chọn thông tin hành khách ở sidebar → **Predict**.
2. Xem **Survival probability** và nhãn dự đoán.
3. Cột phải: **SHAP summary** — feature nào ảnh hưởng mạnh tới sống/sót (train trong notebook).

*Chạy notebook trước nếu thiếu `model.pkl` hoặc `images/shap_summary.png`.*
        """
    )

with st.sidebar:
    st.header("Passenger info")

    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=1.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

    name = st.text_input("Name (for Title)", value="Doe, Mr. John")
    ticket = st.text_input("Ticket (for Group_Size)", value="A/5 21171")
    cabin = st.text_input("Cabin (for Deck)", value="")

    predict_btn = st.button("Predict", type="primary")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")

    if model is None:
        st.error("Missing `model.pkl`. Run `modeling.ipynb` to train and export the model.")
    elif predict_btn:
        x_in = make_input_df(pclass, sex, age, sibsp, parch, fare, embarked, name, ticket, cabin)
        proba = float(model.predict_proba(x_in)[:, 1][0])
        pred = int(proba >= 0.5)

        st.metric("Survival probability", f"{proba:.1%}")
        label = "Survived" if pred == 1 else "Did not survive"
        if pred == 1:
            st.success(f"Predicted: **{label}**")
        else:
            st.error(f"Predicted: **{label}**")

with col2:
    st.subheader("SHAP (global)")

    if SHAP_PLOT_PATH.exists():
        st.image(str(SHAP_PLOT_PATH), caption="SHAP summary — Random Forest")
    else:
        st.info("Run `modeling.ipynb` to generate `images/shap_summary.png`.")
