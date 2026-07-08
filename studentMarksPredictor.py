import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚",
    layout="centered"
)

# ----------------------------
# Dataset
# ----------------------------

data = {
    "Study_hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [20,28,35,45,55,62,70,89,90,97]
}

df = pd.DataFrame(data)

# ----------------------------
# Model Training
# ----------------------------
x = df[["Study_hours"]]
y = df["Marks"]

X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

prediction_score = r2_score(
    Y_test,
    model.predict(X_test)
)

# ----------------------------
# UI
# ----------------------------

st.title("📚 Student Marks Predictor")

st.write(
    "Enter the number of study hours to predict expected marks."
)

st.divider()

hours = st.slider(
    "Study Hours",
    min_value=1,
    max_value=15,
    value=5
)

if st.button("Predict Marks"):

    prediction = model.predict(
        pd.DataFrame([[hours]], columns=["Study_hours"])
    )[0]

    st.success(f"Predicted Marks : {prediction:.2f}")

st.divider()

st.subheader("Model Accuracy")

st.metric(
    label="R² Score",
    value=f"{prediction_score:.2f}"
)

st.subheader("Dataset")

st.dataframe(df, use_container_width=True)

st.subheader("Study Hours vs Marks")

st.line_chart(
    df.set_index("Study_hours")
)