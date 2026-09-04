import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

sibsp = st.number_input(
    "Siblings/Spouses",
    min_value=0,
    max_value=8,
    value=0,
    step=1
)

parch = st.number_input(
    "Parents/Children",
    min_value=0,
    max_value=6,
    value=0,
    step=1
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=512.3292,
    value=32.0,
    step=0.01
)

# Display proper place names, convert to dataset codes internally
embarked_name = st.selectbox(
    "Port of Embarkation",
    ["Southampton", "Cherbourg", "Queenstown"]
)

embarked_mapping = {
    "Southampton": "S",
    "Cherbourg": "C",
    "Queenstown": "Q"
}

embarked = embarked_mapping[embarked_name]

# Prediction
if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex": [sex],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success("Passenger is predicted to SURVIVE.")
    else:
        st.error("Passenger is predicted NOT to survive.")

    st.write(f"Prediction confidence: {probability:.2%}")