# Titanic Survival Predictor

A simple Machine Learning application that predicts whether a Titanic passenger is likely to survive based on passenger details.

## Project Overview

This project demonstrates a complete basic Machine Learning workflow:

- Data preprocessing
- Feature selection
- Model training
- Model evaluation
- Prediction using a Streamlit interface

## Dataset

The project uses the Titanic dataset containing **891 passenger records** and 12 columns.

Important features used for prediction:

- Passenger Class (`Pclass`)
- Sex
- Age
- Siblings/Spouses (`SibSp`)
- Parents/Children (`Parch`)
- Fare
- Port of Embarkation (`Embarked`)

The target variable is `Survived`.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed `PassengerId`, `Name`, `Ticket`, and `Cabin`
- Missing numerical values were filled using the median
- Missing categorical values were filled using the most frequent value
- Categorical features were encoded using One-Hot Encoding
- Unknown categorical values are handled using `handle_unknown="ignore"`

The dataset was divided into:

- Training set: 80%
- Testing set: 20%

Stratified splitting with `random_state=42` was used.

## Machine Learning Model

The project uses a **Random Forest Classifier** with:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model was trained on the preprocessed training data.

## Model Evaluation

The model was evaluated on the test dataset using accuracy.

**Accuracy: 81.56%**

The test dataset contained **179 samples**.

## Streamlit Application

A simple Streamlit interface was created to allow users to enter passenger details and receive a prediction.

The application accepts:

- Passenger Class
- Sex
- Age
- Siblings/Spouses
- Parents/Children
- Fare
- Port of Embarkation

The application provides:

- Survival prediction
- Prediction confidence

The application displays the ports of embarkation using their full names:

- Southampton
- Cherbourg
- Queenstown

These are internally converted to the corresponding dataset values:

- Southampton → `S`
- Cherbourg → `C`
- Queenstown → `Q`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
Project1/
│
├── Titanic-Dataset.csv
├── titanic.ipynb
├── titanic_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Project Workflow

```text
Titanic Dataset
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Train/Test Split
       ↓
Random Forest Model
       ↓
Model Evaluation
       ↓
Streamlit Prediction App
```
