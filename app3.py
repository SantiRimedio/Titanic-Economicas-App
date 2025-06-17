import streamlit as st
import joblib
import pandas as pd

st.title("Predictor supervivencia Titanic ML")

# Input fields
pclass = st.selectbox("Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

model = joblib.load(r'rf_model.joblib')

# Expected features (from your training)
FEATURES = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create input data
    data = {
        'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
        'Pclass_1': 1 if pclass == 1 else 0,
        'Pclass_2': 1 if pclass == 2 else 0,
        'Pclass_3': 1 if pclass == 3 else 0,
        'Sex_female': 1 if sex == 'female' else 0,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_C': 1 if embarked == 'C' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0
    }

    df = pd.DataFrame([data])[FEATURES]
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return "Survived" if prediction == 1 else "Did not survive", f"{probability*100:.1f}%"


if st.button("Predict"):
    prediction, probability = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == "Survived":
        st.success(f"✅ Survived (Probability: {probability})")
    else:
        st.error(f"❌ Did not survive (Probability: {probability})")
#%%
