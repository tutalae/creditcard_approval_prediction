import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the trained Gradient Boosting model
def load_model():
    # Replace 'path/to/your/model.pkl' with the actual path to your pickle file
    model_path = 'best_model/model.pkl'
    loaded_model = joblib.load(model_path)
    return loaded_model

def main():
    st.title("Loan Prediction App")
    st.write("Enter the required details to get your loan prediction.")

    # Load the model
    model = load_model()

    # Get user input for prediction
    code_gender = st.radio("Gender", ["Male", "Female"])
    flag_own_car = st.radio("Own Car", ["Yes", "No"])
    flag_own_realty = st.radio("Own Realty", ["Yes", "No"])
    cnt_children = st.slider("Number of Children", min_value=0, max_value=10, step=1)
    amt_income_total = st.slider("Income Total", min_value=0, max_value=100000, step=1000)
    name_income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    name_education_type = st.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    name_family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
    name_housing_type = st.selectbox("Housing Type", ["House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Rented apartment", "Office apartment"])
    days_birth = st.slider("Days since Birth", min_value=-30000, max_value=0, step=1)
    days_employed = st.slider("Days Employed", min_value=-20000, max_value=0, step=1)
    flag_mobil = st.selectbox("Flag Mobil", [0, 1])
    occupation_type = st.selectbox("Occupation Type", ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff", "Private service staff", "Uncategorized"])
    cnt_fam_members = st.slider("Number of Family Members", min_value=1, max_value=10, step=1)
    begin_month = st.slider("Begin Month", min_value=-60, max_value=0, step=1)

    # Make a prediction
    user_data = pd.DataFrame({
        "CODE_GENDER": [code_gender],
        "FLAG_OWN_CAR": [flag_own_car],
        "FLAG_OWN_REALTY": [flag_own_realty],
        "CNT_CHILDREN": [cnt_children],
        "AMT_INCOME_TOTAL": [amt_income_total],
        "NAME_INCOME_TYPE": [name_income_type],
        "NAME_EDUCATION_TYPE": [name_education_type],
        "NAME_FAMILY_STATUS": [name_family_status],
        "NAME_HOUSING_TYPE": [name_housing_type],
        "DAYS_BIRTH": [days_birth],
        "DAYS_EMPLOYED": [days_employed],
        "FLAG_MOBIL": [flag_mobil],
        "OCCUPATION_TYPE": [occupation_type],
        "CNT_FAM_MEMBERS": [cnt_fam_members],
        "begin_month": [begin_month]
    })

    # Map categorical columns to numerical values using LabelEncoder
    le = LabelEncoder()
    for x in user_data:
        if user_data[x].dtypes == 'object':
            user_data[x] = le.fit_transform(user_data[x])

    # Make prediction
    prediction = model.predict(user_data)

    # Show the prediction
    if st.button("Predict"):
        st.write(f"Loan Status: {'Approved' if prediction[0] == 1 else 'Not Approved'}")

if __name__ == "__main__":
    main()
