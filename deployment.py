import streamlit as st
import pandas as pd
import pickle

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    with open("artifacts/fab659779c9d4e10a83e6a8c34635771/artifacts/XGBoost_model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def main():
    st.title("Loan Prediction App")
    st.write("Enter the required details to get your loan prediction.")

    # Load the model
    model = load_model()

    # Get user input for prediction
    st.sidebar.title("User Input")

    code_gender = st.sidebar.radio("Gender", ["Male", "Female"])
    flag_own_car = st.sidebar.radio("Own Car", ["Yes", "No"])
    flag_own_realty = st.sidebar.radio("Own Realty", ["Yes", "No"])
    cnt_children = st.sidebar.slider("Number of Children", min_value=0, max_value=10, step=1)
    amt_income_total = st.sidebar.slider("Income Total", min_value=0, max_value=100000, step=1000)
    name_income_type = st.sidebar.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    name_education_type = st.sidebar.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    name_family_status = st.sidebar.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
    name_housing_type = st.sidebar.selectbox("Housing Type", ["House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Rented apartment", "Office apartment"])
    days_birth = st.sidebar.slider("Days since Birth", min_value=-30000, max_value=0, step=1)
    days_employed = st.sidebar.slider("Days Employed", min_value=-20000, max_value=0, step=1)
    flag_mobil = st.sidebar.selectbox("Flag Mobil", [0, 1])
    occupation_type = st.sidebar.selectbox("Occupation Type", ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff", "Private service staff", "Uncategorized"])
    cnt_fam_members = st.sidebar.slider("Number of Family Members", min_value=1, max_value=10, step=1)
    begin_month = st.sidebar.slider("Begin Month", min_value=-60, max_value=0, step=1)

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

    prediction = model.predict(user_data)

    # Show the prediction
    if st.button("Predict"):
        st.write(f"Loan Status: {'Approved' if prediction[0] == 1 else 'Not Approved'}")

if __name__ == "__main__":
    main()
