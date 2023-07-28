import datetime
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
    gender = st.radio("Gender", ["Male", "Female"])
    own_car = st.radio("Owns a Car", ["Yes", "No"])
    own_realty = st.radio("Owns Realty", ["Yes", "No"])
    num_children = st.slider("Number of Children", min_value=0, max_value=20, step=1)
    income_total = st.slider("Income Total", min_value=45000, max_value=810000, step=1000)
    income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    employment_values = {
                        "Working": 0,
                        "Commercial associate": 1,
                        "Pensioner": 2,
                        "State servant": 3,
                        "Student": 4
                    }
    
    
    education_type = st.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    education_values = {
                        "Secondary / secondary special": 0,
                        "Higher education": 1,
                        "Incomplete higher": 2,
                        "Lower secondary": 3,
                        "Academic degree": 4
                    }
    family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
    marital_status_values = {
                        "Single / not married": 0,
                        "Married": 1,
                        "Civil marriage": 2,
                        "Separated": 3,
                        "Widow": 4
                    }
    housing_type = st.selectbox("Housing Type", ["House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Rented apartment", "Office apartment"])
    housing_values = {
                        "House / apartment": 0,
                        "Municipal apartment": 1,
                        "With parents": 2,
                        "Co-op apartment": 3,
                        "Rented apartment": 4,
                        "Office apartment": 5
                    }
    
    # Calculate 80 years ago from today
    today = datetime.date.today()
    eighty_years_ago = today - datetime.timedelta(days=80*365)

    # Create the date input widgets
    days_birth = st.date_input("Select Birth Date", eighty_years_ago, min_value=eighty_years_ago, max_value=today)
    st.write('Your birthday is:', days_birth)
    
    days_employed = st.date_input("Select Days Employed", eighty_years_ago, min_value=eighty_years_ago, max_value=today)
    st.write('Your employment date is:', days_employed)    

    has_mobile = st.selectbox("Has Mobile", ["No", "Yes"])
    occupation_type = st.selectbox("Occupation Type", ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff", "Private service staff", "Uncategorized"])
    occupations_values = {
                            "Laborers": 0,
                            "Core staff": 1,
                            "Accountants": 2,
                            "Managers": 3,
                            "Drivers": 4,
                            "Sales staff": 5,
                            "Cleaning staff": 6,
                            "Cooking staff": 7,
                            "Private service staff": 8,
                            "Medicine staff": 9,
                            "Security staff": 10,
                            "High skill tech staff": 11,
                            "Waiters/barmen staff": 12,
                            "Low-skill Laborers": 13,
                            "Realty agents": 14,
                            "Secretaries": 15,
                            "IT staff": 16,
                            "HR staff": 17,
                            "Uncategorized": 18  # Assign 18 for "Uncategorized" occupation
                        }
    
    num_family_members = st.slider("Number of Family Members", min_value=1, max_value=20, step=1)
    begin_month = st.slider("Time as Customer(Months)", min_value=0, max_value=60, step=1)
    
    # Make a prediction
    user_data = pd.DataFrame({
        "CODE_GENDER": [gender],  # 1 for Male, 0 for Female
        "FLAG_OWN_CAR": [own_car],  # 1 for Yes, 0 for No
        "FLAG_OWN_REALTY": [own_realty],  # 1 for Yes, 0 for No
        "CNT_CHILDREN": [num_children],
        "AMT_INCOME_TOTAL": [income_total],
        "NAME_INCOME_TYPE": [income_type],
        "NAME_EDUCATION_TYPE": [education_type],
        "NAME_FAMILY_STATUS": [family_status],
        "NAME_HOUSING_TYPE": [housing_type],
        "DAYS_BIRTH": [(today - days_birth).days * -1],  # Negative days to indicate past date
        "DAYS_EMPLOYED": [(today - days_employed).days * -1],  # Negative days to indicate past date
        "FLAG_MOBIL": [has_mobile],  # 1 for Yes, 0 for No
        "OCCUPATION_TYPE": [occupation_type],
        "CNT_FAM_MEMBERS": [num_family_members],
        "begin_month": [begin_month * -1]  # Negative value to indicate past months
    })    

    st.write("User Data before convert:")
    st.dataframe(user_data)
    
    # Map categorical columns to numerical values using LabelEncoder
    user_data_convert = pd.DataFrame({
        "CODE_GENDER": [1 if gender == "Male" else 0],  # 1 for Male, 0 for Female
        "FLAG_OWN_CAR": [1 if own_car == "Yes" else 0],  # 1 for Yes, 0 for No
        "FLAG_OWN_REALTY": [1 if own_realty == "Yes" else 0],  # 1 for Yes, 0 for No
        "CNT_CHILDREN": [num_children],
        "AMT_INCOME_TOTAL": [income_total],
        "NAME_INCOME_TYPE": [employment_values.get(income_type)],
        "NAME_EDUCATION_TYPE": [education_values.get(education_type)],
        "NAME_FAMILY_STATUS": [marital_status_values.get(family_status)],
        "NAME_HOUSING_TYPE": [housing_values.get(housing_type)],
        "DAYS_BIRTH": [(today - days_birth).days * -1],  # Negative days to indicate past date
        "DAYS_EMPLOYED": [(today - days_employed).days * -1],  # Negative days to indicate past date
        "FLAG_MOBIL": [1 if has_mobile == "Yes" else 0],  # 1 for Yes, 0 for No
        "OCCUPATION_TYPE": [occupations_values.get(occupation_type)],
        "CNT_FAM_MEMBERS": [num_family_members],
        "begin_month": [begin_month * -1]  # Negative value to indicate past months
    })
    st.dataframe(user_data_convert)
    
    # Make prediction
    prediction = model.predict(user_data_convert)

    # Prepare the images for "Approved" and "Not Approved" outcomes
    approved_image = "picture/approved_image.jpg"  # Replace with the path to your approved image
    not_approved_image = "picture/not_approved_image.jpg"  # Replace with the path to your not approved image

    # Show the prediction
    if st.button("Predict"):
        loan_status = 'Approved' if prediction[0] == 1 else 'Not Approved'
        st.write(f"Loan Status: {loan_status}")

        # Display the corresponding image based on the prediction
        if loan_status == 'Approved':
            st.image(approved_image, caption="Loan Approved", use_column_width=True)
        else:
            st.image(not_approved_image, caption="Loan Not Approved", use_column_width=True)

if __name__ == "__main__":
    main()
