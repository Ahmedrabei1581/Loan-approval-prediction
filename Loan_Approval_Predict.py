import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import base64
import plotly.express as px

# Set page config and inject custom CSS
st.set_page_config(page_title="Loan Approval App", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #001f3f;
            color: white;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #001f3f !important;
            color: white !important;
        }
        input, select, textarea {
            color: white !important;
            background-color: #003366 !important;
        }
        .css-1wa3eu0-placeholder, .css-1okebmr-indicatorSeparator {
            color: white !important;
        }
        button[kind="secondary"] {
            background-color: white !important;
            color: red !important;
            font-weight: bold;
        }
        .stAlert {
            background-color: #003366 !important;
            color: white !important;
        }
        thead tr th {
            background-color: #FFD700 !important;
            color: black !important;
        }
        tbody tr td {
            background-color: #fffdd0 !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Paths to model and encoder files
model_path = r"loan_approval_logistic_model.pkl"
scaler_path = r"scaler .pkl"
encoder_path = r"label_encoders .pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoder_path)
except Exception as e:
    st.error(f"Error loading model or preprocessing files: {e}")
    st.stop()

st.title("\U0001F3E6 Loan Approval Prediction App")

# Input Fields
age = st.number_input("Personal Age", min_value=18, max_value=100)
if age < 20 or age > 55:
    st.error("Please input age between 20 to 55 years")

income = st.number_input("Personal Income (EGP)", min_value=1000.0)
if income < 5000:
    st.error("Please enter amount more than or equal 5000 EGP")

home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=0.5)
loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amount = st.number_input("Loan Amount (EGP)", min_value=1000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.1, format="%.2f")
loan_period = st.selectbox("Loan Period (Months)", [36, 48, 60, 72, 84])

# Monthly Payment Calculation
def calculate_monthly_payment(amount, rate, term):
    monthly_rate = rate / 100 / 12
    if monthly_rate == 0:
        return round(amount / term, 2)
    return round(amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1), 2)

monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_period)
st.success(f"Estimated Monthly Payment: {monthly_payment} EGP")

loan_percent_income = round(loan_amount / income, 2) if income > 0 else 0.0
st.info(f"Loan to Income Ratio: {loan_percent_income}")

loan_history = st.selectbox("Defaulted Before?", ['N', 'Y'])
employed_stably = st.selectbox("Employed Stably?", ['NO', 'YES'])
credit_history = st.number_input("Credit History Length (Years)", min_value=0)

# Predict Button
if st.button("Predict Loan Approval"):
    try:
        if age < 20 or age > 55 or income < 5000:
            st.warning("Fix input errors before prediction.")
            st.stop()

        input_data = {
            'person_age': age,
            'person_income': income,
            'person_home_ownership': home_ownership,
            'person_emp_length': emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amount,
            'loan_int_rate': interest_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': loan_history,
            'cb_person_cred_hist_length': credit_history,
            'monthly_payment': monthly_payment,
            'employed_stably': employed_stably
        }

        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'employed_stably']
        numeric_cols = [col for col in input_data if col not in categorical_cols]

        for col in categorical_cols:
            if col in label_encoders:
                encoder = label_encoders[col]
                if input_data[col] in encoder.classes_:
                    input_data[col] = encoder.transform([input_data[col]])[0]
                else:
                    st.error(f"Invalid input '{input_data[col]}' for {col}.")
                    st.stop()

        df_input = pd.DataFrame([input_data])
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

        prediction = model.predict(df_input)[0]

        if prediction == 1:
            credit_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
            credit_risk = credit_map.get(loan_grade, 6)
            risk_ratio = round((credit_risk + loan_amount / income + interest_rate / 100 + loan_percent_income + credit_history / 10) / 5, 2)
            num_checks = loan_period // 3
            check_amount = round(loan_amount * (1 + interest_rate / 100) / num_checks, 2)

            result_df = pd.DataFrame({
                "Loan Amount (EGP)": [loan_amount],
                "Interest Rate (%)": [interest_rate],
                "Loan Total with Interest (EGP)": [round(loan_amount * (1 + interest_rate / 100), 2)],
                "Monthly Payment (EGP)": [monthly_payment],
                "Loan Period (Months)": [loan_period],
                "Number of Checks": [num_checks],
                "Check Amount (EGP)": [check_amount],
                "Risk Ratio": [risk_ratio]
            })

            st.markdown("<h3 style='color:lime;'>\u2705 Loan Approved!</h3>", unsafe_allow_html=True)
            styled_df = result_df.style.applymap(lambda x: 'color: red;' if isinstance(x, float) and x == risk_ratio else 'color: black;')
            st.dataframe(styled_df, use_container_width=True)

            # Export
            export_option = st.radio("Export Results As:", ['None', 'Excel'])
            if export_option == 'Excel':
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Loan Result')
                    writer.save()
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name="loan_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        else:
            st.markdown("<h3 style='color:red;'>\u274C Loan Denied</h3>", unsafe_allow_html=True)
            suggestions = []
            if income < 5000:
                suggestions.append("Increase your income")
            if loan_amount > income * 0.5:
                suggestions.append("Reduce the requested loan amount")
            if credit_history < 3:
                suggestions.append("Improve your credit history")
            if employed_stably == 'NO':
                suggestions.append("Maintain stable employment")

            st.write("Suggestions to improve approval:")
            for s in suggestions:
                st.markdown(f"- {s}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

