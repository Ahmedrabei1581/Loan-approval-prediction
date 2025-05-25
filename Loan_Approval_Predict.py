import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import io
from streamlit_lottie import st_lottie
import requests

# --- Page Config and Styling ---
st.set_page_config(page_title="Loan Approval Prediction App", layout="centered")

# --- Custom CSS for background and styling ---
custom_css = """
<style>
html, body, [class*="css"]  {
    background-color: #001f3f !important;
    color: white !important;
}
section[data-testid="stSidebar"] {
    background-color: #001f3f !important;
}
input, select, textarea {
    color: white !important;
    background-color: #003366 !important;
    border: 1px solid white !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stDataFrame"] {
    color: white !important;
}
.stButton>button {
    background-color: white !important;
    color: #001f3f !important;
    font-weight: bold;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
}
thead th {
    background-color: #001f3f !important;
    color: white !important;
    font-weight: bold;
}
tbody td {
    background-color: #001f3f !important;
    color: white !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Load Lottie Animation from URL ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"⚠️ Failed to load Lottie animation from {url}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading Lottie animation: {e}")
        return None

# URLs for Lottie animations (approved and denied)
lottie_approve = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_yr6zz3wv.json")
lottie_deny = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")

# Paths to model and preprocessing files
model_path = "loan_approval_logistic_model.pkl"
scaler_path = "scaler.pkl"           # fixed whitespace if any
encoder_path = "label_encoders.pkl" # fixed whitespace if any

# Load model, scaler, and encoders safely
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model file '{model_path}': {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading scaler file '{scaler_path}': {e}")
    st.stop()

try:
    label_encoders = joblib.load(encoder_path)
except Exception as e:
    st.error(f"Error loading label encoders file '{encoder_path}': {e}")
    st.stop()

st.title("🏦 Loan Approval Prediction App")

# --- Input Fields ---
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Income (EGP)", min_value=1000.0)

if age < 20 or age > 55:
    st.error("Please input age between 20 to 55 years")
if income < 5000:
    st.error("Please enter amount more than or equal 5000 EGP")

home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=0.5)
loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amount = st.number_input("Loan Amount (EGP)", min_value=1000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.1, format="%.2f")
loan_period = st.selectbox("Loan Period (Months)", [36, 48, 60, 72, 84])

def calculate_monthly_payment(amount, rate, term):
    monthly_rate = rate / 100 / 12
    if monthly_rate == 0:
        return round(amount / term, 2)
    return round(amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1), 2)

monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_period)
st.success(f"Estimated Monthly Payment: {monthly_payment} EGP")

loan_percent_income = round(loan_amount / income, 2) if income > 0 else 0.0
st.info(f"Loan to Income Ratio: {loan_percent_income}")

loan_history = st.selectbox("Defaulted Before? (cb_person_default_on_file)", ['N', 'Y'])
employed_stably = st.selectbox("Employed Stably?", ['0', '1'])
credit_history = st.number_input("Credit History Length (Years)", min_value=0)

# --- Prediction ---
if st.button("Predict Loan Approval"):
    if age < 20 or age > 55 or income < 5000:
        st.stop()

    try:
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
                    st.error(f"Value '{input_data[col]}' not recognized for {col}. Valid: {list(encoder.classes_)}")
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
                "Risk Ratio": [f"🔴 {risk_ratio}"]
            })

            if lottie_approve:
                st_lottie(lottie_approve, height=150)
            st.markdown("## ✅ Loan Approved")
            st.dataframe(result_df.style.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#001f3f'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'tbody td', 'props': [('background-color', '#001f3f'), ('color', 'white')]},
            ]))

            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Loan Results')
                writer.save()
            st.download_button("📥 Download Results as Excel", data=output.getvalue(), file_name="loan_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        else:
            if lottie_deny:
                st_lottie(lottie_deny, height=150)
            st.markdown("## ❌ Loan Denied")
            suggestions = []
            if income < 5000:
                suggestions.append("Increase your income")
            if loan_amount > income * 0.5:
                suggestions.append("Reduce the requested loan amount")
            if credit_history < 3:
                suggestions.append("Improve your credit history")
            if employed_stably == '0':
                suggestions.append("Maintain stable employment")

            st.write("Suggestions to improve approval:")
            for s in suggestions:
                st.markdown(f"- {s}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
