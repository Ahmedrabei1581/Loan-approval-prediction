
# 🏦 Loan Approval Prediction System

This project is an end-to-end machine learning solution for predicting **loan approval status**. It includes data preprocessing, model training, evaluation, and a deployed web app using **Streamlit**. The solution enables financial institutions or loan officers to make faster and data-driven loan approval decisions.

---

## 📘 Project Objectives

- Develop a machine learning model to classify loan applications as **Approved** or **Not Approved**
- Clean and preprocess the input data to make it suitable for modeling
- Deploy the solution in a web app where users can input applicant data and get predictions instantly
- Provide actionable feedback if a loan is not approved

---

## 📂 Project Structure

```
📁 Loan_Approval_Predict/
├── Loan_Approval_Predict.ipynb       # Main Jupyter Notebook
├── loan_approval_logistic_model.pkl  # Trained ML model (logistic regression)
├── label_encoders.pkl                # Encoded categorical feature mappings
├── outliers_handled_data.csv         # Cleaned training dataset
├── streamlit_app.py                  # Streamlit-based prediction app
└── README.md                         # Project documentation
```

---

## 🔍 Step-by-Step Process

### 1. 📥 Data Collection
The dataset contains information about applicants, including:
- Personal details: age, gender, marital status
- Financial attributes: income, loan amount, credit score
- Employment data: employment type, stability
- Family and social indicators: dependents, education, self-employment

> Dataset used: `outliers_handled_data.csv`

---

### 2. 🧹 Data Preprocessing

To ensure the quality and consistency of input data, we performed:

- **Missing Value Handling**: Replaced or removed records with missing values.
- **Outlier Detection & Handling**: Applied methods to detect and cap/fix extreme values.
- **Label Encoding**: Categorical values (e.g., `married`, `loan_type`) were encoded using `LabelEncoder`.
- **Feature Scaling**: Used `StandardScaler` to scale numerical features like income, loan amount, and credit score.

Processed data was saved and reused for inference and deployment.

---

### 3. 🧠 Model Building

A **Logistic Regression** model was trained to predict loan approval based on applicant features. Logistic regression was chosen for its interpretability and performance on binary classification problems.

- **Train-Test Split**: Data was split to evaluate model generalization.
- **Model Training**: The logistic regression model was trained using Scikit-learn.
- **Evaluation Metrics**:
  - Accuracy
  - Precision / Recall
  - Confusion Matrix
  - ROC-AUC Curve

Model performance was validated and saved as `loan_approval_logistic_model.pkl`.

---

### 4. 🧪 Model Inference Logic

The model uses **13 input features**, which include:

| Feature Name         | Description                            |
|----------------------|----------------------------------------|
| `income`             | Applicant's monthly income             |
| `loan_amount`        | Requested loan amount                  |
| `credit_score`       | Credit score (higher is better)        |
| `age`                | Age of the applicant                   |
| `employed_stably`    | Employment stability (YES/NO)          |
| `loan_type`          | Type of loan requested                 |
| `employment_type`    | Type of employment (e.g. salaried)     |
| `married`            | Marital status                         |
| `self_employed`      | Self-employment status                 |
| `dependents`         | Number of dependents                   |
| `education_level`    | Highest education qualification        |
| `residential_status` | Own/rent housing                       |
| `gender`             | Gender                                 |

---

### 5. 🌐 Streamlit Web App

The model is deployed via a **Streamlit web application** for interactive use.

#### Key Features:
- 📋 User input form for 13 required attributes
- ✅ Instant prediction: **Approved** or **Not Approved**
- 💡 If not approved, the app suggests which features to improve
- 📈 If approved, the app calculates:
  - Loan interest (based on base rate)
  - Payment plan
  - Number of checks
  - Risk ratio

To launch the app locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---
https://loan-approval-prediction-abp4c5jgzbjrfclas7fhcv.streamlit.app/
### 6. 📤 Outputs

#### For Approved Loans:
- Detailed loan summary
- Monthly payment estimation
- Total repayment including interest
- Risk analysis (custom ratio)

#### For Denied Loans:
- Smart suggestions to improve approval chances (e.g., increase income, reduce loan amount, improve credit score)

---

## 🧰 Tools & Technologies

- **Python 3.12**
- **Pandas, NumPy** – Data handling
- **Scikit-learn** – Model training and evaluation
- **Matplotlib/Seaborn** – Visualization
- **Pickle** – Model and encoder saving
- **Streamlit** – Web application framework

---

## 📌 Future Improvements

- Add more ML models (Random Forest, XGBoost) for comparison
- Allow user-uploaded CSV files for batch predictions
- Add authentication for loan agents
- Improve feedback system using SHAP or feature importance

---

## 📄 License

This project is licensed under the MIT License.
