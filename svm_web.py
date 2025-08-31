import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Streamlit Page Setup
st.set_page_config(page_title="SVM Loan Approval Classifier", layout="centered")

st.title("📊 Loan Approval Prediction using SVM")

# Load dataset
st.subheader("📂 Dataset Information")
df = pd.read_csv("loan_data.csv")
st.write("Raw Dataset Preview:", df.head())

# Show row/column info
st.write("**Dataset Shape:**", df.shape)
st.write("**Duplicate Rows:**", df.duplicated().sum())

# Clean dataset
df = df.drop_duplicates()
df = df.dropna()

st.write("**After Cleaning - Shape:**", df.shape)
st.write("**Missing Values per Column:**")
st.write(df.isnull().sum())

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Train-test split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm_model = LinearSVC(max_iter=1000)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluation metrics
st.subheader("📈 Model Performance")
st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 2))

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
st.write("**Classification Report:**")
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", matrix)

# Confusion Matrix
st.subheader("🔎 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SVM")
st.pyplot(fig)

###############################################################################################


model = svm_model

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the details below to predict loan approval status.")

st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100)
    person_gender = st.selectbox("Gender", ["", "Male", "Female"])
    person_education = st.selectbox(
        "Education",
        ["", "High school", "Bachelor", "Master", "Associate", "Doctorate"]
    )
with col2:
    person_income = st.number_input("Annual Income", min_value=0, max_value=500000)
    person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["", "Rent", "Own", "Mortgage", "Other"]
    )


st.subheader("💰 Loan Information")
col3, col4 = st.columns(2)
with col3:
    loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=100000)
    loan_intent = st.selectbox(
        "Loan Intent",
        ["", "Education", "Medical", "Venture", "Personal", "Debtconsolidation", "Homeimprovement"]
    )
with col4:
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=40.0, step=0.1)

if person_income > 0 and loan_amnt > 0:
    loan_percent_income = loan_amnt / person_income
    st.info(f"📊 Loan Percent Income: *{loan_percent_income:.2f}*")
else:
    loan_percent_income = 0


st.subheader("📊 Credit Information")
col5, col6 = st.columns(2)
with col5:
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850) 
with col6:
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["", "Yes", "No"])


if st.button("🔍 Predict Loan Approval"):
    if (person_age and person_income and loan_amnt and credit_score
        and person_gender and person_education and person_home_ownership
        and loan_intent and previous_loan_defaults_on_file):

        input_data = pd.DataFrame([{
            "person_age": person_age,
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file
        }])

        # Save feature columns from training data
        feature_columns = X.columns

        # Apply same encoding as training
        input_data = pd.get_dummies(input_data, drop_first=True)

        # Align columns with training features
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("✅ This loan status predict to be ! APPROVED !")
        else:
            st.error("❌ This loan status predict to be ! REJECTED !")
    else:
        st.warning("⚠ Please fill in all required fields before predicting.")