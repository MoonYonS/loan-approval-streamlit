import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ===============================
# 1. Load and preprocess dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    df = df.drop_duplicates()
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)  # encode categorical columns
    return df

df = load_data()

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Save column names (so we can align user input to same structure later)
feature_columns = X.columns.tolist()

# ===============================
# 2. Train SVM Model
# ===============================
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    svm_model = LinearSVC(max_iter=1000)
    svm_model.fit(X_train, y_train)
    return svm_model

model = train_model()

# ===============================
# 3. Streamlit UI
# ===============================
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the details below to predict loan approval status.")

st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100)
    person_gender = st.selectbox("Gender", ["Male", "Female"])
    person_education = st.selectbox(
        "Education",
        ["High school", "Bachelor", "Master", "Associate", "Doctorate"]
    )
with col2:
    person_income = st.number_input("Annual Income", min_value=0, max_value=500000)
    person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["Rent", "Own", "Mortgage", "Other"]
    )

st.subheader("💰 Loan Information")
col3, col4 = st.columns(2)
with col3:
    loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=100000)
    loan_intent = st.selectbox(
        "Loan Intent",
        ["Education", "Medical", "Venture", "Personal", "Debtconsolidation", "Homeimprovement"]
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
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

# ===============================
# 4. Prediction
# ===============================
if st.button("🔍 Predict Loan Approval"):

    # Create raw input dict
    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training features
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.success("✅ This loan status is predicted to be APPROVED!")
    else:
        st.error("❌ This loan status is predicted to be REJECTED!")
