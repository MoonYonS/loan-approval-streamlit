import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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



fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SVM")
st.pyplot(fig)
