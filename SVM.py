import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#Load dataset
df = pd.read_csv("loan_data.csv")

# Show row column num
print("=======Row/Column:", df.shape)

# Show number of duplicate row
print("=======Duplicate rows:", df.duplicated().sum())

# Remove duplicate row
df = df.drop_duplicates()

# Drop empty row
df = df.dropna()
# Show row column num after remove
print("=======After remove(Row/Column):", df.shape)
print("\n=======Missing values per column=======")
print(df.isnull().sum())

print("\n=======First 5 rows=======")
print(df.head())
print("\n=======Last 5 rows=======")
print(df.tail())

# get_dummies converts word into numeric like Gender(male/female) into 0/1, drop_first=True remove one column to avoid redundancy
df = pd.get_dummies(df, drop_first=True)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_model = LinearSVC(max_iter=1000) 
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", matrix)

# Plot Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()