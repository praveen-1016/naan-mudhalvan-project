import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set Streamlit page config
st.set_page_config(page_title="Fraud Detection Model", layout="wide")

# Load local CSV file (make sure 'cdd.csv' is in the same folder)
try:
    df = pd.read_csv("cdd.csv")
except FileNotFoundError:
    st.error("File 'cdd.csv' not found. Make sure it's in the same folder as this app.")
    st.stop()

# Show dataset preview
st.write("### Dataset Preview", df.head())

# Detect target column
target_column = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
st.write(f"Assuming target variable is '{target_column}'.")

# Prepare data
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Show classification report
st.write("### Classification Report", report_df)

# Plot metrics
metrics_df = report_df.drop(columns=['support'], errors='ignore')
metrics_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar')
plt.title('Classification Report Metrics')
plt.ylabel('Score')
plt.xlabel('Class')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc='lower right')
st.pyplot(plt)

# Save model
if st.button("Save Model"):
    joblib.dump(model, 'fraud_detector.pkl')
    st.success("Model saved as 'fraud_detector.pkl'")
