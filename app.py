import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set page title and layout
st.set_page_config(page_title="Fraud Detection Model", layout="wide")

# Load dataset directly
DATA_PATH = "/mnt/data/cdd.csv"
df = pd.read_csv(DATA_PATH)

# Display dataset
st.write("### Dataset Preview", df.head())

# Determine target column
target_column = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
st.write(f"Assuming target variable is '{target_column}'.")

# Split data
X = df.drop([target_column], axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Show metrics
st.write("### Classification Metrics", report_df)

# Plot metrics
plot_df = report_df.drop(columns=['support'], errors='ignore')
plot_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar')
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
