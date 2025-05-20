# 🛡️ AI-Powered Credit Card Fraud Detection System

An AI-based system to detect and prevent credit card fraud using machine learning models like Random Forest, XGBoost, Logistic Regression, and Neural Networks. This solution was built to address challenges such as class imbalance, real-time detection, and evolving fraud patterns using robust feature engineering and ensemble methods.

## 📌 Project Title
**Guarding Transactions with AI-powered Credit Card Fraud Detection and Prevention**

## 👨‍🎓 Student Details
- **Name:** R. Praveen Raj  
- **Institution:** Sri Ramanujar Engineering College  
- **Department:** B.Tech Artificial Intelligence & Data Science    

## 🔗 GitHub Repository
[GitHub Pull Request](https://github.com/praveen-1016/R.Praveen-Raj-project-/pull/1)

## 🧠 Problem Statement
Credit card fraud detection is a binary classification challenge involving the identification of fraudulent (1) or legitimate (0) transactions. The problem is highly imbalanced and requires models that can handle rare-event detection with high precision and recall.

## 🎯 Objectives
- Achieve at least **95% precision** and **85% recall**
- Process transactions in **real-time (< 500ms)**
- Address **class imbalance** using SMOTE
- Provide **interpretable models**
- Ensure **adaptability** to evolving fraud patterns

## 🔁 Workflow
```text
[Data Collection] → [Preprocessing] → [EDA] → [Feature Engineering]
        ↓                ↓                ↓
   [Train-Test Split] → [Model Training] → [Evaluation]
                      ↓
         [Hyperparameter Tuning]
                      ↓
         [Real-time Detection System]
                      ↓
             [Prototype Deployment]
````

## 📦 Dataset Information

* **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Records:** 284,807 transactions
* **Fraudulent Transactions:** 492 (0.172%)
* **Features:** 31 (Time, Amount, V1–V28 via PCA, and Class)
* **Challenge:** Extremely imbalanced binary classification

## 🧹 Data Preprocessing

* ✅ No missing values
* ✅ Retained duplicate transactions for temporal integrity
* ✅ Outliers capped using Isolation Forest & LOF
* ✅ Feature scaling with StandardScaler
* ✅ Used SMOTE for oversampling fraud class
* ✅ Used time-based splitting to simulate real-world scenario

## 📊 Exploratory Data Analysis (EDA)

* Legitimate transactions peak during the day; fraud is more evenly distributed
* Smaller and mid-sized amounts show higher fraud rates
* PCA features V1, V3, V4, V10, V12, V14, V17 are highly correlated with fraud
* Time and amount significantly influence fraud probability

## 🔧 Feature Engineering

* **Time-based:** `Hour_of_day`, `Is_night`, `Day_of_week`
* **Amount-based:** `Amount_bin`, `Is_round_amount`, `Log_amount`
* **Behavioral:** `V1_V3_ratio`, `Abnormality_score`, `V_fluctuation`
* Feature selection via mutual information & RFE

## 🤖 Models Implemented

| Model               | Precision | Recall | F1-Score | AUC-ROC |
| ------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 82.3%     | 77.2%  | 79.7%    | 96.4%   |
| Random Forest       | 93.7%     | 89.0%  | 91.3%    | 98.9%   |
| XGBoost             | 93.0%     | 88.4%  | 90.6%    | 98.8%   |
| Neural Network      | 84.7%     | 78.0%  | 81.2%    | 97.0%   |

✅ **Random Forest** delivered the best overall performance.

## 🔍 Model Insights

* Confusion matrix shows Random Forest minimized false positives and false negatives
* ROC and PR curves show strong model separation, especially for ensemble methods
* Threshold analysis suggests F1-score improves at a 0.3 decision threshold
* Important features: `V17`, `V14`, `V12`, `V10`, `Abnormality_score`, `Hour_of_day`

## 🧪 Tools & Technologies

* **Programming Language:** Python 3.8
* **Development Tools:** Google Colab, Jupyter Notebook, Git/GitHub
* **ML Libraries:** scikit-learn, XGBoost, TensorFlow, Keras, imbalanced-learn
* **Visualization:** Matplotlib, Seaborn, Plotly, Yellowbrick
* **App/Deployment:** Streamlit, Flask
* **Team Management:** Trello, Google Drive, Zoom

## 👥 Team Contributions

**R. Praveen Raj**

* Data preprocessing
* Feature engineering
* EDA & visualizations
* Documentation coordination

**Mugileshwaran**

* Problem definition
* EDA and model comparison
* Evaluation metrics
* Presentation material

**Bharath M.**

* Model implementation
* Feature selection
* Streamlit prototype
* GitHub management

## 🧾 License

This project is open-source and available under the **MIT License**.

## 📫 Contact

For any queries or feedback:

* 📧 Email: praveenrajr207@gmail.com


