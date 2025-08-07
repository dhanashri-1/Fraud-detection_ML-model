
# 💳 Credit Card Fraud Detection

A machine learning model to detect fraudulent transactions using a real-world imbalanced dataset.

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent Cases**: 492

## 🛠️ Tech Stack

- Python
- Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
- Random Forest Classifier
- Streamlit (for deployment)

## 📊 Model Evaluation

- Precision, Recall, F1 Score
- ROC-AUC Score
- Confusion Matrix

## 🧪 How to Run

1. Clone this repo
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run model:
   ```bash
   python fraud_detection.py

https://github.com/user-attachments/assets/5e837723-51ac-4672-bdb2-32f126dce2fe


   ```
4. Run Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🚀 Features

- Handles class imbalance using SMOTE
- Deployable as a web app using Streamlit
- Clean and interpretable metrics

## 📦 Folder Structure

```
fraud-detection/
├── creditcard.csv
├── fraud_detection.py
├── streamlit_app.py
├── fraud_model.pkl
├── requirements.txt
└── README.md
```

## 👩‍💻 Author

Dhanashri Rode
