# FinRisk AI: Explainable Financial Risk Assessment and Decision Support Platform


An end-to-end machine learning platform that predicts customer credit risk and provides explainable insights to support data-driven lending decisions.

The project demonstrates how financial institutions can leverage predictive analytics to automate credit risk assessment, improve decision consistency, and enhance transparency through explainable AI.

---

## 💼 Business Problem

Financial institutions process thousands of loan applications every day. Manual risk assessment is often time-consuming, inconsistent, and difficult to scale.

This platform addresses these challenges by:

- Predicting the probability of customer default using machine learning.
- Providing interpretable risk scores using SHAP Explainability.
- Delivering predictions through a REST API and interactive dashboard.
- Supporting analysts with data-driven decision making.

---

## 🚀 Key Features

- 📊 Automated Credit Risk Prediction
- 🧠 Machine Learning Risk Assessment Engine
- 🔍 SHAP Explainability for transparent predictions
- 🌐 FastAPI REST API for real-time inference
- 📈 Interactive Streamlit Dashboard
- ⚙️ End-to-End Training & Evaluation Pipeline
- 📋 Model Comparison and Performance Tracking
- 🏗️ Modular Production-Ready Project Structure

---

## 🏦 Financial Risk Assessment Workflow


                  Customer Loan Application
                             │
                             ▼
                  Customer Financial Data
                             │
                             ▼
              Data Validation & Cleaning
                             │
                             ▼
                 Feature Engineering
                             │
                             ▼
          Credit Risk Prediction Models
                             │
                             ▼
               Model Performance Evaluation
                             │
                             ▼
             Best Model Selected for Serving
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
      Risk Probability Score        SHAP Explainability
               │                           │
               └─────────────┬─────────────┘
                             ▼
              FastAPI Prediction Service
                             │
                             ▼
           Streamlit Risk Assessment Dashboard
                             │
                             ▼
       Business Decision Support for Loan Approval
```

---

## 📂 Repository Structure

```text
AI-Powered-Financial-Risk-Assessment-Platform/
│
├── app/                     # Streamlit dashboard
├── api/                     # FastAPI backend
├── data/
│   ├── raw/                 # Original Home Credit dataset (ignored)
│   └── processed/           # Cleaned & feature-engineered data
│
├── deployment/              # Deployment configuration
├── docs/                    # Documentation
├── models/                  # Saved models & explainability artifacts
├── notebooks/               # EDA and experimentation
├── src/                     # Core machine learning pipeline
│
├── run_pipeline.py          # End-to-end pipeline
├── verify_training.py       # Model verification
├── requirements.txt
├── README.md
└── vercel.json
```

---

## ⚙️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Model Serialization | Joblib |
| Deployment | Vercel |

---

## 📊 Machine Learning Pipeline

1. Load Home Credit customer data.
2. Clean missing and inconsistent values.
3. Perform feature engineering.
4. Split data into train and test sets.
5. Train multiple classification models.
6. Compare model performance.
7. Select the best-performing model.
8. Serialize the trained model.
9. Serve predictions through FastAPI.
10. Visualize results using Streamlit.

---

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Status |
|--------|----------|---------|--------|
| Logistic Regression | XX% | XX | Baseline |
| Random Forest | XX% | XX | Candidate |
| XGBoost | XX% | XX | ✅ Best Model |

*(Replace with your actual evaluation metrics.)*

---

## 💼 Business Value

This platform demonstrates how predictive analytics can support financial institutions by:

- Reducing manual effort in credit risk assessment.
- Improving consistency in lending decisions.
- Providing explainable predictions for regulatory transparency.
- Enabling faster decision-making through an interactive dashboard.
- Supporting scalable, data-driven loan approval workflows.

---

## 🚀 Future Enhancements

- Real-time risk monitoring
- Cloud deployment (AWS/Azure/GCP)
- Customer authentication
- Automated model retraining
- CI/CD pipeline
- Docker & Kubernetes deployment
- MLOps integration
- Monitoring and logging
