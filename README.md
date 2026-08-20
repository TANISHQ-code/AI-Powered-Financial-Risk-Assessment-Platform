# 🏦 AI-Powered Financial Risk Assessment Platform

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://finrisk-ai-pi.vercel.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)

</p>

> An end-to-end Machine Learning platform that predicts customer credit risk and provides explainable insights to support data-driven lending decisions.

---

## 🌐 Live Demo

🚀 **Interactive Dashboard**

**https://finrisk-ai-pi.vercel.app/**

The live application allows users to:

- Predict customer credit risk in real time.
- View risk probability scores.
- Understand model predictions through SHAP Explainability.
- Explore an interactive dashboard built using Streamlit.

---

## 📌 Project Overview

Financial institutions process thousands of loan applications daily. Manual credit risk assessment is time-consuming, difficult to scale, and prone to inconsistencies.

This project demonstrates an enterprise-style credit risk assessment platform that automates the prediction process while maintaining transparency through Explainable AI.

---

## 💼 Business Value

This platform helps demonstrate how AI can support financial institutions by:

- Automating customer credit risk assessment
- Improving consistency in lending decisions
- Supporting data-driven decision making
- Providing interpretable predictions using SHAP
- Demonstrating a production-style ML workflow from training to deployment

---

## ✨ Features

- 📊 Automated Credit Risk Prediction
- 🧠 Machine Learning Risk Assessment Engine
- 🔍 SHAP Explainability
- ⚡ FastAPI REST API
- 📈 Interactive Streamlit Dashboard
- 📋 Model Comparison
- 📦 Modular Project Structure
- 🚀 Live Cloud Deployment

---

# 🏦 Financial Risk Assessment Workflow

```text
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

# 📂 Repository Structure

```text
AI-Powered-Financial-Risk-Assessment-Platform/
│
├── app/                     # Streamlit Dashboard
├── api/                     # FastAPI Backend
├── data/
│   ├── raw/
│   └── processed/
│
├── deployment/
├── docs/
├── models/
├── notebooks/
├── src/
│
├── model_comparison.json
├── model_metadata.json
├── training_metrics.json
├── requirements.txt
├── run_pipeline.py
├── verify_training.py
├── vercel.json
└── README.md
```

---

# ⚙️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Machine Learning | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Backend | FastAPI |
| Frontend | Streamlit |
| Model Storage | Joblib |
| Deployment | Streamlit Community Cloud |

---

# 📊 Machine Learning Pipeline

1. Load customer credit data
2. Clean and preprocess the dataset
3. Perform feature engineering
4. Split training and testing datasets
5. Train multiple ML models
6. Compare model performance
7. Select the best-performing model
8. Generate explainability using SHAP
9. Save trained artifacts
10. Serve predictions using FastAPI
11. Display insights through Streamlit

---

# 📈 Model Evaluation

| Model | Status |
|---------|--------|
| Logistic Regression | Baseline |
| Random Forest | Candidate |
| XGBoost | Best Performing Model |

*(Detailed metrics are available in `training_metrics.json` and `model_comparison.json`.)*

---

# 🚀 Getting Started

## Clone the repository

```bash
git clone https://github.com/TANISHQ-code/AI-Powered-Financial-Risk-Assessment-Platform.git

cd AI-Powered-Financial-Risk-Assessment-Platform
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the model

```bash
python run_pipeline.py
```

---

## Run the dashboard

```bash
streamlit run app/app.py
```

---

## Run the API

```bash
uvicorn api.main:app --reload
```

---

# 📷 Dashboard Preview

> *(Add screenshots of your Streamlit dashboard here.)*

Example:

```
images/dashboard.png
images/prediction.png
images/shap.png
```

---

# 🔮 Future Improvements

- Cloud Deployment (AWS / Azure / GCP)
- Docker Support
- CI/CD Pipeline
- Automated Model Retraining
- Real-Time Monitoring
- User Authentication
- MLOps Integration
- Credit Portfolio Analytics

---

# 👨‍💻 Author

**Tanishq Mohanty**

GitHub: **https://github.com/TANISHQ-code**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
