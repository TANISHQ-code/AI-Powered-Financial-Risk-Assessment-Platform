# Phase 10: Resume and Interview Preparation

## ATS-friendly resume bullets
- Developed FinRisk AI, an explainable financial risk assessment platform using Home Credit customer and bureau data to support bank credit decisions.
- Designed and implemented a production-style ML pipeline with feature engineering, model comparison, and explainability for risk analysts and portfolio managers.
- Built FastAPI prediction APIs and a Streamlit dashboard to deliver risk scores, recommendations, and portfolio analytics.
- Engineered banking-specific features such as debt-to-income ratio, credit utilization, payment reliability, and employment stability.
- Created explainable AI outputs using SHAP and feature importance to support transparent decision-making.

## Interview talking points
### Why this project?
- It simulates a real bank decision-support system rather than just a model.
- It combines credit risk analytics, explainable AI, and production deployment.
- This is directly relevant to roles in risk analytics, corporate banking, and fintech engineering.

### Why these models?
- Logistic regression provides a transparent baseline and interpretable risk score.
- Random Forest adds nonlinear power while still allowing feature importance analysis.
- XGBoost/LightGBM improve predictive accuracy and support large, structured credit data.

### How does it create business value?
- Detects risky applicants before loan approval.
- Lowers default risk and improves review efficiency.
- Provides analysts with explainable reasons, reducing operational risk.
- Helps monitor portfolio risk and manage credit concentration.

### How would a bank use this system?
- Pre-screen applications via `/predict-risk`.
- Review high-risk cases with detailed explanations.
- Monitor risk distribution and identify stressed customer segments.
- Use analytics dashboards for portfolio oversight.

### Challenges solved
- Joined multiple credit data sources to enrich risk signals.
- Engineered business-relevant features for affordability and payment behavior.
- Built explainability to address trust and compliance concerns.
- Designed a production-friendly API and dashboard layer.
