import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
import altair as alt

# =====================================
# Configuration
# =====================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "model_comparison.json"

# =====================================
# Streamlit Configuration
# =====================================

st.set_page_config(
    page_title="FinRisk AI",
    page_icon="💳",
    layout="wide"
)

# =====================================
# Helper Functions
# =====================================

def call_api(endpoint, payload):
    """Send POST request to FastAPI backend."""

    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=15
    )

    response.raise_for_status()

    return response.json()


def pretty_feature_name(feature):
    """
    Convert raw ML feature names into
    business-friendly financial terms.
    """

    mapping = {

        # ==============================
        # External Credit Scores
        # ==============================

        "EXT_SOURCE_1": "External Credit Score Indicator 1",
        "EXT_SOURCE_2": "External Credit Score Indicator 2",
        "EXT_SOURCE_3": "External Credit Score Indicator 3",

        "Num Ext Source 1": "External Credit Score Indicator 1",
        "Num Ext Source 2": "External Credit Score Indicator 2",
        "Num Ext Source 3": "External Credit Score Indicator 3",

        # ==============================
        # Loan Details
        # ==============================

        "AMT_CREDIT": "Loan Amount",

        "AMT_ANNUITY": "Loan Repayment Amount",

        "Num Credit Annuity Ratio":
            "Loan Payment Burden Ratio",

        # ==============================
        # Customer Information
        # ==============================

        "AMT_INCOME_TOTAL":
            "Annual Income",

        "DAYS_BIRTH":
            "Customer Age",

        "Num Days Birth":
            "Customer Age",

        "DAYS_EMPLOYED":
            "Employment Duration",

        "Num Days Employed":
            "Employment Duration",

        # ==============================
        # Education
        # ==============================

        "Cat Name Education Type Higher Education":
            "Higher Education",

        "Cat Name Education Type Secondary / Secondary Special":
            "Secondary Education",

        # ==============================
        # Income Type
        # ==============================

        "Cat Name Income Type Pensioner":
            "Income Source: Pension",

        "Cat Name Income Type Working":
            "Income Source: Working",

        # ==============================
        # Assets
        # ==============================

        "Cat Flag Own Car N":
            "Does Not Own a Vehicle",

        "Cat Flag Own Car Y":
            "Owns a Vehicle",

        "FLAG_OWN_CAR":
            "Vehicle Ownership",

        # ==============================
        # Family
        # ==============================

        "CNT_CHILDREN":
            "Number of Children",

        "CNT_FAM_MEMBERS":
            "Family Size",

        # ==============================
        # Registration
        # ==============================

        "DAYS_REGISTRATION":
            "Customer Registration Period"

    }

    return mapping.get(
        feature,
        feature.replace("_", " ").title()
    )


# =====================================
# Header
# =====================================

st.title("💳 FinRisk AI")

st.subheader(
    "Enterprise Credit Risk Assessment Platform"
)

st.write(
    """
AI-powered financial risk assessment platform that predicts
customer default probability and explains the key factors
driving each credit decision.

Designed as an end-to-end decision support system for
financial institutions using Explainable AI.
"""
)
# =====================================
# MODEL PERFORMANCE DASHBOARD
# =====================================

st.divider()

st.header("📊 Model Performance Dashboard")

if MODEL_PATH.exists():

    with open(MODEL_PATH, "r") as file:
        model_results = json.load(file)

    metrics_df = pd.DataFrame(model_results).T

    # ==========================
    # BEST MODEL
    # ==========================

    best_model = metrics_df["accuracy"].idxmax()
    best_accuracy = metrics_df.loc[best_model, "accuracy"]

    best_recall = (
        metrics_df.loc[best_model, "recall"]
        if "recall" in metrics_df.columns
        else None
    )

    best_auc = (
        metrics_df.loc[best_model, "roc_auc"]
        if "roc_auc" in metrics_df.columns
        else None
    )

    recall_text = (
        f"{best_recall:.2%}"
        if best_recall is not None
        else "N/A"
    )

    auc_text = (
        f"{best_auc:.2%}"
        if best_auc is not None
        else "N/A"
    )

    st.markdown(
        f"""
<div style="
background:linear-gradient(135deg,#0f172a,#1e293b);
padding:35px;
border-radius:25px;
margin-bottom:30px;
box-shadow:0 10px 25px rgba(0,0,0,.25);
">

<h2 style="text-align:center;color:white;">
🏆 Best Performing Model
</h2>

<h1 style="
text-align:center;
color:#38bdf8;
font-size:44px;
margin-bottom:10px;
">
{best_model}
</h1>

<p style="
text-align:center;
color:#cbd5e1;
">
Selected based on highest validation accuracy.
</p>

<div style="
display:flex;
justify-content:center;
gap:25px;
margin-top:25px;
">

<div style="
background:#111827;
padding:18px;
border-radius:18px;
width:200px;
text-align:center;
">
<h4 style="color:#94a3b8;">Accuracy</h4>
<h2 style="color:white;">{best_accuracy:.2%}</h2>
</div>

<div style="
background:#111827;
padding:18px;
border-radius:18px;
width:200px;
text-align:center;
">
<h4 style="color:#94a3b8;">Recall</h4>
<h2 style="color:white;">{recall_text}</h2>
</div>

<div style="
background:#111827;
padding:18px;
border-radius:18px;
width:200px;
text-align:center;
">
<h4 style="color:#94a3b8;">ROC-AUC</h4>
<h2 style="color:white;">{auc_text}</h2>
</div>

</div>

</div>
""",
        unsafe_allow_html=True
    )

    # ==========================
    # Metrics Table
    # ==========================

    st.subheader("📋 Model Evaluation Metrics")

    format_dict = {
        col: "{:.3f}"
        for col in metrics_df.select_dtypes(include="number").columns
    }

    st.dataframe(
        metrics_df.style.format(format_dict),
        use_container_width=True
    )

    # ==========================
    # Comparison Chart
    # ==========================

    st.subheader("📈 Model Performance Comparison")

    chart_df = (
        metrics_df
        .reset_index()
        .rename(columns={"index": "Model"})
    )

    available_metrics = [
        col
        for col in [
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "f1_score"
        ]
        if col in chart_df.columns
    ]

    chart_data = chart_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "f1_score": "F1 Score"
    }

    chart_data["Metric"] = chart_data["Metric"].map(metric_names)

    comparison_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(
                "Model:N",
                title="Machine Learning Model"
            ),
            y=alt.Y(
                "Score:Q",
                title="Performance Score"
            ),
            color=alt.Color(
                "Metric:N",
                title="Evaluation Metric"
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip(
                    "Score:Q",
                    format=".3f"
                )
            ]
        )
        .properties(height=420)
    )

    st.altair_chart(
        comparison_chart,
        use_container_width=True
    )

else:

    st.warning(
        "⚠️ Model comparison results were not found."
    )
# =====================================
# CUSTOMER RISK EVALUATION
# =====================================

st.divider()

st.header("👤 Customer Risk Evaluation")

st.markdown(
    """
Enter the customer's financial information below to evaluate
their probability of loan default using the trained credit risk model.
"""
)

with st.form("risk_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 💼 Financial Information")

        customer_id = st.number_input(
            "Customer ID",
            min_value=1,
            value=100001,
            step=1
        )

        income = st.number_input(
            "Annual Income (₹)",
            min_value=0.0,
            value=180000.0,
            step=1000.0
        )

        credit = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=0.0,
            value=250000.0,
            step=5000.0
        )

        annuity = st.number_input(
            "Annual Loan Repayment (₹)",
            min_value=0.0,
            value=15000.0,
            step=1000.0
        )

    with col2:

        st.markdown("### 📊 Credit Profile")

        ext1 = st.slider(
            "External Credit Score Indicator 1",
            0.0,
            1.0,
            0.50,
            0.01
        )

        ext2 = st.slider(
            "External Credit Score Indicator 2",
            0.0,
            1.0,
            0.30,
            0.01
        )

        ext3 = st.slider(
            "External Credit Score Indicator 3",
            0.0,
            1.0,
            0.40,
            0.01
        )

        st.info(
            """
Higher external credit scores generally
indicate lower default risk.
"""
        )

    st.divider()

    submit = st.form_submit_button(
        "🔍 Evaluate Customer Risk",
        use_container_width=True
    )
# =====================================
# PREDICTION & EXPLAINABILITY
# =====================================

if submit:

    payload = {

        "SK_ID_CURR": int(customer_id),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,

        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,

        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",

        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,

        "CNT_FAM_MEMBERS": 3,
        "CNT_CHILDREN": 1,
        "DAYS_REGISTRATION": -3000,
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y"
    }

    try:

        prediction = call_api("/predict-risk", payload)
        explanation = call_api("/explain-risk", payload)

        st.divider()

        st.header("📌 Credit Risk Assessment Result")

        risk_score = prediction["risk_score"]
        risk_category = prediction["risk_category"]

        # ----------------------------------
        # Risk Card Colour
        # ----------------------------------

        if risk_score < 0.30:
            bg = "#14532d"
            status = "🟢 Low Risk"

        elif risk_score < 0.60:
            bg = "#854d0e"
            status = "🟡 Medium Risk"

        else:
            bg = "#7f1d1d"
            status = "🔴 High Risk"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
<div style="
background:{bg};
padding:25px;
border-radius:18px;
text-align:center;
">

<h4 style="color:white;">
Default Probability
</h4>

<h1 style="
color:white;
font-size:42px;
">
{risk_score:.2%}
</h1>

</div>
""", unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
<div style="
background:#0f172a;
padding:25px;
border-radius:18px;
text-align:center;
">

<h4 style="color:white;">
Risk Category
</h4>

<h2 style="color:#38bdf8;">
{risk_category}
</h2>

</div>
""", unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
<div style="
background:#0f172a;
padding:25px;
border-radius:18px;
text-align:center;
">

<h4 style="color:white;">
Decision Summary
</h4>

<h2 style="color:white;">
{status}
</h2>

</div>
""", unsafe_allow_html=True)

        st.divider()

        # ===================================
        # CREDIT RISK DRIVERS
        # ===================================

        st.header("🔍 Credit Risk Drivers")

        explanation_df = pd.DataFrame(
            explanation.get("explanation", [])
        )

        if not explanation_df.empty:

            explanation_df["feature"] = (
                explanation_df["feature"]
                .apply(pretty_feature_name)
            )

            explanation_df["importance"] = (
                explanation_df["value"]
                .abs()
            )

            explanation_df = (
                explanation_df
                .sort_values(
                    "importance",
                    ascending=False
                )
            )

            chart = (

                alt.Chart(explanation_df)

                .mark_bar(
                    cornerRadius=6
                )

                .encode(

                    x=alt.X(
                        "importance:Q",
                        title="Contribution to Risk Score"
                    ),

                    y=alt.Y(
                        "feature:N",
                        sort="-x",
                        title="Risk Driver"
                    ),

                    color=alt.condition(
                        alt.datum.value > 0,
                        alt.value("#ef4444"),
                        alt.value("#22c55e")
                    ),

                    tooltip=[

                        alt.Tooltip(
                            "feature:N",
                            title="Risk Driver"
                        ),

                        alt.Tooltip(
                            "value:Q",
                            title="Contribution",
                            format=".3f"
                        )

                    ]

                )

                .properties(
                    height=420
                )

            )

            st.altair_chart(
                chart,
                use_container_width=True
            )

            st.subheader(
                "📋 Key Factors Influencing Risk Assessment"
            )

            display_df = explanation_df[
                [
                    "feature",
                    "value"
                ]
            ].rename(
                columns={
                    "feature": "Risk Driver",
                    "value": "Contribution Score"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.info(
                """
Positive contribution values increase
the predicted credit risk.

Negative contribution values reduce
the predicted credit risk.
"""
            )

        else:

            st.warning(
                "No explainability information available."
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )