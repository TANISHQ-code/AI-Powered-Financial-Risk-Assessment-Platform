# FinRisk AI: Explainable Financial Risk Assessment and Decision Support Platform

This project is being built as a production-style fintech analytics platform using the Home Credit Default Risk dataset. The goal is to move beyond a simple classification model and create a business-facing credit risk decision support system for banks and lending teams.

## Project Goal
Build a system that helps financial institutions assess the probability of customer loan default while providing explainable reasoning, analytics, and a dashboard for credit analysts, risk managers, and operations teams.

## Business Problem
Banks must decide whether to approve, reject, or review a loan application. A poor decision can lead to financial losses from defaults or missed revenue from rejecting good customers. This platform helps reduce risk while increasing transparency and decision efficiency.

## Target Users
- Credit analysts
- Risk managers
- Banking operations teams
- Business stakeholders

## Phase 0 Deliverables
- Business understanding
- Project architecture
- Dataset architecture
- Phase-by-phase roadmap

## Repository Structure
- data/raw/: original Home Credit CSV files
- data/processed/: cleaned and feature-enhanced datasets
- models/: serialized model artifacts and explainability artifacts
- notebooks/: exploration and modeling notebooks
- src/: reusable Python modules for data processing, training, and serving
- app/: backend and dashboard application code
- docs/: business and technical documentation

## Getting started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare raw data in `data/raw/`.
3. Run the core pipeline and generate model comparison output:
   ```bash
   python run_pipeline.py
   ```
4. Train and save the API model (this creates `models/finrisk_model.joblib`, `training_metrics.json`, and `model_metadata.json`):
   ```bash
   python -c "from src.model_service import train_and_save; print(train_and_save())"
   ```
5. Start the API service:
   ```bash
   uvicorn app.main:app --reload
   ```
6. Open the dashboard in a separate terminal:
   ```bash
   streamlit run app/dashboard.py
   ```

> Note: The Streamlit dashboard requires the FastAPI backend to be running locally at `http://localhost:8000`.
