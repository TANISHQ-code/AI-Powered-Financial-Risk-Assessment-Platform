import sys
from pathlib import Path

# Add project root to Python path for Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model_service import predict_risk


app = FastAPI(title="FinRisk AI")


class RiskInput(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str


@app.get("/")
def home():
    return {
        "service": "FinRisk AI",
        "status": "running"
    }


@app.post("/predict-risk")
def predict(payload: RiskInput):
    try:
        result = predict_risk(payload.dict())

        return {
            "risk_score": result["risk_score"],
            "risk_category": result["risk_category"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )