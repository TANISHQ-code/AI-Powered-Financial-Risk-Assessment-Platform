from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.model_service import explain_risk, predict_risk, train_and_save

app = FastAPI(title='FinRisk AI')

class RiskInput(BaseModel):
    SK_ID_CURR: Optional[int] = None
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[int] = 0
    CNT_CHILDREN: Optional[int] = 0
    DAYS_REGISTRATION: Optional[int] = 0
    FLAG_OWN_CAR: Optional[str] = 'N'
    FLAG_OWN_REALTY: Optional[str] = 'N'

@app.get('/')
def home():
    return {'service': 'FinRisk AI', 'status': 'running'}

@app.post('/predict-risk')
def predict_risk_endpoint(payload: RiskInput):
    try:
        result = predict_risk(payload.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        'SK_ID_CURR': payload.SK_ID_CURR,
        'risk_score': result['risk_score'],
        'risk_category': result['risk_category'],
    }

@app.post('/explain-risk')
async def explain_risk_endpoint(payload: RiskInput):
    try:
        result = explain_risk(payload.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        'SK_ID_CURR': payload.SK_ID_CURR,
        'risk_score': result['risk_score'],
        'risk_category': result['risk_category'],
        'explanation': result['explanation'],
    }

@app.post('/train')
def train_endpoint():
    metrics = train_and_save()
    return {'message': 'Model trained and saved', 'metrics': metrics}
