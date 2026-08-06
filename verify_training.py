from pathlib import Path
from src.model_service import train_and_save, predict_risk, explain_risk
from src.pipeline import compare_model_types

root = Path(__file__).resolve().parent
print('root', root)
print('raw train exists', (root / 'data' / 'raw' / 'application_train.csv').exists())

metrics = train_and_save(model_type='xgboost')
print('trained metrics:', metrics)
print('model file exists', (root / 'models' / 'finrisk_model.joblib').exists())
print('metadata exists', (root / 'model_metadata.json').exists())
print('training metrics exists', (root / 'training_metrics.json').exists())

payload = {
    'SK_ID_CURR': 100001,
    'AMT_INCOME_TOTAL': 180000.0,
    'AMT_CREDIT': 250000.0,
    'AMT_ANNUITY': 15000.0,
    'DAYS_BIRTH': -12000,
    'DAYS_EMPLOYED': -2000,
    'DAYS_REGISTRATION': -4000,
    'CNT_FAM_MEMBERS': 2,
    'CNT_CHILDREN': 0,
    'NAME_INCOME_TYPE': 'Working',
    'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
    'NAME_FAMILY_STATUS': 'Single / not married',
    'NAME_HOUSING_TYPE': 'House / apartment',
    'EXT_SOURCE_1': 0.5,
    'EXT_SOURCE_2': 0.3,
    'EXT_SOURCE_3': 0.4,
    'FLAG_OWN_CAR': 'N',
    'FLAG_OWN_REALTY': 'Y',
}
print('prediction:', predict_risk(payload))
print('explanation:', explain_risk(payload)['explanation'][:5])

results = compare_model_types()
print('compare results:', results)
with (root / 'model_comparison.json').open('w', encoding='utf-8') as f:
    import json
    json.dump(results, f, indent=2)
print('saved model_comparison.json', (root / 'model_comparison.json').exists())
