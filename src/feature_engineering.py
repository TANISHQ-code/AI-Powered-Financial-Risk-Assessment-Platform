import pandas as pd
import numpy as np

BASE_FEATURES = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_REGISTRATION',
    'CNT_FAM_MEMBERS',
    'CNT_CHILDREN',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
]

DERIVED_FEATURES = [
    'debt_to_income_ratio',
    'loan_to_income_ratio',
    'credit_income_diff',
    'annuity_to_income_ratio',
    'annuity_income_gap',
    'credit_annuity_ratio',
    'annuity_credit_diff',
    'loan_annuity_income_ratio',
    'age_years',
    'is_retirement_age',
    'employment_stability_score',
    'employment_tenure_years',
    'employment_longer_than_age',
    'employment_age_ratio',
    'registration_age_ratio',
    'income_per_family_member',
    'income_per_child',
    'children_ratio',
    'high_annuity_ratio',
    'large_credit_small_annuity',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-relevant features for credit risk modeling."""
    out = df.copy()

    # Basic income and credit ratios
    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['debt_to_income_ratio'] = out['AMT_CREDIT'] / out['AMT_INCOME_TOTAL'].replace(0, np.nan)
        out['loan_to_income_ratio'] = out['AMT_CREDIT'] / out['AMT_INCOME_TOTAL'].replace(0, np.nan)
        out['credit_income_diff'] = out['AMT_CREDIT'] - out['AMT_INCOME_TOTAL']

    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['annuity_to_income_ratio'] = out['AMT_ANNUITY'] / out['AMT_INCOME_TOTAL'].replace(0, np.nan)
        out['annuity_income_gap'] = out['AMT_INCOME_TOTAL'] - out['AMT_ANNUITY']

    if {'AMT_CREDIT', 'AMT_ANNUITY'}.issubset(out.columns):
        out['credit_annuity_ratio'] = out['AMT_CREDIT'] / out['AMT_ANNUITY'].replace(0, np.nan)
        out['annuity_credit_diff'] = out['AMT_ANNUITY'] - out['AMT_CREDIT']

    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY'}.issubset(out.columns):
        out['loan_annuity_income_ratio'] = out['AMT_CREDIT'] / (out['AMT_INCOME_TOTAL'] + out['AMT_ANNUITY']).replace(0, np.nan)

    # Age and employment features
    if 'DAYS_BIRTH' in out.columns:
        out['age_years'] = (-out['DAYS_BIRTH'] / 365).round(0).astype('Int64')
        out['is_retirement_age'] = (out['age_years'] >= 60).astype('int64')

    if 'DAYS_EMPLOYED' in out.columns:
        out['employment_stability_score'] = np.where(out['DAYS_EMPLOYED'] < 0, 1, 0)
        out['employment_tenure_years'] = (-out['DAYS_EMPLOYED'] / 365).round(1)
        out['employment_longer_than_age'] = np.where(
            out['DAYS_EMPLOYED'] < out['DAYS_BIRTH'], 1, 0
        ) if 'DAYS_BIRTH' in out.columns else 0

    if {'DAYS_EMPLOYED', 'DAYS_BIRTH'}.issubset(out.columns):
        out['employment_age_ratio'] = out['DAYS_EMPLOYED'] / out['DAYS_BIRTH']

    if {'DAYS_REGISTRATION', 'DAYS_BIRTH'}.issubset(out.columns):
        out['registration_age_ratio'] = (-out['DAYS_REGISTRATION'] / out['DAYS_BIRTH']).replace([np.inf, -np.inf], np.nan)

    # Household and family features
    if 'CNT_FAM_MEMBERS' in out.columns and 'AMT_INCOME_TOTAL' in out.columns:
        out['income_per_family_member'] = out['AMT_INCOME_TOTAL'] / out['CNT_FAM_MEMBERS'].replace(0, np.nan)

    if {'CNT_CHILDREN', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['income_per_child'] = out['AMT_INCOME_TOTAL'] / (out['CNT_CHILDREN'].replace(0, np.nan) + 1)

    if {'CNT_CHILDREN', 'CNT_FAM_MEMBERS'}.issubset(out.columns):
        out['children_ratio'] = out['CNT_CHILDREN'] / out['CNT_FAM_MEMBERS'].replace(0, np.nan)

    # Payment capacity flags
    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(out.columns):
        out['high_annuity_ratio'] = (
            out['AMT_ANNUITY'] / out['AMT_INCOME_TOTAL'].replace(0, np.nan) > 0.35
        ).astype('int64')

    if 'AMT_CREDIT' in out.columns and 'AMT_ANNUITY' in out.columns:
        out['large_credit_small_annuity'] = (
            (out['AMT_CREDIT'] > 1_000_000) & (out['AMT_ANNUITY'] < 50_000)
        ).astype('int64')

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def prepare_model_dataset(df: pd.DataFrame, target_col: str = 'TARGET') -> pd.DataFrame:
    engineered = engineer_features(df)
    keep_cols = [col for col in BASE_FEATURES if col in engineered.columns]
    keep_cols += [col for col in DERIVED_FEATURES if col in engineered.columns]
    if target_col in engineered.columns:
        return engineered[[target_col] + keep_cols]
    return engineered[keep_cols]


def get_model_feature_columns() -> list[str]:
    return [col for col in BASE_FEATURES + DERIVED_FEATURES]
