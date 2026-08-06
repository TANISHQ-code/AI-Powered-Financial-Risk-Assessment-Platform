# Phase 1: EDA and Data Quality Plan

## Goals
- Understand the Home Credit dataset at the application and bureau level.
- Identify missing values, duplicates, and feature distributions.
- Explore business questions relevant to banking risk.

## Steps
1. Load `application_train.csv` and inspect schema.
2. Check missing value ratios and duplicate records.
3. Explore numerical distributions for income, credit amount, annuity, and age.
4. Analyze categorical risk factors: income type, education, housing type.
5. Create business questions:
   - Does income level affect default probability?
   - Does employment stability affect risk?
   - Does loan size versus income affect default?
   - Does prior credit history matter?
   - Which customer segments have higher risk?

## Output
- A Jupyter notebook with summary tables and charts.
- A cleaned EDA report for business stakeholders.
