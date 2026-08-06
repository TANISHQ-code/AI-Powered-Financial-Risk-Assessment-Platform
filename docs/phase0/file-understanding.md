# Home Credit Files and Initial Use Strategy

## Primary modeling table
- application_train.csv
  - Main labeled dataset for training
  - Contains the target variable describing whether the customer defaulted
  - Best starting point for model development

## Scoring table
- application_test.csv
  - Same schema as training data without the target label
  - Used for generating predictions for new customers

## Credit bureau context
- bureau.csv
  - External credit bureau history
  - Useful for understanding prior loans and credit behavior outside Home Credit

- bureau_balance.csv
  - Monthly balance behavior for bureau accounts
  - Helpful for identifying delinquency patterns

## Previous application context
- previous_application.csv
  - Prior loan applications made by the same client
  - Valuable for understanding applicant behavior and application history

## Repayment behavior
- installments_payments.csv
  - Actual repayment schedules and payment behavior
  - Strong signal for repayment discipline

## Credit card and POS behavior
- credit_card_balance.csv
  - Credit card utilization and balance trends
- POS_CASH_balance.csv
  - Point-of-sale and cash loan balance history
  - Useful for liquidity and usage behavior

## Initial modeling decision
The best first version should use:
1. application_train.csv as the base table
2. bureau.csv and bureau_balance.csv as historical context
3. previous_application.csv for prior application behavior
4. installments_payments.csv for repayment reliability

This combination gives a strong and realistic foundation for a first production-style risk model.
