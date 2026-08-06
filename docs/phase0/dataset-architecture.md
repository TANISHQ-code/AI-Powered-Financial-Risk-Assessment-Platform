# Phase 0: Dataset Architecture

## Source dataset
The project uses the Home Credit Default Risk dataset, which contains multiple relational tables related to customer applications, bureau history, previous applications, payments, and financial balances.

## Core files
- application_train.csv: labeled training data with target column for default
- application_test.csv: unlabeled test data for scoring
- bureau.csv: information from other credit institutions
- bureau_balance.csv: monthly status history for bureau loans
- previous_application.csv: prior loan applications for the same customer
- installments_payments.csv: repayment schedules and actual payment behavior
- POS_CASH_balance.csv: point-of-sale and cash loan balance history
- credit_card_balance.csv: monthly credit card balance information

## Why these files matter
The business problem is not solved by one table alone. A strong risk model needs both application-level signals and behavioral history from other financial institutions and prior transactions.

## Recommended use order
1. Start with application_train.csv and application_test.csv for the main modeling target.
2. Use bureau.csv and bureau_balance.csv to add credit history context.
3. Use previous_application.csv to capture prior loan behavior.
4. Use installments_payments.csv and credit card/POS data to model repayment behavior.

## Architectural role of the data layer
This layer will be responsible for:
- loading raw CSVs
- cleaning and validating them
- creating engineered features
- storing prepared datasets for modeling and reporting
