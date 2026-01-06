# High-Value Customer Identification Using Data Mining

## Overview
This project focuses on identifying **high-value customers** using data mining techniques, with the goal of **maximizing business profitability rather than prediction accuracy alone**. Instead of treating all classification errors equally, the analysis incorporates a **profit matrix** to reflect the real financial impact of customer targeting decisions.

Classification models including **decision trees, XGBoost, and Neural Networks**, were used to analyze over 30 variables and predict if a client was high value or not.

## Business Problem
Many customer classification models optimize for accuracy, but in practice:
- Some misclassifications are far more costly than others
- Marketing and retention strategies depend on correctly identifying high-value clients

The objective of this project was to determine **which customers should be classified as high-value** in a way that **maximizes expected profit**, not just model performance metrics.

## Data & Features
- Customer demographic and behavioral variables
- Financial and engagement indicators
- Binary target variable indicating high-value status

Feature selection and preprocessing were performed to ensure model interpretability and reliability.

## Modeling Approach
Multiple classification models were evaluated and compared, including:
- Logistic Regression
- Tree-based and ensemble methods

Rather than selecting a model based solely on accuracy or AUC, each model was evaluated using a **custom profit matrix**, allowing business outcomes to drive model selection.

## Profit Matrix & Evaluation
A profit matrix was applied to assign different values to:
- True Positives (correctly identifying high-value customers): +$1200
- False Positives (unnecessary targeting costs): -$600
- False Negatives (missed high-value customers): 0
- True Negatives: 0

This framework revealed that:
- The most accurate model was not always the most profitable
- More flexible models better captured nonlinear customer behavior relevant to profit optimization

## Results
- Identified a modeling approach that **maximized expected profit**
- Demonstrated why logistic regression may not always be optimal in high-value targeting contexts
- Showed the importance of aligning data mining techniques with **business objectives**

## Tools & Technologies
- R
- Data preprocessing and feature engineering
- Classification modeling
- Profit-based model evaluation

## Key Takeaways
- Business-driven evaluation metrics are critical in data mining
- Profit matrices provide more realistic guidance than accuracy alone
- Model choice should reflect organizational goals, not just statistical performance

## How to Explore This Repository
- Model scripts and analysis files are located in the main directory
- Output files summarize model performance under profit-based evaluation

