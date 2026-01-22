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

Dwelling data:<br/>
1 – State: state code (10=Delaware, 24=Maryland, 42=Pennsylvania)  <br/>
2 – Lot: lot size (1=less than one acre, 2=one to ten acres, 3=more than ten acres)<br/>
3 – Bedroom: number of bedrooms <br/>
4 – Type: units in structure (1=Mobile home or trailer, 2=One-family house detached, 3=One-family house attached, 4=two apartments,5=3-4 apartments, 6=5-9 apartments, 7=10-19 apartments, 8=20-49 apartments, 9=50+ apartments, 10=boat, RV, van, etc.)<br/>
5 – Rooms: #rooms <br/>
6 – Built: year structure was built (1: <=1939, 2: 1940-1949, 3: 1950-1959, 4: 1960-1969, 5=1970-1979, 6=1980-1989, 7=1990-1999, 8=2000-2004, 9=2005, 10=2011…19=2015)<br/>
7 – Internet: access to the internet (1= yes with subscription, 2=yes without subscription, 3=no)<br/>
8 – Water: annual water cost (0=$0, 1=$1..., 9999 = $9999+)<br/>
9 – Zestimate: Zillow estimate of property<br/>

Mortgage data:<br/>
10 – SecMtg: is there a second mortgage on the property (1=yes, 0=no)<br/>
11 – SecMtgStat: second mortgage or home equity status (1=second mortgage, 2=home equity loan, 3=no, 4=both second mortgage and home equity loan)<br/>

Demographic data for head of household: <br/>
12 – FamEmp: family type and employment status (1=Married-couple family: Husband and wife in labor force (LF), 2=Married-couple family: Husband in labor force, wife not in LF, 3= Married-couple family: Husband not in LF, wife in LF, 4 =Married-couple family: Neither husband nor wife in LF, 5=Other family: Male householder, no wife present, in LF, 6=Other family: Male householder, no wife present, not in LF, 7=Other family: Female householder, no husband present, in LF, 8=Other family: Female householder, no husband present, not in LF)<br/>
13 – ChildAge: 1=under five years only, 2= 5-17 years only, 3= under five and 5 to 17 years, 4= no children<br/>
14 – Language: household language (1= English only, 2=Spanish, 3=Other Indo-European languages, 4=Asian and Pacific Island languages, 5= Other languages) <br/>
15 – Family: household and family type (1=Married couple household, 2=Other family household: male head of household (HH), no wife present, 3=Other family household: female HH, no husband present, 4=Nonfamily household: Male HH: Living alone, 5 =Nonfamily household: Male HH: Not living alone, 6 =Nonfamily household: Female HH: Living alone, 7=Nonfamily household: Female HH: Not living alone)<br/>
16 - Grandparent: household with grandparent living with grandchildren (1= yes, 0 = no)<br/>
17 - SSMC: same-sex married couple household (0 =Households without a same-sex married couple, 1 =Same-sex married-couple household where not all relevant data shown as reported, 2 =All other same-sex married-couple households)<br/>
18 - English: limited English-speaking household (1=no, 2 =yes) <br/>
19 - MultiGen: Multigenerational household (1=no, 2 =yes)<br/>
20 - Move: Length of time at current residence (1: <=12 months, 2= 13-23 months, 3=2-4 years, 4=5-9 years, 5=10-19 years, 6=20-29 years, 7=30+ years)
21 - NChildren: Number of children in household<br/>
22 - NPersons: Number of persons in family
23 - Partner: unmarried partner household (0 =No unmarried partner in household, 1 =Male householder, male partner, 2 =Male householder, female partner, 3 =Female householder, female partner, 4 =Female householder, male partner)<br/>
24 - Under18: presence of persons under 18 in household (0=no, 1=yes)<br/>
25 - Over60: presence of persons over 60 in household (0=none, 1=one person 60+, 2=two or more persons 60+)<br/>
26 - Over65: presence of persons over 65 in household (0=none, 1=one person 65+, 2=two or more persons 65+)<br/>
27 - Workers: Workers in family during the past 12 months (0= none, 1, 2, 3=3+ workers in family)<br/>
28 - WkExp: Work experience of head of household (HH) and spouse (1 =HH and spouse worked FT,  2 =HH worked FT; spouse worked < FT, 3 =HH worked FT; spouse did not work, 4 =HH worked < FT; spouse worked FT, 5 =HH worked < FT; spouse worked < FT, 6 =HH worked < FT; spouse did not work, 7 =HH did not work; spouse worked FT, 8 =HH did not work; spouse worked < FT, 9 =HH did not work; spouse did not work, 10 =Male HH worked FT; no spouse present, 11 =Male HH worked < FT; no spouse present, 12=Male HH did not work; no spouse present, 13 =Female HH worked FT; no spouse present, 14 =Female HH worked < FT; no spouse present, 15 =Female HH did not work; no spouse present)<br/>
29 - WkStatus: Work status of HH or spouse (1 =Husband and wife both in LF, both employed or in Armed Forces, 2 =Husband and wife both in LF, husband employed or in Armed Forces, wife unemployed, 3 =Husband in LF and wife not in LF, husband employed or in Armed Forces, 4 =Husband and wife both in LF, husband unemployed, wife employed or in Armed Forces, 5 =Husband and wife both in LF, husband unemployed, wife unemployed, 6 =Husband in LF, husband unemployed, wife not in LF, 7 =Husband not in LF, wife in LF, wife employed or in Armed Forces, 8 =Husband not in LF, wife in LF, wife unemployed, 9 =Neither husband nor wife in LF, 10 =Male HH with no wife present, HH in LF, employed or in Armed Forces, 11 =Male HH with no wife present, HH in LF and unemployed, 12 =Male HH with no wife present, HH not in LF, 13 =Female HH with no husband present, HH in LF, employed or in Armed Forces, 14 =Female HH with no husband present, HH in LF and unemployed, 15 =Female HH with no husband present, HH not in LF)<br/>
30 – Vehicles: number of vehicles owned (0, 1, 2, 3, 4, 5, 6=six or more)<br/>
 
Dependent variable <br/>
31 – HiValue: household owns home and has income over $150K (1=yes, 0=no)<br/>

## Modeling Approach
Multiple classification models were evaluated and compared, including:
- Logistic Regression<br/>
- Decision Trees<br/>
- Neural Networks<br/>
- XGBoost<br/>

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

## Tools & Technologies
- **JMP**
- Data preprocessing and feature engineering
- Classification modeling
- Profit-based model evaluation

## Key Takeaways
- Business-driven evaluation metrics are critical in data mining
- Profit matrices provide more realistic guidance than accuracy alone
- Model choice should reflect organizational goals, not just statistical performance
