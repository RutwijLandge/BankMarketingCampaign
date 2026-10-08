# Bank Marketing Campaign Classification

A machine learning project that predicts whether a customer will subscribe to a bank term deposit based on customer characteristics, previous campaign interactions, and economic indicators.

## Project Overview

The objective of this project is to build a classification model that can identify customers who are more likely to subscribe to a bank term deposit.

The project uses the UCI Bank Marketing dataset containing customer information and marketing campaign data.

Since the target variable is highly imbalanced, multiple evaluation metrics such as Precision, Recall, F1-Score, and ROC-AUC were used instead of relying only on accuracy.

## Dataset

The dataset contains 41,188 customer records and 21 original features.

### Target Variable

`y`

- `0` → Customer did not subscribe
- `1` → Customer subscribed

The target distribution is approximately:

- No: 88.73%
- Yes: 11.27%

### Important Features

Some of the important variables include:

- Age
- Job
- Marital Status
- Education
- Housing Loan
- Personal Loan
- Contact Type
- Campaign
- Previous Contact Information
- Previous Campaign Outcome
- Employment Variation Rate
- Consumer Price Index
- Consumer Confidence Index
- Euribor 3 Month Rate
- Number of Employees

The `duration` feature was removed because it represents information available after/during the call and could introduce target leakage when making predictions before contacting a customer.

## Project Workflow

The project follows the following machine learning workflow:

1. Data Loading
2. Exploratory Data Analysis
3. Target Distribution Analysis
4. Feature Engineering
5. Train-Test Split
6. Data Preprocessing
7. Model Training
8. Model Comparison
9. Hyperparameter Tuning
10. Feature Selection Experiment
11. Model Evaluation
12. Feature Importance Analysis
13. Final Model Saving

## Feature Engineering

A new feature was created:

`was_previously_contacted`

This feature indicates whether the customer had been contacted during a previous campaign.

- `1` → Previously contacted
- `0` → Not previously contacted

The original `pdays` value of `999` was used to determine whether a previous contact existed.

## Data Preprocessing

A Scikit-learn preprocessing pipeline was used.

### Numerical Features

Numerical variables were standardized using:

`StandardScaler`

### Categorical Features

Categorical variables were converted into numerical representations using:

`OneHotEncoder`

### ColumnTransformer

`ColumnTransformer` was used to apply different preprocessing operations to numerical and categorical features.

### Pipeline

Scikit-learn `Pipeline` was used to combine preprocessing and model training into a reusable workflow.

## Models Evaluated

The following classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8349 | 0.3675 | 0.6455 | 0.4683 | 0.8010 |
| Decision Tree | 0.8335 | 0.3618 | 0.6261 | 0.4586 | 0.7788 |
| Random Forest | 0.8779 | 0.4602 | 0.4860 | 0.4727 | 0.7865 |

## Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest model.

The search optimized the model based on F1-score because the target variable is imbalanced and both precision and recall are important.

### Best Parameters

- `n_estimators`: 200
- `max_depth`: 16
- `min_samples_split`: 5
- `min_samples_leaf`: 1
- `max_features`: sqrt

### Tuned Random Forest Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8688 |
| Precision | 0.4413 |
| Recall | 0.6196 |
| F1 Score | 0.5155 |
| ROC-AUC | 0.8119 |

## Feature Selection Experiment

Feature selection was also evaluated using `SelectFromModel` with a Random Forest estimator.

### Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned Random Forest | 0.8688 | 0.4413 | 0.6196 | 0.5155 | **0.8119** |
| Random Forest + Feature Selection | **0.8699** | **0.4441** | 0.6164 | **0.5162** | 0.8042 |

Although feature selection slightly improved accuracy, precision, and F1-score, the tuned Random Forest achieved better recall and ROC-AUC.

Therefore, the tuned Random Forest with the full engineered feature set was selected as the final model.

## Why Recall and F1-Score Matter

The dataset contains significantly more customers who did not subscribe than customers who subscribed.

Therefore, accuracy alone can be misleading.

Recall is important because the model should identify as many potential subscribers as possible.

F1-score provides a balance between precision and recall.

ROC-AUC was also used to evaluate the model's ability to distinguish between customers who subscribed and those who did not across different classification thresholds.

## Feature Importance

Random Forest feature importance was analyzed to understand which variables contributed most to the model's predictions.

The analysis helps identify influential customer, campaign, and economic indicators.

Feature importance represents the model's contribution/association with predictions and should not be interpreted as causal relationships.

## Final Model

The final model is a tuned Random Forest classifier integrated into a Scikit-learn preprocessing pipeline.

The trained model was saved using Joblib:

`bank_marketing_random_forest.pkl`

## Project Structure

```text
Bank_Campaign/
│
├── data/
│   └── bank-additional-full.csv
│
├── notebook/
│   └── Bank_Marketing_Campaign_Classification.ipynb
│
├── models/
│   └── bank_marketing_random_forest.pkl
│
├── visualizations/
│
├── README.md
├── requirements.txt
└── .gitignore
```
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Jupyter Notebook
Installation

Clone the repository:
```
git clone https://github.com/RutwijLandge/BankMarketingCampaign
cd Bank_Campaign
```
Create a virtual environment:
```
python -m venv .venv
```
Activate it on Windows:
```
.venv\Scripts\activate
```
Install dependencies:
```
pip install -r requirements.txt
```
Running the Project

Open the notebook:
```
jupyter notebook
```
Then open:
```
notebook/Bank_Marketing_Campaign_Classification.ipynb
```
Run the notebook cells sequentially to reproduce the analysis and model training.

Business Use Case

A bank can use this type of predictive model to prioritize customers who are more likely to subscribe to a term deposit.

Instead of treating every customer equally, marketing teams could use model predictions to focus their efforts on higher-probability customers.

This can potentially improve campaign efficiency and reduce unnecessary customer outreach.

Limitations
The dataset represents a specific bank marketing campaign and may not generalize to other banks or markets.
Model predictions depend on the quality and relevance of the available features.
Feature importance indicates model behavior and does not establish causality.
The duration feature was excluded because it would not be available for a genuine pre-contact prediction scenario.
Future Improvements
Experiment with XGBoost and LightGBM.
Perform probability threshold optimization.
Apply advanced class-imbalance techniques such as SMOTE.
Perform cross-validation with additional evaluation metrics.
Deploy the model using FastAPI or Flask.
Build a web interface for customer prediction.
Monitor model performance after deployment.
Author

Rutwij Landge

GitHub: https://github.com/RutwijLandge