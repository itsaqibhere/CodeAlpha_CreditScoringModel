# CodeAlpha_CreditScoringModel
Credit scoring model predicting loan applicant risk (Good/Bad) using Logistic Regression, Decision Tree, and Random Forest on the German Credit dataset. Includes feature engineering, class-imbalance handling, and evaluation via Precision/Recall/F1/ROC-AUC.

CREDIT SCORING MODEL

A machine learning project that predicts an individual's creditworthiness (Good vs Bad credit risk) using their financial and demographic history, built with classic classification algorithms.

OBJECTIVE
Given an applicant's financial profile — account status, existing debts, payment history, savings, employment, etc. — predict whether they are likely to be a Good or Bad credit risk. This is the same core problem banks and lenders solve when deciding whether to approve a loan.

DATASET
Statlog (German Credit Data) — UCI Machine Learning Repository.

1,000 loan applicants
20 features: checking/savings account status, credit history, loan purpose, credit amount, duration, employment, age, housing, job, and more
Target: credit risk (Good / Bad), with a natural ~70/30 class imbalance
german_credit.csv (raw, coded format) is included in this repo.

APPROACH

Data decoding — mapped the dataset's coded categorical values (e.g. A11, A34) to human-readable labels.
EDA — examined class balance, and how age, credit amount, and loan duration relate to risk.
Feature engineering:
credit_per_month — credit amount spread over loan duration (repayment burden proxy)
age_group — bucketed age ranges
Ordinal encoding for checking/savings account levels (these have a natural order, more funds = lower risk)
One-hot encoding for remaining nominal categorical fields
Modeling — trained and compared three classifiers: Logistic Regression (interpretable baseline), Decision Tree, Random Forest. All three use class_weight="balanced" to account for class imbalance.
Evaluation — Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices, and ROC curve comparison, since accuracy alone is misleading on imbalanced data.

RESULTS
Logistic Regression: Accuracy 0.712, Precision 0.514, Recall 0.760, F1 0.613, ROC-AUC 0.780
Decision Tree: Accuracy 0.632, Precision 0.434, Recall 0.747, F1 0.549, ROC-AUC 0.690
Random Forest: Accuracy 0.792, Precision 0.662, Recall 0.627, F1 0.644, ROC-AUC 0.814 (best overall)

Random Forest performed best overall (highest ROC-AUC and F1-Score). Logistic Regression had the highest Recall on defaulters, which can matter more than overall accuracy in a real lending context, since missing an actual defaulter is usually costlier than being cautious with a borderline applicant.

TECH STACK
Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

PROJECT STRUCTURE
Credit_Scoring_Model.ipynb — full notebook: EDA, feature engineering, modeling, evaluation
german_credit.csv — raw dataset
README.md

HOW TO RUN
git clone <this-repo-url>
cd credit-scoring-model
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Credit_Scoring_Model.ipynb
Run all cells to reproduce the analysis, charts, and metrics.

POSSIBLE EXTENSIONS

Try gradient boosting models (XGBoost / LightGBM) for higher ROC-AUC
Use SMOTE for minority-class oversampling as an alternative to class_weight
Hyperparameter tuning with GridSearchCV, optimized for recall on the "Bad" class
SHAP values for individual-level explainability
Probability calibration if used to generate an actual numeric credit score

