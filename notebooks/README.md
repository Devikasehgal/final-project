
# HR Attrition & Succession Planning - Logistic Regression Project

This project focuses on predicting employee attrition and understanding succession readiness using logistic regression, SHAP explainability, and HR-related features. The goal is to identify key drivers of attrition and provide HR stakeholders with interpretable, data-driven insights.

---

## 📊 Dataset & Objective

The dataset is sourced from an IBM HR analytics dataset, extended with **succession planning features** such as:
- `PotentialRating`
- `LeadershipScore`
- `SuccessorReady`

The primary target variable is **Attrition**, encoded as binary (`1 = Yes`, `0 = No`). The objective is twofold:
1. Predict whether an employee is at risk of leaving.
2. Explain why, using interpretable AI.

---

## 🧹 Data Cleaning & Preprocessing

- Loaded the dataset with 368 records and multiple categorical & numerical fields.
- Converted binary categorical fields (`Yes`/`No`) into numeric (`1`/`0`).
- Encoded ordinal succession fields (`Low`, `Medium`, `High`) numerically.
- Addressed class imbalance using **SMOTE** to balance the `Attrition` target.
- Applied `StandardScaler` for model-ready feature scaling.

---

## 📊 Exploratory Data Analysis

- Analyzed class distribution of attrition.
- Examined feature distributions and correlations.
- Visualized key attributes like **OverTime**, **JobSatisfaction**, **LeadershipScore**, etc.
- Understood how attrition varies with succession planning indicators.

---

## 🤖 Model Building

### Models Tested:
- **Random Forest**: Very poor accuracy (~0.18), failed to generalize.
- **XGBoost**: Overfit on minority class, high recall but low precision.
- **Logistic Regression**: Provided the best balance between accuracy, interpretability, and recall.

### Final Chosen Model:
✅ **Logistic Regression**
- Accuracy: **0.74**
- Recall (Attrition): **0.69**
- ROC AUC: ~**0.72**

---

## 📈 Model Interpretability

Used **SHAP** (SHapley Additive Explanations) to interpret feature impact:
- Created global summary plot of top features.
- Generated individual waterfall plots to explain predictions for each employee.

### Key Insights:
- Employees with **high OverTime**, **low income**, and **low leadership scores** are more likely to leave.
- Strong **cultural fit**, **work-life balance**, and **job involvement** reduce attrition risk.

---

## 💡 Final Deliverables

- Logistic regression model with balanced performance
- SHAP-based explainability for both global and individual analysis
- Confusion matrix and threshold tuning for HR use cases
- Streamlit dashboard to visualize predictions (optional)

---

## 🏷️ Hashtags
#LogisticRegression #HRAnalytics #SHAP #SuccessionPlanning #AttritionPrediction #ExplainableAI #Python #ScikitLearn #MachineLearning

