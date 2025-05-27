import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ========== 1. PAGE SETUP ==========
st.set_page_config(page_title="SuccessionAI Dashboard", layout="wide")
st.title("📊 SuccessionAI: HR Analytics Dashboard")

# ========== 2. LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("../data/raw/ibm_attrition_plus_succession_reordered.csv", sep=";")
    return df

df = load_data()
# ========== 2.1 SIDEBAR FILTERS ==========
st.sidebar.header("🔍 Filter Employees")

# Dropdowns using actual data
selected_dept = st.sidebar.selectbox("Select Department", ["All"] + sorted(df['Department'].dropna().unique().tolist()))
selected_role = st.sidebar.selectbox("Select Job Role", ["All"] + sorted(df['JobRole'].dropna().unique().tolist()))
selected_tenure = st.sidebar.selectbox("Select Tenure Bucket", ["All"] + sorted(df['YearsAtCompany'].dropna().unique().tolist()))
high_risk_only = st.sidebar.checkbox("Only show High Attrition Risk", value=False)

# Apply filters
df_filtered = df.copy()
if selected_dept != "All":
    df_filtered = df_filtered[df_filtered['Department'] == selected_dept]
if selected_role != "All":
    df_filtered = df_filtered[df_filtered['JobRole'] == selected_role]
if selected_tenure != "All":
    df_filtered = df_filtered[df_filtered['YearsAtCompany'] == selected_tenure]
if high_risk_only:
    df_filtered = df_filtered[df_filtered['Attrition'] == 'Yes']

st.write(f"Showing {len(df_filtered)} matching employees.")


# ========== 3. FEATURE ENGINEERING ==========
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop non-numeric or identifier fields for modeling
exclude_cols = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
df_model = df.drop(columns=exclude_cols)

# Encode categorical variables
df_model = pd.get_dummies(df_model, drop_first=True)

# ========== 4. TRAIN-TEST SPLIT ==========
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ========== 5. SCALE NUMERIC FEATURES ==========
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols)

# ========== 6. MODEL TRAINING ==========
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ========== 7. PREDICTIONS ==========
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# ========== 8. METRICS ==========
st.subheader("📈 Model Performance")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 SHAP Insights", "🔍 Single Prediction", "📁 Raw Data"])


with tab1:
    st.subheader("📈 Model Performance Overview")
    st.caption("ℹ️ Charts reflect current sidebar filters.")

    # Metric Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df_filtered))
    col2.metric("At Risk (Attrition=Yes)", df_filtered['Attrition'].value_counts().get('Yes', 0))
    col3.metric("Successor Ready", df_filtered['SuccessorReady'].value_counts().get('Yes', 0))

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix:**")
    fig_conf, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig_conf)

    roc = roc_auc_score(y_test, y_proba)

        # ========== 📊 ATTRITION BY DEPARTMENT ==========
    st.markdown("### 🔹 Attrition by Department")
    attr_dept = df_filtered.groupby(['Department'])['Attrition'].value_counts(normalize=True).unstack().fillna(0)
    attr_dept['Yes'] *= 100  # convert to percentage
    attr_dept = attr_dept.sort_values('Yes', ascending=False)

    fig_attr, ax = plt.subplots()
    sns.barplot(x=attr_dept['Yes'], y=attr_dept.index, palette="coolwarm", ax=ax)
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate by Department")
    st.pyplot(fig_attr)

    # ========== 📈 POTENTIAL VS READINESS MATRIX ==========
    st.markdown("### 🔹 Succession Potential vs Readiness")
    matrix_df = df_filtered.copy()
    matrix_df = matrix_df[
        matrix_df['SuccessorReady'].notna() &
        matrix_df['PotentialRating'].notna()
    ]

    if not matrix_df.empty:
        fig_matrix, ax = plt.subplots()
        sns.heatmap(
            pd.crosstab(matrix_df['PotentialRating'], matrix_df['SuccessorReady']),
            annot=True, cmap="YlGnBu", fmt='d', ax=ax
        )
        ax.set_xlabel("Successor Ready")
        ax.set_ylabel("Potential Rating")
        ax.set_title("Succession Readiness Matrix")
        st.pyplot(fig_matrix)
    else:
        st.info("No data available for PotentialRating vs SuccessorReady.")

    # ========== 📉 TENURE BUCKET VS ATTRITION ==========
    st.markdown("### 🔹 Attrition by Tenure Bucket")
    tenure_df = df_filtered.copy()
    if 'YearsAtCompany' in tenure_df.columns:
        fig_tenure, ax = plt.subplots()
        sns.countplot(data=tenure_df, x='YearsAtCompany', hue='Attrition', palette='Set2', ax=ax)
        ax.set_title('Attrition Distribution by Tenure Bucket')
        ax.set_xlabel('YearsAtCompany')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig_tenure)
    else:
        st.info("TenureBucket column not found in data.")

    st.metric(label="ROC AUC Score", value=f"{roc:.2f}")

# ========== 9. SHAP EXPLAINABILITY ==========

with tab2:
    st.subheader("🧠 SHAP Feature Importance")

    # Prepare data for SHAP
    X_sample = X_test_scaled.copy()
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # ✅ Extract correct SHAP values for binary classification
    shap_vals_to_plot = shap_values[..., 1]

    # ✅ Now shape matches
    assert shap_vals_to_plot.shape == X_sample.shape, f"Mismatch: SHAP {shap_vals_to_plot.shape}, X {X_sample.shape}"

    # ✅ SHAP Summary Bar Plot
    shap.summary_plot(shap_vals_to_plot, X_sample, plot_type="bar", show=False)
    st.pyplot(plt.gcf())  # Get current figure

    # ✅ SHAP Beeswarm Plot
    shap.summary_plot(shap_vals_to_plot, X_sample, show=False)
    st.pyplot(plt.gcf())  # Get current figure


# ========== 10. Optional: Waterfall Plot for 1 Prediction ==========
with tab3:
    st.subheader("🔍 Explain Single Prediction")

    i = st.slider("Select index to explain", 0, X_sample.shape[0] - 1, 0)

    fig_water = plt.figure()
    shap.plots.waterfall(shap.Explanation(
        values=shap_vals_to_plot[i],
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[i],
        feature_names=X_sample.columns.tolist()
    ), show=False)
    st.pyplot(fig_water)

with tab4:
    st.subheader("📁 Raw Data Viewer")
    st.dataframe(df_filtered)
