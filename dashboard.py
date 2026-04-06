import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Bank AI Dashboard", page_icon="🏦", layout="wide")
st.title("🏦 AI Banking Risk & NPA Prediction Dashboard")
st.markdown("An interactive dashboard powered by XGBoost and Explainable AI (SHAP).")


# ==========================================
# 2. CACHED DATA GENERATION (Loads instantly)
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    banks = [f"Bank_{i}" for i in range(1, 21)]
    dates = [f"{y}-{q}" for y in range(2008, 2026) for q in ['Q1', 'Q2', 'Q3', 'Q4']]

    data = []
    for bank in banks:
        base_npa = np.random.uniform(1.5, 5.0)
        for i, date in enumerate(dates):
            year = int(date[:4])
            gdp_growth = 7.0 + np.sin(i / 10) * 3 + np.random.normal(0, 0.5)
            repo_rate = 6.0 + np.cos(i / 8) * 2 + np.random.normal(0, 0.2)
            inflation = 5.0 + np.random.normal(0, 1.0)
            cd_ratio = 70 + np.random.normal(0, 5) + (repo_rate * 0.5)
            car = 12 + np.random.normal(0, 2)
            bank_size = np.random.uniform(10, 15)
            gross_npa = max(0.1, base_npa + (cd_ratio * 0.05) - (gdp_growth * 0.2) + np.random.normal(0, 0.5))

            data.append([bank, year, date, cd_ratio, gdp_growth, repo_rate, inflation, car, bank_size, gross_npa])

    df = pd.DataFrame(data,
                      columns=['Bank_Name', 'Year', 'Quarter_Date', 'CD_Ratio', 'GDP_Growth', 'Repo_Rate', 'Inflation',
                               'CAR', 'Bank_Size', 'Gross_NPA_Ratio'])
    df['CD_Ratio_Lag1Y'] = df.groupby('Bank_Name')['CD_Ratio'].shift(4)
    df['CD_Ratio_Lag2Y'] = df.groupby('Bank_Name')['CD_Ratio'].shift(8)
    return df.dropna()


# ==========================================
# 3. CACHED MODEL TRAINING
# ==========================================
@st.cache_resource
def train_model(df):
    train_df = df[df['Year'] <= 2021]
    test_df = df[df['Year'] >= 2022]

    features = ['CD_Ratio', 'CD_Ratio_Lag1Y', 'CD_Ratio_Lag2Y', 'GDP_Growth', 'Repo_Rate', 'Inflation', 'CAR',
                'Bank_Size']
    X_train, y_train = train_df[features], train_df['Gross_NPA_Ratio']
    X_test, y_test = test_df[features], test_df['Gross_NPA_Ratio']

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Generate overall predictions
    test_df = test_df.copy()
    test_df['Predicted_NPA'] = model.predict(X_test)

    return model, test_df, features, X_test


# Load everything
df = load_data()
model, test_df, features, X_test = train_model(df)

# ==========================================
# 4. SIDEBAR (User Controls)
# ==========================================
st.sidebar.header("⚙️ Dashboard Controls")
selected_bank = st.sidebar.selectbox("Select a Bank to Analyze:", test_df['Bank_Name'].unique())

# Filter data based on user selection
bank_data = test_df[test_df['Bank_Name'] == selected_bank]
latest_quarter = bank_data.iloc[-1]

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================

# Top Row: Key Metrics
st.markdown(f"### Current Risk Profile: **{selected_bank}** (Latest Quarter)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Actual Gross NPA", f"{latest_quarter['Gross_NPA_Ratio']:.2f}%")
col2.metric("AI Predicted NPA", f"{latest_quarter['Predicted_NPA']:.2f}%")
col3.metric("GDP Growth", f"{latest_quarter['GDP_Growth']:.2f}%")
col4.metric("CD Ratio", f"{latest_quarter['CD_Ratio']:.2f}%")

st.divider()

# Middle Row: Interactive Charts
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("📈 NPA Prediction Timeline (2022-2025)")
    # Using Plotly for a highly interactive chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=bank_data['Quarter_Date'], y=bank_data['Gross_NPA_Ratio'], mode='lines+markers', name='Actual NPA',
                   line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=bank_data['Quarter_Date'], y=bank_data['Predicted_NPA'], mode='lines+markers',
                             name='AI Predicted NPA', line=dict(color='red', dash='dash')))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.subheader("🧠 Why did the AI predict this?")
    st.write("Explainable AI (SHAP) shows which factors drove the risk highest for all banks in the test period.")

    # SHAP Summary Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig_shap, ax = plt.subplots(figsize=(6, 5))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False, plot_size=(6, 5))
    st.pyplot(fig_shap)