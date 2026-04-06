import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from linearmodels.panel import PanelOLS
import xgboost as xgb
import shap

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# STEP 1: MOCK DATA GENERATION (PANEL DATA)
# ==========================================
print("Generating mock panel dataset...")
np.random.seed(42)

banks = [f"Bank_{i}" for i in range(1, 21)]  # 20 Banks
years = range(2008, 2026)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# Create a time index
dates = [f"{y}-{q}" for y in years for q in quarters]

data = []
for bank in banks:
    # Introduce some bank-specific base risk to simulate real-world variance
    base_npa = np.random.uniform(1.5, 5.0)

    for i, date in enumerate(dates):
        year = int(date[:4])

        # Macro variables (mostly time-dependent, same for all banks)
        gdp_growth = 7.0 + np.sin(i / 10) * 3 + np.random.normal(0, 0.5)
        repo_rate = 6.0 + np.cos(i / 8) * 2 + np.random.normal(0, 0.2)
        inflation = 5.0 + np.random.normal(0, 1.0)

        # Bank specific variables
        cd_ratio = 70 + np.random.normal(0, 5) + (repo_rate * 0.5)  # CD ratio slightly influenced by rates
        car = 12 + np.random.normal(0, 2)
        bank_size = np.random.uniform(10, 15)  # Log of assets

        # Target Variable: Gross NPA (Simulating relationship with lagged CD Ratio and low GDP)
        # We add some noise to make the ML models work for it
        gross_npa = base_npa + (cd_ratio * 0.05) - (gdp_growth * 0.2) + np.random.normal(0, 0.5)
        gross_npa = max(0.1, gross_npa)  # NPAs can't be negative

        data.append([bank, year, date, cd_ratio, gdp_growth, repo_rate, inflation, car, bank_size, gross_npa])

columns = ['Bank_Name', 'Year', 'Quarter_Date', 'CD_Ratio', 'GDP_Growth', 'Repo_Rate', 'Inflation', 'CAR', 'Bank_Size', 'Gross_NPA_Ratio']
df = pd.DataFrame(data, columns=columns)

# FIX: Convert the text '2008-Q1' into actual DateTime objects so PanelOLS can understand the time dimension
df['Quarter_Date'] = pd.to_datetime(df['Quarter_Date'].str.replace('-Q1', '-03-31').str.replace('-Q2', '-06-30').str.replace('-Q3', '-09-30').str.replace('-Q4', '-12-31'))

# ==========================================
# STEP 2: FEATURE ENGINEERING & PREPROCESSING
# ==========================================
print("Performing Feature Engineering...")

# 1. Create Lagged Variables for CD Ratio (Crucial for time-series / causality)
# We lag by 4 quarters (1 year) and 8 quarters (2 years)
df['CD_Ratio_Lag1Y'] = df.groupby('Bank_Name')['CD_Ratio'].shift(4)
df['CD_Ratio_Lag2Y'] = df.groupby('Bank_Name')['CD_Ratio'].shift(8)

# Drop NaN values created by lagging
df = df.dropna()

# 2. Time-Based Train-Test Split (Train: 2008-2021, Test: 2022-2025)
train_df = df[df['Year'] <= 2021].copy()
test_df = df[df['Year'] >= 2022].copy()

# Features and Target
features = ['CD_Ratio', 'CD_Ratio_Lag1Y', 'CD_Ratio_Lag2Y', 'GDP_Growth', 'Repo_Rate', 'Inflation', 'CAR', 'Bank_Size']
target = 'Gross_NPA_Ratio'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 3. Standard Scaling (Important for Deep Learning and Econometrics)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

# ==========================================
# STEP 3: MODEL BUILDING
# ==========================================

# --- Model 1: Baseline Panel Econometric Model (Fixed Effects) ---
print("\nTraining Baseline Panel Model...")
# Panel data requires MultiIndex (Entity, Time)
panel_train = train_df.set_index(['Bank_Name', 'Quarter_Date'])
panel_X = panel_train[features]
panel_y = panel_train[target]

# Using Pooled OLS as a baseline
panel_model = PanelOLS(panel_y, panel_X, entity_effects=True)
panel_results = panel_model.fit()
print("Panel Model R-squared:", panel_results.rsquared)

# --- Model 2: Tree-Based ML Model (XGBoost) ---
print("\nTraining XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# --- Model 3: Deep Learning (Basic Sequential Model for Time-Series logic) ---
# Note: For a true LSTM, you need to reshape data to 3D: [samples, time_steps, features]
# Here we use a dense neural network as a placeholder for the DL requirement
print("\nTraining Deep Learning Model (Keras)...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Linear activation for regression
])

dl_model.compile(optimizer='adam', loss='mse')
# Training silently (verbose=0) to keep output clean
dl_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# ==========================================
# STEP 4: EVALUATION METRICS
# ==========================================
print("\nEvaluating Models on Future Data (2022-2025)...")

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_dl = dl_model.predict(X_test_scaled).flatten()


def print_metrics(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")


print_metrics("XGBoost", y_test, y_pred_xgb)
print_metrics("Deep Learning", y_test, y_pred_dl)

# Plot Actual vs Predicted for a single bank to visualize performance
sample_bank = test_df[test_df['Bank_Name'] == 'Bank_1']
plt.figure(figsize=(10, 5))
plt.plot(sample_bank['Quarter_Date'], sample_bank['Gross_NPA_Ratio'], label='Actual NPA', marker='o')
plt.plot(sample_bank['Quarter_Date'], xgb_model.predict(sample_bank[features]), label='Predicted NPA (XGBoost)',
         marker='x', linestyle='--')
plt.title("Actual vs Predicted Gross NPA for Bank_1 (Test Period)")
plt.xticks(rotation=45)
plt.ylabel("Gross NPA Ratio (%)")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: EXPLAINABLE AI (XAI) using SHAP
# ==========================================
print("\nGenerating SHAP explanations for XGBoost Model...")

# Initialize JavaScript visualizations in Jupyter
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Plot the Summary Plot
# This will show you exactly how much CD_Ratio_Lag1Y impacts the NPA prediction
# compared to GDP_Growth or Inflation.
shap.summary_plot(shap_values, X_test, feature_names=features)