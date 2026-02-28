import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Concrete_Data.csv")

df.columns = df.columns.str.strip()

# Split into features and what we're trying to predict (strength)
feature_cols = [col for col in df.columns if col != "Concrete compressive strength(MPa, megapascals)"]
X = df[feature_cols]
y = df["Concrete compressive strength(MPa, megapascals)"]

test_start = 500
test_end = 630

X_test = X.iloc[test_start:test_end].reset_index(drop=True)
y_test = y.iloc[test_start:test_end].reset_index(drop=True)
X_train = pd.concat([X.iloc[:test_start], X.iloc[test_end:]]).reset_index(drop=True)
y_train = pd.concat([y.iloc[:test_start], y.iloc[test_end:]]).reset_index(drop=True)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("-" * 50)

# raw predictors (untransformed)
print("\n1. RUNNING MODEL WITH RAW PREDICTORS")
print("=" * 50)

X_train_raw = sm.add_constant(X_train)
X_test_raw = sm.add_constant(X_test)

model_raw = sm.OLS(y_train, X_train_raw).fit()

y_train_pred_raw = model_raw.predict(X_train_raw)
y_test_pred_raw = model_raw.predict(X_test_raw)

print(f"Training MSE: {mean_squared_error(y_train, y_train_pred_raw):.4f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_test_pred_raw):.4f}")
print(f"Training R²: {r2_score(y_train, y_train_pred_raw):.4f}")
print(f"Testing R²: {r2_score(y_test, y_test_pred_raw):.4f}")

# standardized predictors
print("\n2. RUNNING MODEL WITH STANDARDIZED PREDICTORS")


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

X_train_std = pd.DataFrame(X_train_std, columns=feature_cols)
X_test_std = pd.DataFrame(X_test_std, columns=feature_cols)

X_train_std = sm.add_constant(X_train_std)
X_test_std = sm.add_constant(X_test_std)

model_std = sm.OLS(y_train, X_train_std).fit()

y_train_pred_std = model_std.predict(X_train_std)
y_test_pred_std = model_std.predict(X_test_std)

print(f"Training MSE: {mean_squared_error(y_train, y_train_pred_std):.4f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_test_pred_std):.4f}")
print(f"Training R²: {r2_score(y_train, y_train_pred_std):.4f}")
print(f"Testing R²: {r2_score(y_test, y_test_pred_std):.4f}")

# log transformed predictors
print("\n3. RUNNING MODEL WITH LOG-TRANSFORMED PREDICTORS")
print("=" * 50)

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

X_train_log = pd.DataFrame(X_train_log, columns=feature_cols)
X_test_log = pd.DataFrame(X_test_log, columns=feature_cols)

X_train_log = sm.add_constant(X_train_log)
X_test_log = sm.add_constant(X_test_log)

model_log = sm.OLS(y_train, X_train_log).fit()

y_train_pred_log = model_log.predict(X_train_log)
y_test_pred_log = model_log.predict(X_test_log)

print(f"Training MSE: {mean_squared_error(y_train, y_train_pred_log):.4f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_test_pred_log):.4f}")
print(f"Training R²: {r2_score(y_train, y_train_pred_log):.4f}")
print(f"Testing R²: {r2_score(y_test, y_test_pred_log):.4f}")


# Coefficients and p-values
print("COEFFICIENTS AND P-VALUES FOR ALL THREE MODELS")


features = ['const'] + feature_cols

raw_coef = model_raw.params.values
raw_pvals = model_raw.pvalues.values

std_coef = model_std.params.values
std_pvals = model_std.pvalues.values

log_coef = model_log.params.values
log_pvals = model_log.pvalues.values

print(f"{'Feature':<35} {'Raw Model':>25} {'Standardized':>25} {'Log-transformed':>25}")
print(f"{'':<35} {'coef / p-value':>25} {'coef / p-value':>25} {'coef / p-value':>25}")
print("-" * 110)

for i, feat in enumerate(features):
    if feat == 'const':
        display_name = 'Intercept'
    else:
        display_name = feat.split('(')[0].strip()
    
    raw_str = f"{raw_coef[i]:.3f} / {raw_pvals[i]:.2e}"
    std_str = f"{std_coef[i]:.2f} / {std_pvals[i]:.2e}"
    log_str = f"{log_coef[i]:.2f} / {log_pvals[i]:.2e}"
    
    print(f"{display_name:<35} {raw_str:>25} {std_str:>25} {log_str:>25}")

