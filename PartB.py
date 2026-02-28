import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#load
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("Concrete_Data.csv")
#standardize column names and identify target
df = df.copy()
df.columns = [c.strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
target_candidates = [c for c in df.columns if "strength" in c.lower()]
y_col = target_candidates[0]
x_cols = [c for c in df.columns if c != y_col]
x_all = df[x_cols].to_numpy(dtype=np.float64)
y_all = df[y_col].to_numpy(dtype=np.float64)

#train/test split(testing sample from 501-630)
test_start = 500
test_end = 630

x_test = x_all[test_start:test_end]
y_test = y_all[test_start:test_end]

x_train = np.vstack((x_all[:test_start], x_all[test_end:]))
y_train = np.concatenate((y_all[:test_start], y_all[test_end:]))

#print("Train size: ", y_train.shape[0], x_train.shape)
#Train size:  900 (900, 8)
#print("Test size: ", y_test.shape)
#Test size:  (130,)
#print("mean(y_train)=", np.mean(y_train))

#GD trainer for univariate
#for standardize lr_m==lr_b,
#for raw lr_b ~1e-3, db not multiply by x, avoid tiny lr makes b update slowly
def fit_gd_univariate(x_t, y_t, lr_m, lr_b, epochs=2000, m0=0.0, b0=0.0):
    x_t = np.asarray(x_t, dtype=np.float64).reshape(-1)
    y_t = np.asarray(y_t, dtype=np.float64).reshape(-1)

    n = x_t.shape[0]
    m, b = m0, b0

    for e in range(epochs):
        yp = m*x_t+b
        err = yp-y_t

        dm = (2.0/n)*np.sum(err*x_t)
        db = (2.0/n)*np.sum(err)

        gd_norm = np.sqrt(dm**2+db**2)

        m -= dm * lr_m
        b -= db * lr_b

        if gd_norm < 1e-6:
            print("Converged at epoch", e)
            break

    return m, b

def fit_gd_multi(x_tr, y_tr, lr, ep, m0=1.0, b0=1.0, toler=1e-12, patience=50):
    x = np.asarray(x_tr, dtype=float) #x_tr: (900, 8)
    y = np.asarray(y_tr, dtype=float).reshape(-1) #y_tr: (900,)
    n, d = x.shape  # (n,8)

    m = np.full(d, m0, dtype=float) #m1...8=1
    b = float(b0)

    for e in range(ep):
        y_h = x@m+b
        err = y_h - y

        dm = (2.0 / n) * (x.T@err)
        db = (2.0 / n) * np.sum(err)
        m -= dm * lr
        b -= db * lr

        #no improve, early stop
        best = np.inf
        bad = 0
        cur_mse = mse(x@m+b, y)
        if cur_mse+toler < best:
            best = cur_mse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epochs {ep+1}, best_mse: {best}, cur_mse={cur_mse}")
            break
    return m, b

def predict_univariate(x,m,b):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return m*x+b

#metric
def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean((y_true-y_pred)**2)

def r2_from_mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    var = np.var(y_true)
    if var == 0:
        return np.nan
    return 1.0-mse(y_true, y_pred)/var

#standardize
def uni_std(x_train, x_test):
    x_train = np.asarray(x_train, dtype=np.float64).reshape(-1)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(-1)

    u = np.mean(x_train)
    sigma = np.std(x_train)
    if sigma == 0:
        sigma = 1.0

    x_train_scaled = (x_train - u)/sigma
    x_test_scaled = (x_test - u) / sigma

    return x_train_scaled, x_test_scaled

def multi_std(x_tr, x_te):
    scaler = StandardScaler()
    xtr_std = scaler.fit_transform(x_tr)
    xte_std = scaler.transform(x_te)
    return xtr_std, xte_std

#run experiment
def run_set_uni(set_n, trans_fn, lr, epochs):
    rows=[]
    for i in range(len(x_cols)):
        xtr = x_train[:, i]
        xte = x_test[:, i]

        feature = x_cols[i]

        if trans_fn is None:
            xtr_t, xte_t = xtr, xte
            lr_m = 1e-6
        else:
            xtr_t, xte_t = trans_fn(xtr, xte)
            lr_m = lr

        # print("xtr_t", xtr_t)
        # print("xte_t", xte_t)
        # print("Len check:", len(xtr_t), len(y_train))
        m, b = fit_gd_univariate(xtr_t, y_train, lr_m, lr, epochs)

        ytr_pred = predict_univariate(xtr_t, m, b)
        yte_pred = predict_univariate(xte_t, m, b)

        rows.append({
            "set": set_n,
            "feature": feature,
            "lr_m": lr_m,
            "lr_b": lr,
            "epochs": epochs,
            "m": round(m, 4),
            "b": round(b, 4),
            "train_mse": round(mse(y_train, ytr_pred), 4),
            "train_r2": round(r2_from_mse(y_train, ytr_pred), 4),
            "test_mse": round(mse(y_test, yte_pred), 4),
            "test_r2": round(r2_from_mse(y_test, yte_pred), 4)
        })

    return pd.DataFrame(rows)

def run_set_multi(set_n, xtr, ytr, xte, yte, w, b):
    ytr_pred = xtr@w+b
    yte_pred = xte@w+b

    print("set: ", set_n)
    print("train_mse:", round(mse(ytr, ytr_pred), 4))
    print("train_r2:", round(r2_from_mse(ytr, ytr_pred), 4))
    print("test_mse:", round(mse(yte, yte_pred), 4))
    print("test_r2:", round(r2_from_mse(yte, yte_pred), 4))

eph= [2000, 3000, 5000, 10000]

'''
#Set 1.1: standardized X(safe lr roughly < 1/lambda_max ~2, up to 0.5 can work, 0.01-0.1 common safe)
std_lr=[0.005, 0.01, 0.05, 0.1]
results_set1_1 = run_set_uni("Set1_standardized", uni_std, 0.05, 3000)

#Set 1.2: raw x (large x, small lr avoids diverge)
raw_lr=[1e-4, 1e-5, 1e-6, 1e-7] #1e-4, 1e-5 too large
lr_b=1e-3
results_set1_2 = run_set_uni("Set1_raw", None, lr_b, 10000)

result = pd.concat([results_set1_1, results_set1_2], ignore_index=True)

print(result.sort_values(["set", "train_r2"], ascending=[True, False])[
          ["set", "feature", "lr_m", "lr_b", "epochs", "m", "b", "train_mse", "train_r2", "test_mse", "test_r2"]
      ])
'''

'''
# x = [[3, 4, 5]]
# y = [4]
x = [[3,4,4], [4, 2, 1], [10, 2, 5], [3, 4, 5], [11, 1, 1]]
y=[3,2,8,4,5]
m, b = fit_gd_multi(x, y, 0.1, 1)
print("m=", np.round(m, 6))
print("b=", np.round(b, 6))
'''

'''
#Set 2.1: standardized
xtr_std, xte_std = multi_std(x_train, x_test)

m1, b1 = fit_gd_multi(xtr_std, y_train, 0.05, 5000)
run_set_multi("Set2_standardize", xtr_std, y_train, xte_std, y_test, m1, b1)
print("m=", np.round(m1, 4))
print("b=", np.round(b1, 4))


#Set 2.2: raw
m2, b2 = fit_gd_multi(xtr_std, y_train, 1e-4, 35000)
run_set_multi("Set2_raw", xtr_std, y_train, xte_std, y_test, m2, b2)
print("m=", np.round(m2, 4))
print("b=", np.round(b2, 4))
'''
