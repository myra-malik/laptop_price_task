# data_preprocessing.py — based on your notebook steps (student style)

import pandas as pd
import numpy as np

def load_and_preprocess(csv_path, test_size=0.2, seed=36):
    # load
    df = pd.read_csv(csv_path).copy()

    # BASIC NUMBER CLEANING remove letters
    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)

    # weight cleaning
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

    # MEMORY CLEANING 
    df["Memory"] = df["Memory"].astype(str).str.replace("TB", "000GB", regex=False)

    # replace " + " with space 
    df["Memory"] = df["Memory"].str.replace(r"\s*\+\s*", " ", regex=True)

    # extract sizes 
    memory_new = pd.DataFrame({
        "HDD":            df["Memory"].str.extract(r"(\d+)GB HDD")[0].fillna(0).astype(int),
        "SSD":            df["Memory"].str.extract(r"(\d+)GB SSD")[0].fillna(0).astype(int),
        "Hybrid":         df["Memory"].str.extract(r"(\d+)GB Hybrid")[0].fillna(0).astype(int),
        "Flash_Storage":  df["Memory"].str.extract(r"(\d+)GB Flash")[0].fillna(0).astype(int),
    }, index=df.index)
    df = pd.concat([df, memory_new], axis=1)

    # CPU CLEANING
    # brand
    df["Cpu_Brand"] = df["Cpu"].astype(str).str.split().str[0]
    # GHz value
    df["Cpu_Speed"] = df["Cpu"].astype(str).str.extract(r"(\d+\.?\d*)GHz").astype(float)

    # SCREEN CLEANING 
    df["ScreenWidth"]  = df["ScreenResolution"].astype(str).str.extract(r"(\d+)x")[0].astype(int)
    df["ScreenHeight"] = df["ScreenResolution"].astype(str).str.extract(r"x(\d+)")[0].astype(int)
    df["Touchscreen"]  = df["ScreenResolution"].astype(str).str.contains("Touchscreen").astype(int)

    # GPU CLEANING 
    df["Gpu_Brand"] = df["Gpu"].astype(str).str.split().str[0]

    # OS 
    def simplify_os(os_string):
        s = str(os_string)
        if "Windows" in s:
            return "Windows"
        elif "Mac" in s:
            return "Mac"
        elif "Linux" in s:
            return "Linux"
        elif "No OS" in s:
            return "No OS"
        else:
            return "Other"
    df["OpSys_Simplified"] = df["OpSys"].apply(simplify_os)

    # target 
    y = df["Price"].to_numpy(dtype=float).ravel()
    X = df.drop(columns=["Price"])

    # encode 
    X_enc = pd.get_dummies(X, drop_first=True)
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_np = X_enc.to_numpy(dtype=float)
    feature_columns = list(X_enc.columns)

    # split
    n = len(X_np)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int((1 - test_size) * n)
    train_idx, test_idx = idx[test_size_n:], idx[:test_size_n]   

    x_train_encoded = X_np[train_idx]
    x_test_encoded  = X_np[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    return x_train_encoded, x_test_encoded, y_train, y_test, feature_columns
