import pickle
import pandas as pd

MODEL_FILE = "linreg_model.pkl"

def predict_cost(energy, hour, dayofweek):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    X_pred = pd.DataFrame([[energy, hour, dayofweek]], columns=["energy_kWh_interval", "hour", "dayofweek"])
    return float(model.predict(X_pred)[0])