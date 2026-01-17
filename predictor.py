import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

CSV_FILE = "data/aqi_data.csv"

def train_model():
    df = pd.read_csv(CSV_FILE, parse_dates=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    X = df[["hour","day","month","temp","humidity","wind","pm25"]]
    y = df["aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    print("Model trained")
    return model

def predict_next_hours(model, hours=6):
    now = datetime.now()
    predictions = []
    for i in range(1, hours+1):
        future_time = now + timedelta(hours=i)
        X_future = [[
            future_time.hour,
            future_time.day,
            future_time.month,
            25,    # placeholder temp
            60,    # placeholder humidity
            10,    # placeholder wind
            50     # placeholder pm25
        ]]
        pred = model.predict(X_future)[0]
        predictions.append({"time": future_time.strftime("%Y-%m-%d %H:%M"), "aqi": round(pred)})
    return predictions

if __name__ == "__main__":
    model = train_model()
    next_hours = predict_next_hours(model)
    for p in next_hours:
        print(p)
