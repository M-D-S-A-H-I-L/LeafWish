import requests
import csv
from datetime import datetime

API_TOKEN = "YOUR_REAL_WAQI_API_KEY"
CITIES = ["New Delhi", "Shanghai", "Beijing", "Los Angeles"]
CSV_FILE = "data/aqi_data.csv"

def fetch_aqi(city):
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    res = requests.get(url).json()
    if res["status"] != "ok":
        return None
    data = res["data"]
    aqi = data["aqi"]
    temp = data["iaqi"].get("t", {"v": 25})["v"]
    humidity = data["iaqi"].get("h", {"v": 60})["v"]
    wind = data["iaqi"].get("w", {"v": 10})["v"]
    pm25 = data["iaqi"].get("pm25", {"v": 0})["v"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [timestamp, city, aqi, pm25, temp, humidity, wind]

def save_to_csv(row):
    try:
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("Error writing CSV:", e)

if __name__ == "__main__":
    # Optional: write header if CSV doesn't exist
    try:
        with open(CSV_FILE, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["datetime","city","aqi","pm25","temp","humidity","wind"])
    except FileExistsError:
        pass

    # Fetch AQI for all cities
    for city in CITIES:
        row = fetch_aqi(city)
        if row:
            save_to_csv(row)
            print(f"Saved data for {city}")
