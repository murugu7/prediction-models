import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
import datetime
import mysql.connector  # MySQL for XAMPP

# Database Connection (for XAMPP)
db = mysql.connector.connect(
    host="127.0.0.1",  # Use 127.0.0.1 instead of localhost
    user="root",        # Default XAMPP user
    passwd="",          # Default XAMPP has no password (change if needed)
    database="electricity_db"
)
cur = db.cursor()

# Suppress warnings
warnings.simplefilter("ignore")

# Load Dataset
df = pd.read_csv(r"C:\Users\tsmur\Downloads\powerconsumption.csv")
df = df.fillna(0)  # Fill missing values if any

# Rename columns to remove spaces
df.columns = df.columns.str.replace(" ", "_")

# Create total power consumption as target variable
df["Total_PowerConsumption"] = df["PowerConsumption_Zone1"] + df["PowerConsumption_Zone2"] + df["PowerConsumption_Zone3"]

# Define Features (X) and Target (Y)
X = df.drop(columns=["Datetime", "PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3", "Total_PowerConsumption"], errors="ignore")  # Avoid KeyError
Y = df["Total_PowerConsumption"]  # Using total power consumption as the target variable

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Model (Using Linear Regression)
model = LinearRegression()
model.fit(x_train, y_train)

# Real-time Prediction Loop
while True:
    # Get current timestamp
    current_time = datetime.datetime.now()
    time_updated = current_time.strftime("%Y-%m-%d %H:%M")

    # Predicting for a sample row (taking the first row as an example)
    sample_data = X.iloc[0].values.reshape(1, -1)
    predicted_consumption = model.predict(sample_data)[0]

    # Accuracy Calculation
    accuracy = max(0, min(100, r2_score(y_test, model.predict(x_test)) * 100))

    # Extracting parameter values from the sample row
    sample_row = df.iloc[0]
    temperature = sample_row["Temperature"]
    humidity = sample_row["Humidity"]
    wind_speed = sample_row["WindSpeed"]
    general_diffuse_flows = sample_row["GeneralDiffuseFlows"]
    diffuse_flows = sample_row["DiffuseFlows"]

    # Print Predictions
    print(f"Prediction at {time_updated}:")
    print(f"  Temperature: {temperature:.2f}°C")
    print(f"  Humidity: {humidity:.2f}%")
    print(f"  Wind Speed: {wind_speed:.2f} m/s")
    print(f"  General Diffuse Flows: {general_diffuse_flows:.2f}")
    print(f"  Diffuse Flows: {diffuse_flows:.2f}")
    print(f"  Predicted Total Power Consumption: {predicted_consumption:.2f} kWh (Accuracy: {accuracy:.2f}%)")

    # Insert into Database
    sql = """INSERT INTO power_consumption (time_updated, temperature, humidity, wind_speed, general_diffuse_flows, 
             diffuse_flows, predicted_consumption) 
             VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    try:
        print("Writing to the database...")
        cur.execute(sql, (time_updated, temperature, humidity, wind_speed, general_diffuse_flows,
                          diffuse_flows, predicted_consumption))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("Database error:", e)

    time.sleep(1)  # Wait before running again

cur.close()
db.close()
