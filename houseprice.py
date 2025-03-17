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
    database="house"
)
cur = db.cursor()

# Suppress warnings
warnings.simplefilter("ignore")

# Load Dataset (for Excel file)
df = pd.read_excel(r"C:\Users\tsmur\Downloads\HousePrice.xlsx")  # Update the file path as needed
df = df.fillna(0)  # Fill missing values if any

# Rename columns to remove spaces and make them consistent
df.columns = df.columns.str.replace(" ", "_")

# Rename the specific column to match the database column name
df.rename(columns={"distance_to_the_nearest_MRT_station": "distance_mrt"}, inplace=True)

# Define Features (X) and Target (Y)
X = df.drop(columns=["house_price"])
Y = df["house_price"]

# Train-Test Split to ensure consistency
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Model (Using Linear Regression)
house_price_model = LinearRegression()
house_price_model.fit(x_train, y_train)

# Real-time Prediction Loop
while True:
    # Get current timestamp
    current_time = datetime.datetime.now()
    time_updated = current_time.strftime("%Y-%m-%d %H:%M")

    # Predicting for a sample (taking the first row as an example)
    sample_house = X.iloc[0].values.reshape(1, -1)
    price_pred = house_price_model.predict(sample_house)

    # Accuracy Calculation
    price_accuracy = max(0, min(100, r2_score(y_test, house_price_model.predict(x_test)) * 100))

    # Print Predictions and Accuracy
    print(f"Prediction at {time_updated}:")
    print(f"  Predicted House Price: {price_pred[0]:.2f} (Accuracy: {price_accuracy:.2f}%)")

    # Insert into Database
    sql = """INSERT INTO houseprice (date, house_age, distance_mrt, no_convinience_stores, latitude, longitude, house_price) 
             VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    
    # Assuming you have the relevant data for the insertion
    try:
        print("Writing to the database...")
        cur.execute(sql, (time_updated, X.iloc[0]['house_age'], X.iloc[0]['distance_mrt'], X.iloc[0]['no_convinience_stores'],
                          X.iloc[0]['latitude'], X.iloc[0]['longitude'], price_pred[0]))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("We have a problem:", e)

    time.sleep(1)  # Wait before running again

cur.close()
db.close()
