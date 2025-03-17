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
    database="salesdb"
)
cur = db.cursor()

# Suppress warnings
warnings.simplefilter("ignore")

# Load Dataset
df = pd.read_csv(r"C:\Users\tsmur\Downloads\salesrevenue.csv")
df = df.fillna(0)  # Fill missing values if any

# Define Features (X) and Target (Y)
X = df[['TV', 'Radio', 'Newspaper']]
Y = df['Sales']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# Real-time Prediction Loop
while True:
    # Get current timestamp
    current_time = datetime.datetime.now()
    time_updated = current_time.strftime("%Y-%m-%d %H:%M")

    # Predict for a sample input (first row of dataset)
    sample_input = X.iloc[0].values.reshape(1, -1)
    predicted_sales = model.predict(sample_input)[0]

    # Extract feature values
    tv_value, radio_value, newspaper_value = sample_input[0]

    # Accuracy Calculation
    model_accuracy = max(0, min(100, r2_score(y_test, model.predict(x_test)) * 100))

    # Print Predictions and Accuracy
    print(f"Prediction at {time_updated}:")
    print(f"  TV Budget: {tv_value}")
    print(f"  Radio Budget: {radio_value}")
    print(f"  Newspaper Budget: {newspaper_value}")
    print(f"  Predicted Sales: {predicted_sales:.2f} (Accuracy: {model_accuracy:.2f}%)")

    # Insert into Database
    sql = """INSERT INTO sales_predictions (time_updated, TV, Radio, Newspaper, Predicted_Sales) 
             VALUES (%s, %s, %s, %s, %s)"""
    try:
        print("Writing to the database...")
        cur.execute(sql, (time_updated, tv_value, radio_value, newspaper_value, predicted_sales))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("We have a problem:", e)

    time.sleep(1)  # Wait before running again

cur.close()
db.close()
