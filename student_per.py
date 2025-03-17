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
    database="studentper"
)
cur = db.cursor()

# Suppress warnings
warnings.simplefilter("ignore")

# Load Dataset
df = pd.read_csv(r"C:\Users\tsmur\Downloads\StudentsPerformance.csv")
df = df.fillna(0)  # Fill missing values if any

# Rename columns to remove spaces
df.columns = df.columns.str.replace(" ", "_")

# Convert Categorical Data to Numeric
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"], drop_first=True)

# Define Features (X) and Targets (Y)
X = df.drop(columns=["math_score", "reading_score", "writing_score"])
Y_math = df["math_score"]
Y_reading = df["reading_score"]
Y_writing = df["writing_score"]
Y_overall = (Y_math + Y_reading + Y_writing) / 3  # Overall performance score

# Single Train-Test Split to ensure consistency
x_train, x_test, y_train, y_test = train_test_split(X, Y_math, test_size=0.2, random_state=42)
y_math_train, y_math_test = y_train, y_test
y_reading_train, y_reading_test = train_test_split(Y_reading, test_size=0.2, random_state=42)
y_writing_train, y_writing_test = train_test_split(Y_writing, test_size=0.2, random_state=42)
y_overall_train, y_overall_test = train_test_split(Y_overall, test_size=0.2, random_state=42)

# Train Models (Using Linear Regression)
math_model = LinearRegression()
reading_model = LinearRegression()
writing_model = LinearRegression()
overall_model = LinearRegression()

math_model.fit(x_train, y_math_train)
reading_model.fit(x_train, y_reading_train)
writing_model.fit(x_train, y_writing_train)
overall_model.fit(x_train, y_overall_train)

# Real-time Prediction Loop
while True:
    # Get current timestamp
    current_time = datetime.datetime.now()
    time_updated = current_time.strftime("%Y-%m-%d %H:%M")

    # Predicting for a sample student (taking the first row as an example)
    sample_student = X.iloc[0].values.reshape(1, -1)

    math_pred = math_model.predict(sample_student)
    reading_pred = reading_model.predict(sample_student)
    writing_pred = writing_model.predict(sample_student)
    overall_pred = overall_model.predict(sample_student)

    # Accuracy Calculation
    math_accuracy = max(0, min(100, r2_score(y_math_test, math_model.predict(x_test)) * 100))
    reading_accuracy = max(0, min(100, r2_score(y_reading_test, reading_model.predict(x_test)) * 100))
    writing_accuracy = max(0, min(100, r2_score(y_writing_test, writing_model.predict(x_test)) * 100))
    overall_accuracy = max(0, min(100, r2_score(y_overall_test, overall_model.predict(x_test)) * 100))

    # Print Predictions and Accuracy
    print(f"Prediction at {time_updated}:")
    print(f"  Math Score: {math_pred[0]:.2f} (Accuracy: {math_accuracy:.2f}%)")
    print(f"  Reading Score: {reading_pred[0]:.2f} (Accuracy: {reading_accuracy:.2f}%)")
    print(f"  Writing Score: {writing_pred[0]:.2f} (Accuracy: {writing_accuracy:.2f}%)")
    print(f"  Overall Performance: {overall_pred[0]:.2f} (Accuracy: {overall_accuracy:.2f}%)")

    # Insert into Database
    sql = """INSERT INTO student_scores (time_updated, Math, Reading, Writing, Overall) VALUES (%s, %s, %s, %s, %s)"""
    try:
        print("Writing to the database...")
        cur.execute(sql, (time_updated, math_pred[0], reading_pred[0], writing_pred[0], overall_pred[0]))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("We have a problem:", e)

    time.sleep(1)  # Wait before running again

cur.close()
db.close()
