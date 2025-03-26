from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Load the dataset and preprocess
input_file = "Input+DayMuscle.xlsx"  # Update this if needed
df = pd.read_excel(input_file)

# Encoding categorical variables
label_encoders = {}
for col in ["body_part_to_train"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].str.lower())  # Convert to lowercase before encoding
    label_encoders[col] = le

# Selecting features (X) and targets (Y: day1 to day6)
X = df[["day_commitment", "body_part_to_train"]]
Y = df[["day1", "day2", "day3", "day4", "day5", "day6"]]

# Handle missing values by filling with "Rest"
Y = Y.fillna("Rest")

# Encode target labels
for col in Y.columns:
    le = LabelEncoder()
    Y[col] = le.fit_transform(Y[col].astype(str))
    label_encoders[col] = le

# Train models for each day
models = {}
for day in Y.columns:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, Y[day])
    models[day] = model

# Function to predict workout plan
def predict_workout(day_commitment, body_part_to_train):
    body_part_encoded = label_encoders["body_part_to_train"].transform([body_part_to_train.lower()])[0]  # Convert input to lowercase
    input_data = np.array([[day_commitment, body_part_encoded]])
    predictions = {}
    for day, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[day] = label_encoders[day].inverse_transform([pred])[0]
    return predictions

# Create an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract inputs from the POST request
    day_commitment = data.get('day_commitment')
    body_part_to_train = data.get('body_part_to_train')
    
    if not day_commitment or not body_part_to_train:
        return jsonify({'error': 'Missing required input'}), 400
    
    # Get predictions
    try:
        result = predict_workout(day_commitment, body_part_to_train)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
