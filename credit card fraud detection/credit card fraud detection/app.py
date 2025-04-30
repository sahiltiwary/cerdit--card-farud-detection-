import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
from flask_cors import CORS

# Load trained models
model = joblib.load("voting_classifier.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

# Flask app setup
app = Flask(__name__, template_folder="templates")
CORS(app)

# Serve Frontend
@app.route('/')
def index():
    return render_template("index.html")

# Serve CSS from templates folder
@app.route('/templates/style.css')
def serve_css():
    return send_from_directory('templates', 'style.css')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json

    # Create DataFrame
    # Create DataFrame and ensure correct data types
    df = pd.DataFrame([data])

    # Explicitly convert numeric fields
    df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')

    # Check for invalid or missing numeric values
    if df['Transaction_Amount'].isnull().any():
        return jsonify({"error": "Transaction_Amount must be a valid number."}), 400

    # Apply feature engineering
    df['Transaction_Frequency'] = df.groupby('Transaction_ID')['Transaction_Amount'].transform('count')
    df['Avg_Transaction_Amount'] = df.groupby('Transaction_ID')['Transaction_Amount'].transform('mean')
    df['Transaction_Ratio'] = df['Transaction_Amount'] / df['Avg_Transaction_Amount']

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=["Transaction_Type", "Device_Type"])

    # Ensure all columns match the training set
    missing_cols = set(training_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[training_columns]

    # Standardize the features
    unseen_X = scaler.transform(df_encoded)

    # Make prediction
    prediction = model.predict(unseen_X)[0]
    probability = model.predict_proba(unseen_X)[0, 1]

    result = {
        "Fraudulent_Transaction": int(prediction),
        "Fraud_Probability": float(probability)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
