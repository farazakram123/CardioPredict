from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('cardio_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print("Error: Model files not found. Please train the model first.")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        })
    
    try:
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        chest_pain_type = request.form['chest_pain_type']
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ecg = request.form['resting_ecg']
        max_hr = int(request.form['max_hr'])
        exercise_angina = request.form['exercise_angina']
        oldpeak = float(request.form['oldpeak'])
        st_slope = request.form['st_slope']
        
        # Create a dataframe with the input (same format as training data)
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })
        
        # Apply the same one-hot encoding as training (drop_first=True)
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure all columns from training are present (in case some categories are missing)
        # The model expects these 15 features after one-hot encoding
        expected_columns = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_M', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA',
            'RestingECG_ST', 'RestingECG_LVH', 'ExerciseAngina_Y',
            'ST_Slope_Flat', 'ST_Slope_Up'
        ]
        
        # Add missing columns with 0s
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[expected_columns]
        
        # Convert to numpy array
        features = input_encoded.values
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Prepare result
        if prediction == 1:
            result = "⚠️ Risk of Heart Attack Detected"
            result_class = "risk"
        else:
            result = "✓ No Risk of Heart Attack"
            result_class = "no-risk"
        
        risk_percentage = f"{probability * 100:.2f}%"
        
        return jsonify({
            'result': result,
            'result_class': result_class,
            'risk_percentage': risk_percentage,
            'confidence': f"{max(probability, 1-probability) * 100:.2f}%"
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error making prediction: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)
