from flask import Flask, request, render_template
import numpy as np
import pickle
import xgboost as xgb
from DataPreProcessing import sc  # Ensure scaler is properly saved & loaded

app = Flask(__name__)

# Load Models
models = {
    'logisticModel': pickle.load(open('logisticModel.pkl', 'rb')),
    'decisionModel': pickle.load(open('decisionModel.pkl', 'rb')),
    'randomForestmodel': pickle.load(open('randomForestmodel.pkl', 'rb')),
}

# Load XGBoost model from JSON format (correct way)
xg_model = xgb.XGBClassifier()
xg_model.load_model("XGBoostModel.json")
models['xgBoostModel'] = xg_model

# Load the trained scaler
with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form inputs
    amount = float(request.form['amount'])
    old_balance = float(request.form['old_balance'])
    new_balance = float(request.form['new_balance'])
    transaction_category = int(request.form['transaction_category'])

    # Prepare input features for prediction
    input_features = np.array([[transaction_category, amount, old_balance, new_balance]])

    # Scale the input features
    input_features_scaled = sc.transform(input_features)

    # Perform predictions for the selected model
    selected_model_name = request.form['type']
    selected_model = models[selected_model_name]

    if hasattr(selected_model, "predict_proba"):  # For logistic regression and XGBoost
        prediction = selected_model.predict_proba(input_features_scaled)[0][1]
    else:  # For DecisionTree and RandomForest
        prediction = selected_model.predict(input_features_scaled)[0]

    # Determine the result based on prediction
    result = f'{selected_model_name}: Fraud Detected' if prediction >= 0.5 else f'{selected_model_name}: No Fraud'

    return render_template('index.html', pred=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

