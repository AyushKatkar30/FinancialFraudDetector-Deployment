import pickle
import xgboost as xgb

try:
    # Load XGBoost model from pickle file
    with open("XGBoostModel.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    # Save the model in JSON format
    xgb_model.save_model("XGBoostModel.json")

    print("✅ XGBoost model successfully converted to JSON format.")

except FileNotFoundError:
    print("❌ Error: XGBoostModel.pkl not found. Make sure it's in the correct directory.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
