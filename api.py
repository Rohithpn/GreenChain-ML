from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import warnings

# --- Setup ---
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load Trained Model and Helper Files ---
try:
    model = joblib.load('esg_risk_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ Tuned model and helper files loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run the training script first.")
    model = None

# --- Recreate the Feature Engineering Pipeline ---
def create_features(df):
    # This function contains all the logic from your step2 script
    GEOPOLITICAL_RISK = {'India': 3, 'China': 4, 'Vietnam': 2, 'Bangladesh': 4, 'USA': 1, 'Turkey': 3, 'Pakistan': 5, 'Brazil': 3, 'Morocco': 3}
    df['geopolitical_risk'] = df['country'].map(GEOPOLITICAL_RISK).fillna(3)

    df['industry_description'] = df.apply(
        lambda row: row.get('processing_type') if pd.notna(row.get('processing_type')) else row.get('industryVertical'),
        axis=1
    )
    INDUSTRY_RISK = {'Dyeing': 5, 'Printing': 4, 'Finishing': 5, 'Spinning': 4, 'Weaving': 3, 'Manufacturing': 3, 'Logistics': 1, 'Packaging': 1, 'Unspecified': 2}
    def map_industry_risk(desc):
        for risk_word, score in INDUSTRY_RISK.items():
            if risk_word.lower() in str(desc).lower(): return score
        return 2
    df['industry_risk'] = df['industry_description'].apply(map_industry_risk)

    def parse_workers(worker_str):
        worker_str = str(worker_str)
        if '5001+' in worker_str: return 7500
        if '-' in worker_str:
            try:
                low, high = map(int, worker_str.split('-'))
                return (low + high) / 2
            except: return 0
        return 0
    df['worker_count_avg'] = df['number_of_workers'].apply(parse_workers)
    df['emission_intensity'] = df.apply(
        lambda row: row['total_emissions_kg_co2e'] / row['worker_count_avg'] if row['worker_count_avg'] > 0 else 0,
        axis=1
    )
    
    df['is_iso14001_certified'] = df['is_iso14001_certified'].astype(bool)
    df['is_sa8000_certified'] = df['is_sa8000_certified'].astype(bool)
    df['has_anti_corruption_policy'] = df['has_anti_corruption_policy'].astype(bool)
    df['publishes_esg_report'] = df['publishes_esg_report'].astype(bool)
    
    return df

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not loaded.'}), 500

    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'No input data provided.'}), 400

    new_df = pd.DataFrame([json_data])
    featured_df = create_features(new_df)
    
    categorical_cols = ['country', 'industryVertical', 'processing_type', 'sector', 'industry_description']
    encoded_df = pd.get_dummies(featured_df, columns=categorical_cols, drop_first=True)
    
    # FIX: Select only the numeric/boolean columns to ensure the format perfectly matches the training data.
    # This removes original text columns like 'number_of_workers' before prediction.
    numeric_encoded_df = encoded_df.select_dtypes(include=['number', 'bool'])

    # Align columns with the model's training data
    final_df = numeric_encoded_df.reindex(columns=model_columns, fill_value=0)

    prediction_encoded = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    confidence_scores = {
        label: float(prob) for label, prob in zip(label_encoder.classes_, prediction_proba[0])
    }
    
    response = {
        'prediction': prediction_label,
        'confidenceScores': confidence_scores 
    }

    return jsonify(response)

# --- Run Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
