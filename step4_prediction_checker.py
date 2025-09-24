import pandas as pd
import joblib
import warnings

# --- Setup ---
warnings.filterwarnings('ignore')
print("--- Advanced AI Model Checker for Low, Medium, and High Risk ---")

# --- 1. Load the Saved Model and Helper Files ---
try:
    model = joblib.load('esg_risk_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ Model and helper files loaded successfully.\n")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run 'step2_and_3_train_and_tune.py' first.")
    exit()

# --- 2. Define Sample Data for Each Risk Category ---

# Designed to be clearly LOW risk
low_risk_supplier = {
    'name': 'Green Threads USA', 'country': 'USA', 'industryVertical': 'Raw Material Farming',
    'water_usage_m3': 20000, 'turnover_rate_percent': 5, 'workplace_accidents_last_year': 0,
    'has_anti_corruption_policy': True, 'publishes_esg_report': True,
    'total_emissions_kg_co2e': 30000, 'processing_type': 'Farming',
    'number_of_workers': '51-200', 'sector': 'Apparel',
    'is_iso14001_certified': True, 'is_sa8000_certified': True
}

# Designed to have a mix of good and bad factors, resulting in MEDIUM risk
medium_risk_supplier = {
    'name': 'Ankara Weaving Mill', 'country': 'Turkey', 'industryVertical': 'Weaving & Knitting',
    'water_usage_m3': 90000, 'turnover_rate_percent': 18, 'workplace_accidents_last_year': 4,
    'has_anti_corruption_policy': True, 'publishes_esg_report': False,
    'total_emissions_kg_co2e': 115000, 'processing_type': 'Weaving',
    'number_of_workers': '501-1000', 'sector': 'Apparel',
    'is_iso14001_certified': True, 'is_sa8000_certified': False
}

# Designed to have multiple high-risk factors
high_risk_supplier = {
    'name': 'Dhaka Dye Works', 'country': 'Bangladesh', 'industryVertical': 'Dyeing & Finishing',
    'water_usage_m3': 200000, 'turnover_rate_percent': 30, 'workplace_accidents_last_year': 10,
    'has_anti_corruption_policy': False, 'publishes_esg_report': False,
    'total_emissions_kg_co2e': 350000, 'processing_type': 'Dyeing|Finishing',
    'number_of_workers': '1001-5000', 'sector': 'Apparel',
    'is_iso14001_certified': False, 'is_sa8000_certified': False
}

test_cases = [low_risk_supplier, medium_risk_supplier, high_risk_supplier]

# --- 3. Recreate the Feature Engineering Pipeline ---
# This function MUST be identical to the one in your training and API scripts.
def create_features(df):
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

# --- 4. Process and Predict for Each Case ---
for supplier_data in test_cases:
    print(f"\n--- Testing Supplier: {supplier_data['name']} ---")
    
    # Convert dict to DataFrame
    new_df = pd.DataFrame([supplier_data])
    
    # Apply feature engineering
    featured_df = create_features(new_df)
    
    # One-hot encode categorical columns
    categorical_cols = ['country', 'industryVertical', 'processing_type', 'sector', 'industry_description']
    encoded_df = pd.get_dummies(featured_df, columns=categorical_cols, drop_first=True)
    
    # Align columns to match the model's training data
    final_df = encoded_df.reindex(columns=model_columns, fill_value=0)
    
    # Make prediction
    prediction_encoded = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Display results
    print(f"✅ PREDICTION:   {prediction_label.upper()}")
    
    confidence_scores = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    print("Confidence Scores:")
    for risk_class in confidence_scores.columns:
        score = confidence_scores[risk_class].iloc[0]
        print(f"  - {risk_class}: {score:.2%}")