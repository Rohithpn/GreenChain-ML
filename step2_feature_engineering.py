import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')
random.seed(42)

print("--- Step 2 (Advanced): Feature Engineering ---")

INPUT_FILE = 'master_dataset.csv'
try:
    master_df = pd.read_csv(INPUT_FILE)
    print(f"Loaded '{INPUT_FILE}' with {len(master_df)} records.")
except FileNotFoundError:
    print(f"Error: '{INPUT_FILE}' not found. Please run Step 1 first.")
    exit()

master_df['number_of_workers'].fillna('0', inplace=True)
master_df['processing_type'].fillna('Unspecified', inplace=True)

print("Creating advanced ESG features...")

# E - Environmental Features
GEOPOLITICAL_RISK = {'India': 3, 'China': 4, 'Vietnam': 2, 'Bangladesh': 4, 'USA': 1, 'Turkey': 3, 'Pakistan': 5, 'Brazil': 3, 'Morocco': 3}
master_df['geopolitical_risk'] = master_df['country'].map(GEOPOLITICAL_RISK).fillna(3)

master_df['industry_description'] = master_df.apply(
    lambda row: row['processing_type'] if pd.notna(row['processing_type']) and row['processing_type'] != 'Unspecified' else row['industryVertical'],
    axis=1
)
INDUSTRY_RISK = {'Dyeing': 5, 'Printing': 4, 'Finishing': 5, 'Spinning': 4, 'Weaving': 3, 'Manufacturing': 3, 'Logistics': 1, 'Packaging': 1, 'Unspecified': 2}
def map_industry_risk(desc):
    for risk_word, score in INDUSTRY_RISK.items():
        if risk_word.lower() in str(desc).lower(): return score
    return 2
master_df['industry_risk'] = master_df['industry_description'].apply(map_industry_risk)

def parse_workers(worker_str):
    worker_str = str(worker_str)
    if '5001+' in worker_str: return 7500
    if '-' in worker_str:
        try:
            low, high = map(int, worker_str.split('-'))
            return (low + high) / 2
        except: return 0
    return 0
master_df['worker_count_avg'] = master_df['number_of_workers'].apply(parse_workers)
master_df['emission_intensity'] = master_df.apply(
    lambda row: row['total_emissions_kg_co2e'] / row['worker_count_avg'] if row['worker_count_avg'] > 0 else 0,
    axis=1
)

# S & G - Social & Governance Features
master_df['is_iso14001_certified'] = [random.choice([True, False]) for _ in range(len(master_df))]
master_df['is_sa8000_certified'] = [random.choice([True, False]) for _ in range(len(master_df))]

def assign_risk_level(row):
    score = 0
    # E Score
    score += row['geopolitical_risk'] + row['industry_risk']
    if row['total_emissions_kg_co2e'] > 80000: score += 2
    if row['water_usage_m3'] > 100000: score += 1
    if row['is_iso14001_certified']: score -= 2
    # S Score
    if row['turnover_rate_percent'] > 20: score += 1
    if row['workplace_accidents_last_year'] > 5: score += 2
    if row['is_sa8000_certified']: score -= 3
    # G Score
    if not row['has_anti_corruption_policy']: score += 2
    if not row['publishes_esg_report']: score += 1
    # Classification
    if score >= 9: return 'High'
    if score >= 5: return 'Medium'
    return 'Low'
master_df['risk_level'] = master_df.apply(assign_risk_level, axis=1)
print("Feature engineering complete.")

OUTPUT_FILE = 'featured_dataset.csv'
master_df.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset with engineered features saved to '{OUTPUT_FILE}'.")
print("\nFinal Risk Level Distribution:")
print(master_df['risk_level'].value_counts())
print("\n--- Step 2 Complete ---")