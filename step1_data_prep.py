import pandas as pd
import io

print("--- Step 1 (Final & Expanded): Data Consolidation & Preparation ---")

# This dataset has been expanded to 30 suppliers and is carefully curated
# to ensure a robust and balanced distribution of risk profiles.
suppliers_data = """supplierId,name,country,industryVertical,water_usage_m3,turnover_rate_percent,workplace_accidents_last_year,has_anti_corruption_policy,publishes_esg_report
sup_001,Apex Garments,Bangladesh,Garment Manufacturing,50000,15,2,True,True
sup_002,Rainbow Dyers,India,Dyeing & Finishing,120000,22,5,False,False
sup_003,Tiruppur Spinning Mills,India,Spinning Mill,75000,10,1,True,False
sup_004,Organic Cotton Collective,USA,Raw Material Farming,30000,5,0,True,True
sup_005,Vietnam Weavers,Vietnam,Weaving & Knitting,60000,18,3,False,True
sup_006,Global Weaving Co.,Turkey,Weaving & Knitting,80000,12,2,True,True
sup_007,Eco-Friendly Packaging,India,Packaging,10000,8,0,True,True
sup_008,Risky Dyers Pakistan,Pakistan,Dyeing & Finishing,250000,35,12,False,False
sup_009,China Silk Manufacturing,China,Weaving & Knitting,110000,25,6,False,False
sup_010,USA Apparel Co,USA,Garment Manufacturing,20000,7,1,True,True
sup_011,Turkish Denim Mill,Turkey,Spinning Mill,90000,14,3,True,False
sup_012,Dhaka Finishing Plant,Bangladesh,Dyeing & Finishing,180000,28,8,False,False
sup_013,Saigon Logistics,Vietnam,Logistics,5000,10,1,True,False
sup_014,Mumbai Prints,India,Printing,45000,16,4,False,True
sup_015,Karachi Cotton Exporters,Pakistan,Raw Material Farming,60000,30,7,False,False
sup_016,4 Star Textiles,India,Garment Manufacturing,60000,18,3,True,False
sup_017,Aadhavan Textiles Printing,India,Printing,90000,15,4,False,True
sup_018,Santana Textiles,Brazil,Weaving & Knitting,85000,11,2,True,True
sup_019,Zouping Taizi Hongfu Home Textiles Factory,China,Weaving & Knitting,95000,28,6,False,False
sup_020,4G TEXTILES SARL,Morocco,Garment Manufacturing,40000,21,5,False,False
sup_021,Clean Earth Mills,USA,Spinning Mill,40000,6,0,True,True
sup_022,Vietnam Finishing Touch,Vietnam,Dyeing & Finishing,95000,20,4,False,False
sup_023,Brazil Cotton Hub,Brazil,Raw Material Farming,70000,15,2,True,False
sup_024,Morocco Leather Goods,Morocco,Manufacturing,30000,19,6,False,False
sup_025,Istanbul Garments,Turkey,Garment Manufacturing,45000,13,1,True,True
sup_026,United Tex,Bangladesh,Garment Manufacturing,150000,33,9,False,False
sup_027,Premium Packaging Solutions,USA,Packaging,5000,4,0,True,True
sup_028,Surat Weaving Mills,India,Weaving & Knitting,88000,17,3,True,False
sup_029,Nantong High-Tech Textiles,China,Spinning Mill,130000,26,7,False,False
sup_030,Lahore Tannery,Pakistan,Manufacturing,220000,38,15,False,False
"""

activity_data = """supplierId,dataType,value,unit
sup_001,Electricity,150000,kWh
sup_002,Natural Gas,40000,m³
sup_003,Diesel Fuel,10000,Liters
sup_004,Electricity,50000,kWh
sup_005,Electricity,90000,kWh
sup_006,Electricity,110000,kWh
sup_007,Electricity,25000,kWh
sup_008,Natural Gas,180000,m³
sup_009,Electricity,200000,kWh
sup_010,Electricity,60000,kWh
sup_011,Diesel Fuel,25000,Liters
sup_012,Natural Gas,150000,m³
sup_013,Diesel Fuel,30000,Liters
sup_014,Electricity,130000,kWh
sup_015,Diesel Fuel,15000,Liters
sup_016,Electricity,160000,kWh
sup_017,Electricity,140000,kWh
sup_018,Electricity,190000,kWh
sup_019,Natural Gas,60000,m³
sup_020,Diesel Fuel,18000,Liters
sup_021,Electricity,75000,kWh
sup_022,Natural Gas,70000,m³
sup_023,Diesel Fuel,20000,Liters
sup_024,Diesel Fuel,15000,Liters
sup_025,Electricity,85000,kWh
sup_026,Natural Gas,120000,m³
sup_027,Electricity,10000,kWh
sup_028,Electricity,125000,kWh
sup_029,Electricity,220000,kWh
sup_030,Natural Gas,160000,m³
"""

emission_factors_data = """source,unit,factor
Grid,kg CO2e/kWh,0.82
Natural Gas,kg CO2e/m³,2.02
Diesel Fuel,kg CO2e/Liter,2.68
"""

suppliers_df = pd.read_csv(io.StringIO(suppliers_data))
activity_df = pd.read_csv(io.StringIO(activity_data))
factors_df = pd.read_csv(io.StringIO(emission_factors_data))
activity_df['source'] = activity_df['dataType'].apply(lambda x: 'Grid' if 'Electric' in x else x)

print("Calculating emissions for internal suppliers...")
emissions_df = pd.merge(activity_df, factors_df, on='source', how='left')
emissions_df['total_emissions_kg_co2e'] = emissions_df['value'] * emissions_df['factor']
emissions_summary_df = emissions_df.groupby('supplierId')['total_emissions_kg_co2e'].sum().reset_index()
internal_master_df = pd.merge(suppliers_df, emissions_summary_df, on='supplierId', how='left').fillna(0)
print(f"Processed {len(internal_master_df)} internal supplier records.")

REAL_DATA_FILE = 'facilities-2.csv'
try:
    print(f"Loading real-world data from '{REAL_DATA_FILE}'...")
    real_df = pd.read_csv(REAL_DATA_FILE, low_memory=False, usecols=['name', 'country_name', 'number_of_workers', 'sector', 'processing_type', 'lat', 'lng'])
    real_df.rename(columns={'country_name': 'country'}, inplace=True)
    master_df = pd.merge(internal_master_df, real_df, on=['name', 'country'], how='left')
    master_df = master_df.drop_duplicates(subset=['supplierId']).reset_index(drop=True)
    print("Enrichment complete.")
except FileNotFoundError:
    print(f"Error: '{REAL_DATA_FILE}' not found. Using internal data only.")
    master_df = internal_master_df

OUTPUT_FILE = 'master_dataset.csv'
master_df.to_csv(OUTPUT_FILE, index=False)
print(f"Consolidated data saved to '{OUTPUT_FILE}'.")
print("\n--- Step 1 Complete ---")