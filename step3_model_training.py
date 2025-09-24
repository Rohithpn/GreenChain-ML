import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

warnings.filterwarnings('ignore')

print("--- Step 3 (Advanced): Model Training & Evaluation ---")

INPUT_FILE = 'featured_dataset.csv'
try:
    featured_df = pd.read_csv(INPUT_FILE)
    print(f"Loaded '{INPUT_FILE}' with {len(featured_df)} records.")
except FileNotFoundError:
    print(f"Error: '{INPUT_FILE}' not found. Please run Step 2 first.")
    exit()

print("Preparing data for the model...")
categorical_cols = ['country', 'industryVertical', 'processing_type', 'sector', 'industry_description']
model_df = pd.get_dummies(featured_df, columns=categorical_cols, drop_first=True)

target = 'risk_level'
y = model_df[target]
X = model_df.select_dtypes(include=['number', 'bool'])
if target in X.columns:
    X = X.drop(columns=[target])
X = X.drop(columns=[col for col in ['lat', 'lng'] if col in X.columns], errors='ignore')
joblib.dump(X.columns, 'model_columns.pkl')

print(f"Splitting data into training ({int((1-0.25)*100)}%) and testing ({int(0.25*100)}%) sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

le = LabelEncoder()
all_possible_classes = ['Low', 'Medium', 'High']
le.fit(all_possible_classes)
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)
joblib.dump(le, 'label_encoder.pkl')
num_classes = len(le.classes_)

print("\n--- Training the XGBoost Classifier ---")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=num_classes, n_estimators=150,
    learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42
)
xgb_model.fit(X_train, y_train_encoded)
print("Model training complete.")

print("\n--- Evaluating Model on Unseen Test Data ---")
y_pred_encoded = xgb_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)

print(f"\nModel Accuracy on Test Data: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=['Low', 'Medium', 'High'], zero_division=0))

print("\n--- Saving the final trained model ---")
joblib.dump(xgb_model, 'esg_risk_model.pkl')
print("Model successfully saved to 'esg_risk_model.pkl'.")
print("\n--- Step 3 Complete ---")