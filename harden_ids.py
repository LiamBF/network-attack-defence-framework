import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#_____1. PATH CONFIGURATION_____
ORIGINAL_DATA = 'CICIDS2017_cleaned\cicids_NormalBotIntrusion_subset.csv'
ADVERSARIAL_DATA = 'successful_attacks.csv' 

RF_DIR = 'model_RandomForest_hardened'
XGB_DIR = 'model_XGBoost_hardened'

os.makedirs(RF_DIR, exist_ok=True)
os.makedirs(XGB_DIR, exist_ok=True)

print("="*65)
print("   CYBER-SENTINEL: ADVERSARIAL HARDENING (RELIABLE)   ")
print("="*65)

#_____2. LOAD DATASETS_____
print("[*] Loading original and adversarial datasets...")
if not os.path.exists(ADVERSARIAL_DATA):
    print(f"ERROR: {ADVERSARIAL_DATA} not found. Run your fuzzer first!")
    exit()

df_orig = pd.read_csv(ORIGINAL_DATA)
df_adv = pd.read_csv(ADVERSARIAL_DATA)

#_____3. DATA PREP & AUGMENTATION_____
if 'fuzz_mode' in df_adv.columns:
    df_adv = df_adv.drop(columns=['fuzz_mode'])

df_hardened = pd.concat([df_orig, df_adv], ignore_index=True)

leakage = ['category', 'label_encoded', 'category_encoded', 'unnamed: 0']
X = df_hardened.drop(columns=[c for c in leakage if c in df_hardened.columns])
y = df_hardened['category_encoded']

target_names = df_hardened.sort_values('category_encoded')['category'].unique()

#_____4. SCALING_____
print("[*] Scaling the combined dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#_____5. THE SPLIT_____
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

#_____6. RETRAINING & SAVING_____

# MODEL A: RANDOM FOREST
print("\n[*] Training Hardened Random Forest...")
hardened_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
hardened_rf.fit(X_train, y_train)

# Save RF
joblib.dump(hardened_rf, os.path.join(RF_DIR, 'hardened_rf_model.joblib'))
joblib.dump(scaler, os.path.join(RF_DIR, 'cicids_scaler.joblib'))
joblib.dump(target_names, os.path.join(RF_DIR, 'cicids_labels.joblib'))

# MODEL B: XGBOOST
print("[*] Training Hardened XGBoost...")
hardened_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
hardened_xgb.fit(X_train, y_train)

# Save XGB
joblib.dump(hardened_xgb, os.path.join(XGB_DIR, 'hardened_xgb_model.joblib'))
joblib.dump(scaler, os.path.join(XGB_DIR, 'cicids_scaler.joblib'))
joblib.dump(target_names, os.path.join(XGB_DIR, 'cicids_labels.joblib'))

# _____7. EVALUATION_____
print("\n" + "="*35)
print("    PERFORMANCE REPORT   ")
print("="*35)
print(f"\n[RF] Testing on unseen data ({len(X_test)} samples):")
print(classification_report(y_test, hardened_rf.predict(X_test), target_names=target_names))

print(f"\n[XGB] Testing on unseen data ({len(X_test)} samples):")
print(classification_report(y_test, hardened_xgb.predict(X_test), target_names=target_names))

print("\n" + "="*65)
print(f"[SUCCESS] Retraining Complete.")
print(f"-> Folders ready for API: {RF_DIR}, {XGB_DIR}")
print("="*65)