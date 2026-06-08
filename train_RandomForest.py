import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib 
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#_____1. SETUP_____
TRAIN_FILE = 'CICIDS2017_cleaned\cicids_NormalBotIntrusion_subset.csv'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#_____2. THE MENU_____
scale_label = "FULL"
pct_str = "100pct"
train_pct_str = "80Train"

#_____3. UPDATED FOLDER MANAGEMENT_____
folder_name = 'model_RandomForest'
output_path = os.path.join(SCRIPT_DIR, folder_name)

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created new folder: {output_path}")

#_____4. LOADING DATA_____
print(f"\n[1/5] Loading {folder_name} Data...")
try:
    df = pd.read_csv(TRAIN_FILE)
    print(f"      Ready with {len(df):,} rows.")
except Exception as e:
    print(f"Error: {e}")
    exit()

#_____5. DATA PREP_____
y = df['category_encoded']
leakage_list = ['category', 'category_encoded', 'label_encoded', 'unnamed: 0', 'id']
X = df.drop(columns=[c for c in leakage_list if c in df.columns], axis=1)
X = X.select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

#_____6. TRAINING_____
print(f"[2/5] Training Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

#_____7. EVALUATION_____
print(f"[3/5] Analyzing Performance...")
y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
target_names = LabelEncoder().fit(df['category']).classes_
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

#_____8. SAVE TEXT REPORT_____
report_filename = f"Metrics_{folder_name}.txt"
with open(os.path.join(output_path, report_filename), "w") as f:
    f.write(f"CICIDS-2017 NBI SUBSET REPORT\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write(f"Balanced Acc: {bal_acc:.5f}\n")
    f.write(report)

#_____9. SAVE VISUALS_____
# Confusion Matrix
print(f"[4/5] Generating Plots...")
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis', 
            xticklabels=target_names, yticklabels=target_names)
plt.savefig(os.path.join(output_path, f"CM_{folder_name}.png"))
plt.close()
# Feature Importance
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = (importances.argsort()[::-1])[:10]
plt.barh([X.columns[i] for i in indices], importances[indices], color='teal')
plt.title(f'Top 10 Features ({pct_str} {scale_label} {train_pct_str})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"Features_{pct_str}_{scale_label}_{train_pct_str}.png"))
plt.close()

#_____10. EXPORT JOBLIB FILES TO FOLDER_____
print(f"[5/5] Exporting Files for Victim API to {folder_name}...")
joblib.dump(rf_model, os.path.join(output_path, 'cicids_rf_model.joblib'))
joblib.dump(scaler, os.path.join(output_path, 'cicids_scaler.joblib'))
joblib.dump(target_names, os.path.join(output_path, 'cicids_labels.joblib'))

print("\n" + "!"*60)
print(f"      EXPERIMENT COMPLETE: {folder_name}")
print("!"*60)

#_____11. TERMINAL OUTPUT_____
print("\n" + "!"*60)
print(f"      EXPERIMENT COMPLETE & FILES SAVED")
print("!"*60)
print(f"Folder Created:   {folder_name}")
print(f"Joblib Files:     Saved in {SCRIPT_DIR}")
print(f"Balanced Accuracy: {bal_acc:.5f}")
print("="*60)