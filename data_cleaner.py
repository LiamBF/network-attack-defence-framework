import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

OUTPUT_DIR = 'CICIDS2017_cleaned'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[*] Created directory: {OUTPUT_DIR}")

attack_map = {
    'BENIGN': 'Benign',
    'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS', 'DDoS': 'DoS', 'Heartbleed': 'DoS',
    'PortScan': 'Probe',
    'Web Attack  Brute Force': 'Web Attack', 'Web Attack  XSS': 'Web Attack', 'Web Attack  Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force',
    'Bot': 'Bot'
}

def clean_and_save(input_path, output_name):
    print(f"Processing: {input_path}...")
    
    df = pd.read_csv(input_path, low_memory=False)
    
    # 1. Clean column names
    df.columns = df.columns.str.strip()
    
    # 2. Handle Infinity and NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 3. Map labels to categories
    df['category'] = df['Label'].map(attack_map).fillna('Benign')
    df.drop(['Label'], axis=1, inplace=True)
    
    # 4. Separate features from labels for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 5. Scale Numeric Features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 6. Encode Categories
    label_le = LabelEncoder()
    df['label_encoded'] = label_le.fit_transform(df['category'])
    
    # 7. Save to folder
    final_save_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_csv(final_save_path, index=False)
    print(f"[+] Saved to {final_save_path}")

#_____EXECUTION_____
base_path = r'CICID2017\\'

files = {
    "Monday-WorkingHours.pcap_ISCX.csv": "cleaned_cicids_monday.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "cleaned_cicids_tuesday.csv",
    "Wednesday-workingHours.pcap_ISCX.csv": "cleaned_cicids_wednesday.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "cleaned_cicids_thursday_web.csv",
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv": "cleaned_cicids_thursday_inf.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "cleaned_cicids_friday_morning.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "cleaned_cicids_friday_port.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "cleaned_cicids_friday_ddos.csv"
}

for raw_file, clean_file in files.items():
    full_path = os.path.join(base_path, raw_file)
    if os.path.exists(full_path):
        clean_and_save(full_path, clean_file)
    else:
        print(f"Skipping: {raw_file} (Check file name or path!)")

print(f"\nAll CICIDS datasets processed and saved to {OUTPUT_DIR}!")