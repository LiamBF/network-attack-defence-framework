import requests
import pandas as pd
import os
import random
import time

# _____1. CONFIGURATION_____
API_URL = "http://127.0.0.1:5000/predict"
OUTPUT_FILE = 'successful_attacks.csv'
DATA_FILE = 'CICIDS2017_cleaned\cicids_NormalBotIntrusion_subset.csv' 

def query_api(data_dict):
    try:
        clean_data = {k: v for k, v in data_dict.items() if k not in ['category', 'label_encoded', 'category_encoded']}
        response = requests.post(API_URL, json=clean_data, timeout=5)
        return response.json().get('prediction')
    except Exception:
        return "Connection_Error"

# _____2. DATA LOADING_____
if not os.path.exists(DATA_FILE):
    print(f"ERROR: Could not find {DATA_FILE}")
    exit()

df = pd.read_csv(DATA_FILE)

# _____3. USER PARAMETERS & INTENSITY SELECTION_____
print("="*65)
print("   CYBER-SENTINEL: ADVERSARIAL FUZZER   ")
print("="*65)

intensities = {
    "1": {"name": "Subtle", "range": (0.8, 1.2)},
    "2": {"name": "Realistic", "range": (0.5, 2.5)},
    "3": {"name": "Stress Test", "range": (0.1, 10.0)}
}

print("[?] Select Attack Intensity:")
print(" [1] Subtle (Small changes, hardest to detect)")
print(" [2] Realistic (Standard network variations)")
print(" [3] Stress Test (Drastic changes/Extreme outliers)")

choice = input("\nSelection [1-3]: ")
selected_mode = intensities.get(choice, intensities["2"])
min_mult, max_mult = selected_mode["range"]

try:
    MAX_ATTEMPTS = int(input("[?] How many fuzzing attempts would you like to run? "))
except ValueError:
    MAX_ATTEMPTS = 1000

print(f"\n[*] Mode: {selected_mode['name']} | Range: {min_mult}x to {max_mult}x")
print(f"[*] Starting {MAX_ATTEMPTS} attempts...")

# _____4. THE MAIN LOOP_____
classes_to_test = ['Infiltration', 'Bot']
features_to_test = [
    'destination port', 'init_win_bytes_forward', 
    'active min', 'active max', 'bwd packet length min',
    'flow duration', 'total fwd packets', 'subflow bwd bytes'
]

stats = {
    'Infiltration': {'attempts': 0, 'success': 0, 'single': 0, 'double': 0, 'triple': 0, 'failed': 0},
    'Bot': {'attempts': 0, 'success': 0, 'single': 0, 'double': 0, 'triple': 0, 'failed': 0}
}

start_time = time.time()
found_total = 0

for i in range(1, MAX_ATTEMPTS + 1):
    current_class = random.choice(classes_to_test)
    class_df = df[df['category'] == current_class]
    
    if class_df.empty: continue
        
    sample_row = class_df.sample(n=1).iloc[0].to_dict()
    modified = sample_row.copy()
    stats[current_class]['attempts'] += 1
    
    # Triple-Tap Logic
    num_to_tweak = random.randint(1, 3)
    selected_feats = random.sample(features_to_test, num_to_tweak)
    for feat in selected_feats:
        multiplier = random.uniform(min_mult, max_mult)
        modified[feat] = sample_row[feat] * multiplier

    res = query_api(modified)
    
    if res == "Benign":
        stats[current_class]['success'] += 1
        found_total += 1
        if num_to_tweak == 1: stats[current_class]['single'] += 1
        elif num_to_tweak == 2: stats[current_class]['double'] += 1
        else: stats[current_class]['triple'] += 1
            
        modified['category'] = current_class
        modified['fuzz_mode'] = selected_mode['name'] # Record intensity in CSV
        adv_df = pd.DataFrame([modified])
        file_exists = os.path.isfile(OUTPUT_FILE)
        adv_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
        
        print(f"[!] SUCCESS ({selected_mode['name']}): {current_class} bypassed via {num_to_tweak}-Tap!")
    else:
        stats[current_class]['failed'] += 1

    step = max(1, MAX_ATTEMPTS // 10)
    if i % step == 0:
        percent = (i / MAX_ATTEMPTS) * 100
        print(f"--- PROGRESS: {i}/{MAX_ATTEMPTS} ({percent:.0f}%) | Bypasses: {found_total} ---")

#_____5. FINAL SUMMARY REPORT_____
duration = time.time() - start_time
print("\n" + "="*65)
print(f"   STRESS TEST: {selected_mode['name'].upper()} FINAL REPORT   ")
print("="*65)
print(f"Total Time: {duration:.2f}s | Intensity: {selected_mode['name']}")

for cls in ['Infiltration', 'Bot']:
    s = stats[cls]
    if s['attempts'] > 0:
        rate = (s['success'] / s['attempts'] * 100)
        print(f"\nCLASS: {cls}")
        print(f"  > Attempts: {s['attempts']}")
        print(f"  > Success (Bypass): {s['success']} (Rate: {rate:.2f}%)")

print(f"\n[+] Results saved to: {os.path.abspath(OUTPUT_FILE)}")
print("="*65)