from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

#_____1. DYNAMIC MODEL SELECTION_____
def select_model():
    folders = [f for f in os.listdir('.') if os.path.isdir(f) and (f.startswith('100pct') or f.startswith('model'))]
    
    if not folders:
        print("[-] No model folders found! Ensure your training scripts have run.")
        exit()

    print("\n=== CYBER-SENTINEL: AVAILABLE DEFENDERS ===")
    for i, folder in enumerate(folders):
        print(f"[{i}] {folder}")
    
    try:
        choice = int(input("\nSelect model index to load: "))
        selected_folder = folders[choice]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        exit()
        
    return selected_folder

#_____2. LOADING THE FULL DEFENSIVE SUITE_____
SELECTED_FOLDER = select_model()

# Search for the .joblib files inside the chosen folder
files = os.listdir(SELECTED_FOLDER)
model_file = next((f for f in files if 'model.joblib' in f), None)
scaler_file = 'cicids_scaler.joblib'
label_file = 'cicids_labels.joblib'

if model_file and scaler_file in files and label_file in files:
    print(f"[*] Loading assets from {SELECTED_FOLDER}...")
    model = joblib.load(os.path.join(SELECTED_FOLDER, model_file))
    scaler = joblib.load(os.path.join(SELECTED_FOLDER, scaler_file))
    target_names = joblib.load(os.path.join(SELECTED_FOLDER, label_file))
    print(f"[+] {model_file} is ONLINE. System is protected.")
else:
    print(f"ERROR: Missing files in {SELECTED_FOLDER}. Need model, scaler, and labels.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        query_df = pd.DataFrame([data])
        
        # 1. APPLY SCALING
        query_scaled = scaler.transform(query_df)
        
        # 2. PREDICT
        prediction_encoded = model.predict(query_scaled)[0]
        
        # 3. MAP TO NAMES
        prediction_label = target_names[prediction_encoded]
        
        if prediction_label == "Benign":
            print(f"[-] ALERT: Possible Bypass! Prediction: {prediction_label}")
        else:
            print(f"[+] BLOCK: Detected {prediction_label}")

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("="*50)
    print(f"   API RUNNING: {SELECTED_FOLDER}   ")
    print("="*50)
    app.run(port=5000, debug=False)