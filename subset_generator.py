import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#_____1. CONFIGURATION_____
OUTPUT_DIR = 'CICIDS2017_cleaned'
BOT_FILE = os.path.join(OUTPUT_DIR, 'cleaned_cicids_friday_morning.csv')
INF_FILE = os.path.join(OUTPUT_DIR, 'cleaned_cicids_thursday_inf.csv')
OUTPUT_FILE = 'cicids_NormalBotIntrusion_subset.csv'
CHART_FILE = 'nbi_subset_distribution.png'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[*] Created directory: {OUTPUT_DIR}")

print("[*] Loading specific day files...")

# Verify files exist
if not os.path.exists(BOT_FILE) or not os.path.exists(INF_FILE):
    print(f"ERROR: Ensure files exist in {OUTPUT_DIR}")
    exit()

# Load and combine
df_day1 = pd.read_csv(BOT_FILE)
df_day2 = pd.read_csv(INF_FILE)
df = pd.concat([df_day1, df_day2], ignore_index=True)

df.columns = df.columns.str.strip().str.lower()

# _____2. FILTERING & SAMPLING _____
print("[*] Filtering and balancing dataset...")

df_bot = df[df['category'].str.lower() == 'bot']
df_inf = df[df['category'].str.lower() == 'infiltration']
df_benign_full = df[df['category'].str.lower() == 'benign']

if len(df_benign_full) > 2000:
    df_benign = df_benign_full.sample(n=2000, random_state=42)
else:
    df_benign = df_benign_full

df_subset = pd.concat([df_benign, df_bot, df_inf], ignore_index=True)

# _____3. RE-ENCODING & VISUALIZATION _____
le = LabelEncoder()
df_subset['category_encoded'] = le.fit_transform(df_subset['category'])

print("[*] Generating distribution chart...")
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.countplot(data=df_subset, x='category', hue='category', 
                   palette='viridis', legend=False,
                   order=['Benign', 'Bot', 'Infiltration'])

plt.title('Class Distribution: NBI Subset', fontsize=15)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, CHART_FILE))
print(f"[+] Chart saved to {os.path.join(OUTPUT_DIR, CHART_FILE)}")

#_____4. SAVE_____
final_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df_subset.to_csv(final_path, index=False)

print("\n" + "="*30)
print("SUBSET GENERATION COMPLETE")
print("="*30)
print(f"File Saved: {final_path}")
print(f"Encodings:  {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(df_subset['category'].value_counts())