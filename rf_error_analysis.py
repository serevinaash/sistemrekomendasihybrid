import os
import pandas as pd
import joblib
from utils import preprocess_menu

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset/dataset450_indonesia.csv"
RF_DIR = "rftrain"

# Pilih split yang mau dianalisis
SPLIT_NAME = "70_30"   # ganti: 70_30 / 80_20 / 90_10

BASE_PATH = os.path.join(RF_DIR, SPLIT_NAME)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

print("="*70)
print(f"🔍 RF MISCLASSIFICATION ANALYSIS - SPLIT {SPLIT_NAME}")
print("="*70)

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded: {len(df)} rows")

# Load model & vectorizer
rf_model = joblib.load(os.path.join(BASE_PATH, "rf_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_PATH, "tfidf_vectorizer.pkl"))
print("✅ Model & vectorizer loaded")

# Load split indices
split_df = pd.read_csv(os.path.join(BASE_PATH, "split_indices.csv"))

# ============================================================
# REBUILD TEST SET
# ============================================================

test_indices = split_df['test_indices'].dropna().astype(int).values
test_df = df.iloc[test_indices].copy()

test_df['enhanced_corpus'] = test_df['Nama_Menu'].apply(preprocess_menu)

X_test_vec = vectorizer.transform(test_df['enhanced_corpus'])
y_true = test_df['Kategori']

print(f"📊 Total test data: {len(test_df)}")

# ============================================================
# PREDICTION & ERROR ANALYSIS
# ============================================================

y_pred = rf_model.predict(X_test_vec)

test_df['Prediksi'] = y_pred
test_df['Benar'] = test_df['Kategori'] == test_df['Prediksi']

wrong_df = test_df[test_df['Benar'] == False]
correct_df = test_df[test_df['Benar'] == True]

# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n📈 HASIL PREDIKSI")
print("-"*70)
print(f"✅ Benar prediksi : {len(correct_df)}")
print(f"❌ Salah prediksi : {len(wrong_df)}")
print(f"📉 Error rate     : {len(wrong_df)/len(test_df)*100:.2f}%")

print("\n📊 POLA KESALAHAN (Aktual → Prediksi)")
print("-"*70)
print(pd.crosstab(
    wrong_df['Kategori'],
    wrong_df['Prediksi']
))

# ============================================================
# SAVE RESULTS
# ============================================================

output_wrong = os.path.join(BASE_PATH, "misclassified_data.csv")
output_correct = os.path.join(BASE_PATH, "correctly_classified_data.csv")

wrong_df.to_csv(output_wrong, index=False)
correct_df.to_csv(output_correct, index=False)

print("\n💾 FILE TERSIMPAN")
print(f"   - {output_wrong}")
print(f"   - {output_correct}")

print("\n🎉 ANALISIS KESALAHAN SELESAI")
print("="*70)
