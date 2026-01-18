import os
import joblib
import numpy as np
import pandas as pd
from utils import (
    preprocess_menu, clean_karbo_list,
    compute_mccbf_scores
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset/dataset450_indonesia.csv"
RF_DIR = "rftrain"
HYBRID_DIR = "hybrid"

SPLIT_NAME = "70_30"   # ganti sesuai kebutuhan
ALPHA = 0.7
MODE = "seimbang"

BASE_RF = os.path.join(RF_DIR, SPLIT_NAME)
BASE_OUT = os.path.join(HYBRID_DIR, SPLIT_NAME)
os.makedirs(BASE_OUT, exist_ok=True)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

print("="*70)
print(f"🔍 HYBRID RF ERROR ANALYSIS - SPLIT {SPLIT_NAME}")
print("="*70)

df = pd.read_csv(DATASET_PATH)

rf_model = joblib.load(os.path.join(BASE_RF, "rf_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_RF, "tfidf_vectorizer.pkl"))

indices_df = pd.read_csv(os.path.join(BASE_RF, "split_indices.csv"))
test_indices = indices_df['test_indices'].dropna().astype(int).tolist()

# ============================================================
# PREPARE DATA
# ============================================================

df["Karbo_List"] = df["Karbo_List"].apply(clean_karbo_list)
df["enhanced_corpus"] = df["Nama_Menu"].apply(preprocess_menu)

test_df = df.iloc[test_indices].copy()

X_all = vectorizer.transform(df["enhanced_corpus"])
rf_proba = rf_model.predict_proba(X_all)
rf_scores_all = rf_proba.max(axis=1)

# ============================================================
# HYBRID ERROR ANALYSIS
# ============================================================

error_rows = []

print(f"📊 Total test data: {len(test_df)}")

for _, row in test_df.iterrows():
    query = row["Nama_Menu"].lower()
    true_label = row["Kategori"]

    # MCCBF
    mccbf_scores, _ = compute_mccbf_scores(
        df, X_all, vectorizer,
        user_query=query,
        target_calorie=None,
        preferred_karbo=None,
        mode=MODE
    )

    # Hybrid score
    hybrid_scores = ALPHA * mccbf_scores + (1 - ALPHA) * rf_scores_all
    best_idx = hybrid_scores.argmax()
    pred_label = df.iloc[best_idx]["Kategori"]

    # Simpan hanya yang SALAH
    if pred_label != true_label:
        error_rows.append({
            "Query_Menu": row["Nama_Menu"],
            "True_Label": true_label,
            "Pred_Label": pred_label,
            "Hybrid_Score": hybrid_scores[best_idx],
            "MCCBF_Score": mccbf_scores[best_idx],
            "RF_Score": rf_scores_all[best_idx]
        })

error_df = pd.DataFrame(error_rows)

# ============================================================
# OUTPUT
# ============================================================

print(f"❌ Total hybrid misclassification: {len(error_df)}")

if not error_df.empty:
    print("\n📋 Contoh kesalahan hybrid:")
    print(error_df.head())

    output_path = os.path.join(BASE_OUT, "hybrid_rf_error_analysis.csv")
    error_df.to_csv(output_path, index=False)

    print(f"\n💾 Saved error analysis to:")
    print(f"   - {output_path}")
else:
    print("🎉 Tidak ditemukan kesalahan hybrid!")

print("="*70)
print("✅ HYBRID RF ERROR ANALYSIS COMPLETED")
