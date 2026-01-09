import pandas as pd
import re
import ast

# ==============================
# 1. LOAD DATASET
# ==============================
INPUT_PATH = "dataset/dataset450_indonesia.csv"
OUTPUT_PATH = "dataset/dataset450_indonesia_preprocessed.csv"

df = pd.read_csv(INPUT_PATH)

# ==============================
# 2. HELPER FUNCTIONS
# ==============================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Kamus normalisasi karbohidrat
KARBO_MAP = {
    "beras merah": "nasi merah",
    "beras putih": "nasi putih",
    "nasi beras merah": "nasi merah",
    "nasi beras putih": "nasi putih"
}

VALID_KARBO = [
    "nasi merah",
    "nasi putih",
    "kentang",
    "ubi",
    "jagung"
]


def extract_karbo_list(sumber_karbo):
    sumber_karbo = clean_text(sumber_karbo)

    # normalisasi sinonim
    for k, v in KARBO_MAP.items():
        sumber_karbo = sumber_karbo.replace(k, v)

    karbo_list = []
    for karbo in VALID_KARBO:
        if karbo in sumber_karbo:
            karbo_list.append(karbo)

    return sorted(list(set(karbo_list)))


def karbo_to_clean(karbo_list):
    return ", ".join(karbo_list)


def build_corpus_rf(row):
    return " ".join([
        row["Nama_Menu"],
        row["Deskripsi_Menu"]
    ]).strip()


def build_corpus_mccbf(row):
    return " ".join([
        row["Nama_Menu"],
        row["Deskripsi_Menu"],
        row["Karbo_Clean"],
        row["Keyword_Semantik"]
    ]).strip()


# ==============================
# 3. CLEAN BASIC TEXT
# ==============================
df["Nama_Menu"] = df["Nama_Menu"].apply(clean_text)
df["Deskripsi_Menu"] = df["Deskripsi_Menu"].apply(clean_text)
df["Keyword_Semantik"] = df["Keyword_Semantik"].fillna("").apply(clean_text)

# ==============================
# 4. PREPROCESS KARBOHIDRAT
# ==============================
df["Karbo_List"] = df["Sumber_Karbohidrat"].apply(extract_karbo_list)
df["Karbo_Clean"] = df["Karbo_List"].apply(karbo_to_clean)

# ==============================
# 5. BUILD CORPUS
# ==============================
df["corpus_rf"] = df.apply(build_corpus_rf, axis=1)
df["enhanced_corpus_mccbf"] = df.apply(build_corpus_mccbf, axis=1)

# ==============================
# 6. REORDER COLUMNS
# ==============================
FINAL_COLUMNS = [
    "No",
    "Nama_Menu",
    "Kategori",
    "Kalori",
    "Sumber_Karbohidrat",
    "Deskripsi_Menu",
    "Karbo_List",
    "Karbo_Clean",
    "Keyword_Semantik",
    "enhanced_corpus_mccbf",
    "corpus_rf"
]

df = df[FINAL_COLUMNS]

# ==============================
# 7. SAVE DATASET
# ==============================
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Preprocessing selesai.")
print(f"📁 File disimpan di: {OUTPUT_PATH}")
