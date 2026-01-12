import pandas as pd
import re

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("dataset/dataset450.csv")


# ================================
# 2. FIX KATEGORI (4 kategori)
# ================================
def detect_category(nama):
    nama = nama.lower()

    # AYAM
    if any(x in nama for x in [
        "ayam", "chicken", "wing", "fillet", "paha", "dada", "suwir",
        "rolade ayam", "chick"
    ]):
        return "ayam"

    # SAPI
    if any(x in nama for x in [
        "sapi", "beef", "daging", "rawon", "rendang", "lada hitam",
        "bistik", "teriyaki beef"
    ]):
        return "sapi"

    # IKAN
    if any(x in nama for x in [
        "ikan", "fish", "dori", "salmon", "tuna", "kakap", "gurame",
        "nila", "seafood", "udang", "shrimp"
    ]):
        return "ikan"

    # VEGETARIAN (termasuk telur)
    if any(x in nama for x in [
        "tempe", "tahu", "tofu", "vegetarian", "veggie",
        "vegetable", "sayur", "salad", "telur", "egg",
        "omelet", "omelette"
    ]):
        return "vegetarian"

    return "ayam"  # fallback default


df["Kategori"] = df["Nama_Menu"].apply(detect_category)


# ================================
# 3. STANDARISASI KARBOHIDRAT
# ================================
def clean_karbo(x):
    if pd.isna(x):
        return []
    x = x.lower()
    x = re.sub(r"[,/]", " ", x)
    tokens = [k.strip() for k in x.split() if k.strip()]
    return list(set(tokens))

df["Karbo_List"] = df["Sumber_Karbohidrat"].apply(clean_karbo)


# ================================
# 4. NORMALISASI TEKS
# ================================
df["Nama_Lower"] = df["Nama_Menu"].str.lower()
df["Deskripsi_Lower"] = df["Deskripsi_Menu"].astype(str).str.lower()


# ================================
# 5. EXTRACT SEMANTIC KEYWORDS (MCCBF ONLY)
# ================================
def extract_keywords(text):
    t = text.lower()
    keywords = []

    # protein
    if "ayam" in t or "chicken" in t: keywords.append("protein_ayam")
    if "sapi" in t or "beef" in t or "daging" in t: keywords.append("protein_sapi")
    if "ikan" in t or "fish" in t or "dori" in t or "salmon" in t: keywords.append("protein_ikan")
    if any(x in t for x in ["tempe", "tahu", "vegetarian", "salad"]):
        keywords.append("protein_vegetarian")

    # cooking
    if any(x in t for x in ["bakar", "panggang", "grill"]):
        keywords.append("panggang")
    if any(x in t for x in ["goreng", "fried"]):
        keywords.append("goreng")
    if any(x in t for x in ["kukus", "steam"]):
        keywords.append("kukus")
    if any(x in t for x in ["rebus", "boil"]):
        keywords.append("rebus")

    # flavor
    if "pedas" in t: keywords.append("pedas")
    if "manis" in t: keywords.append("manis")
    if "teriyaki" in t: keywords.append("teriyaki")
    if "blackpepper" in t or "lada hitam" in t: keywords.append("blackpepper")

    return " ".join(keywords)

df["Keyword_Semantik"] = df["Nama_Lower"].apply(extract_keywords)


# ================================
# 6. CORPUS UNTUK MCCBF
# ================================
def build_mccbf_corpus(row):
    return (
        row["Nama_Lower"] + " " +
        row["Deskripsi_Lower"] + " " +
        " ".join(row["Karbo_List"]) + " " +
        row["Keyword_Semantik"]
    )

df["enhanced_corpus_mccbf"] = df.apply(build_mccbf_corpus, axis=1)


# ================================
# 7. CORPUS UNTUK RANDOM FOREST (BERSIH)
# ================================
def build_rf_corpus(row):
    return row["Nama_Lower"] + " " + row["Deskripsi_Lower"]

df["corpus_rf"] = df.apply(build_rf_corpus, axis=1)


# ================================
# 8. SAVE
# ================================
df.to_csv("dataset450_indonesia.csv", index=False)
print("✓ PREPROCESSING FINAL COMPLETE")
print("File saved: dataset450_indonesia.csv")
