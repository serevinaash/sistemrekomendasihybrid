import pandas as pd
import re
import ast

# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):
    """Lowercase, remove punctuation, normalize whitespace."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================
# PARSE KARBOHIDRAT — ANTI ERROR
# ==========================================

def parse_karbo(value):
    """
    Convert berbagai format ke list aman.
    Bisa handle:
    - NaN
    - "Jagung, Nasi Putih"
    - "['jagung','nasi putih']"
    - list Python
    - Series (error fix)
    """
    # Jika Series → ambil elemen pertama
    if isinstance(value, pd.Series):
        if len(value) > 0:
            value = value.iloc[0]
        else:
            return []

    # Jika NaN → []
    if value is None or pd.isna(value):
        return []

    # Jika list Python → bersihkan
    if isinstance(value, list):
        return [clean_text(v) for v in value if isinstance(v, str)]

    # Jika string list Python
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [clean_text(v) for v in parsed]
        except:
            pass  # fallback ke parse biasa

    # Format biasa "Jagung, Nasi Putih"
    value = str(value).replace(",", " ")
    parts = [p.strip().lower() for p in value.split() if p.strip()]

    return parts


# ==========================================
# MAP KATEGORI → MCCBF LAMA
# ==========================================

def map_kategori(kat):
    if kat is None or pd.isna(kat):
        return "lainnya"
    kat = str(kat).lower().strip()

    if kat == "ayam":
        return "ayam"
    if kat == "sapi":
        return "sapi"
    if kat in ["ikan", "seafood"]:
        return "ikan"
    return "lainnya"


# ==========================================
# MAIN PREPROCESSOR
# ==========================================

def preprocess_dataset(
    input_path="data/dataset450.csv",
    output_path="dataset450_clean_mccbf.csv"
):
    print("🔧 Memuat dataset 450...")

    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")

    print("🔧 Normalisasi kolom...")

    column_map = {
        "No": "No",
        "Nama Menu": "Nama_Menu",
        "Nama_Menu": "Nama_Menu",
        "Kategori": "Kategori",
        "Kategori_Clean": "Kategori",
        "Kalori": "Kalori",
        "Kalori (kcal)": "Kalori",
        "Deskripsi": "Deskripsi_Menu",
        "Deskripsi Singkat": "Deskripsi_Menu",
        "Deskripsi_Menu": "Deskripsi_Menu",
        "Sumber Karbohidrat": "Sumber_Karbohidrat",
        "Karbo_List": "Karbo_List",
        "Bahan Utama / Pendamping": "Sumber_Karbohidrat"
    }

    df = df.rename(columns=column_map)

    # FIX: hapus kolom duplikat
    df = df.loc[:, ~df.columns.duplicated()]

    print("Kolom setelah rename:", df.columns.tolist())

    print("🔧 Bersihkan nomor baris...")

    df = df[df["No"].astype(str).str.isnumeric()]
    df["No"] = df["No"].astype(int)
    df = df.sort_values("No").reset_index(drop=True)

    print("🔧 Map kategori...")

    df["Kategori"] = df["Kategori"].apply(map_kategori)

    print("🔧 Clean nama menu & deskripsi...")

    df["Nama_Menu"] = df["Nama_Menu"].apply(clean_text)
    df["Deskripsi_Menu"] = df["Deskripsi_Menu"].apply(clean_text)

    print("🔧 Parse karbohidrat...")

    df["Karbo_List"] = df["Sumber_Karbohidrat"].apply(parse_karbo)
    df["Sumber_Karbohidrat"] = df["Karbo_List"].apply(lambda lst: " ".join(lst))

    print("🔧 Bangun corpus MCCBF...")

    df["corpus"] = (
        df["Nama_Menu"] + " " +
        df["Kategori"] + " " +
        df["Sumber_Karbohidrat"] + " " +
        df["Deskripsi_Menu"]
    ).apply(clean_text)

    # Kolom sesuai MCCBF Lama
    final_cols = [
        "No",
        "Nama_Menu",
        "Kategori",
        "Kalori",
        "Sumber_Karbohidrat",
        "Deskripsi_Menu",
        "Karbo_List",
        "corpus"
    ]

    df = df[final_cols]

    print("💾 Menyimpan hasil:", output_path)
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing selesai!")
    print(f"📊 Total baris final: {len(df)}")

    return df


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    preprocess_dataset()
