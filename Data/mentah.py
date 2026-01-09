import pandas as pd

# ================================
# 1. LOAD DATA ASLI
# ================================
df = pd.read_csv("dataset450_indonesia.csv")

# ================================
# 2. PILIH KOLOM DATA MENTAH
# ================================
raw_columns = [
    "No",
    "Nama_Menu",
    "Kategori",
    "Kalori",
    "Sumber_Karbohidrat",
    "Deskripsi_Menu"
]

df_raw = df[raw_columns]

# ================================
# 3. SIMPAN SEBAGAI DATA MENTAH
# ================================
df_raw.to_csv("dataset450_raw.csv", index=False)

print("✓ DATA MENTAH BERHASIL DIBUAT")
print("File saved as: dataset450_raw.csv")
