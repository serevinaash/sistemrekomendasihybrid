import pandas as pd
import ast

# Baca file CSV asli
print("=" * 100)
print("MEMBACA FILE CSV...")
print("=" * 100)

df = pd.read_csv('dataset/dataset450_indonesia.csv')
print(f"✓ File berhasil dibaca!")
print(f"✓ Total baris: {len(df)}")
print(f"✓ Total kolom: {len(df.columns)}")

# Backup data asli
df_original = df.copy()

# PERBAIKAN 1: KATEGORI
print("\n" + "=" * 100)
print("PERBAIKAN 1: KATEGORI 'lainnya' → lebih spesifik")
print("=" * 100)

kategori_fixes = {
    69: "campuran",   # Rawon Surabaya Sapi Ayam
    76: "telur",      # Scotch Telur
    96: "telur",      # Scotch Telur
    138: "sayuran",   # Dengan Nasi Merah Dengan Potate Wedges Sehat
    140: "paket",     # Dengan Nasi Merah Paket
    145: "campuran",  # Rawon Surabaya Sapi Ayam Premium
    149: "paket",     # Nasi Merah Ce Apple Tea Saldo Gopay Diet
    232: "paket",     # Dengan Nasi Merah Sehat
    255: "paket",     # Dengan Nasi Merah Paket
    291: "paket",     # Dengan Nasi Merah Sehat
}

for no, new_kategori in kategori_fixes.items():
    idx = df[df['No'] == no].index
    if len(idx) > 0:
        old_kategori = df.loc[idx, 'Kategori'].values[0]
        df.loc[idx, 'Kategori'] = new_kategori
        nama_menu = df.loc[idx, 'Nama_Menu'].values[0]
        print(f"✓ Baris {no}: '{nama_menu}' → {old_kategori} menjadi {new_kategori}")

# PERBAIKAN 2: NORMALISASI SUMBER KARBOHIDRAT
print("\n" + "=" * 100)
print("PERBAIKAN 2: NORMALISASI SUMBER KARBOHIDRAT")
print("=" * 100)

# Fungsi untuk replace di semua kolom yang relevan
def normalize_karbo(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = text.replace('nasi coklat', 'nasi merah')
    text = text.replace('red grain rice', 'nasi merah')
    text = text.replace('garlic rice', 'nasi putih')
    return text

# Hitung perubahan
count_nasi_coklat = df['Sumber_Karbohidrat'].astype(str).str.contains('nasi coklat', na=False).sum()
count_red_grain = df['Sumber_Karbohidrat'].astype(str).str.contains('red grain rice', na=False).sum()
count_garlic = df['Sumber_Karbohidrat'].astype(str).str.contains('garlic rice', na=False).sum()

print(f"• 'nasi coklat' → 'nasi merah': {count_nasi_coklat} baris")
print(f"• 'red grain rice' → 'nasi merah': {count_red_grain} baris")
print(f"• 'garlic rice' → 'nasi putih': {count_garlic} baris")

# Terapkan normalisasi ke semua kolom terkait
df['Sumber_Karbohidrat'] = df['Sumber_Karbohidrat'].apply(normalize_karbo)
df['Karbo_Clean'] = df['Karbo_Clean'].apply(normalize_karbo)
df['Karbo_List'] = df['Karbo_List'].apply(normalize_karbo)

if 'corpus' in df.columns:
    df['corpus'] = df['corpus'].apply(normalize_karbo)

print("\n✓ Normalisasi selesai!")

# HASIL AKHIR
print("\n" + "=" * 100)
print("DISTRIBUSI KATEGORI SETELAH PERBAIKAN:")
print("=" * 100)
print(df['Kategori'].value_counts().to_string())

print("\n" + "=" * 100)
print("CONTOH PERUBAHAN DATA:")
print("=" * 100)

# Tampilkan beberapa contoh perubahan
contoh_no = [69, 76, 302, 312, 321]
for no in contoh_no:
    idx = df[df['No'] == no].index
    if len(idx) > 0:
        row_new = df.loc[idx].iloc[0]
        row_old = df_original.loc[idx].iloc[0]
        
        print(f"\nBaris {no}: {row_new['Nama_Menu']}")
        
        # Cek perubahan kategori
        if row_old['Kategori'] != row_new['Kategori']:
            print(f"  Kategori: {row_old['Kategori']} → {row_new['Kategori']}")
        
        # Cek perubahan karbo
        if str(row_old['Sumber_Karbohidrat']) != str(row_new['Sumber_Karbohidrat']):
            print(f"  Karbo: {row_old['Sumber_Karbohidrat']} → {row_new['Sumber_Karbohidrat']}")

# Simpan ke file baru
output_file = 'dataset/dataset450_indonesia_fixed.csv'
df.to_csv(output_file, index=False)

print("\n" + "=" * 100)
print("FILE BERHASIL DIPERBAIKI DAN DISIMPAN!")
print("=" * 100)
print(f"✓ File output: {output_file}")
print(f"✓ Total baris: {len(df)}")
print(f"✓ Total kolom: {len(df.columns)}")

# Summary perbaikan
print("\n" + "=" * 100)
print("RINGKASAN PERBAIKAN:")
print("=" * 100)
print(f"✓ Kategori diperbaiki: {len(kategori_fixes)} baris")
print(f"✓ Sumber karbohidrat dinormalisasi: {count_nasi_coklat + count_red_grain + count_garlic} baris")
print(f"✓ Total perubahan: {len(kategori_fixes) + count_nasi_coklat + count_red_grain + count_garlic} perubahan")

print("\nKategori baru yang ditambahkan:")
print("  • campuran: untuk menu dengan pilihan protein (sapi/ayam)")
print("  • telur: untuk menu berbasis telur (Scotch Telur)")
print("  • sayuran: untuk menu vegetarian")
print("  • paket: untuk paket yang tidak jelas proteinnya")

print("\nNormalisasi sumber karbohidrat:")
print("  • 'nasi coklat' → 'nasi merah' (brown rice)")
print("  • 'red grain rice' → 'nasi merah'")
print("  • 'garlic rice' → 'nasi putih'")

print("\n" + "=" * 100)
print("SELESAI! ✓")
print("=" * 100)