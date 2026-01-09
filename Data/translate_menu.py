import pandas as pd
import re

# ================================================================
# DICTIONARY TERJEMAHAN LENGKAP
# ================================================================

translation_dict = {
    # Protein utama
    'chicken': 'ayam',
    'beef': 'sapi',
    'fish': 'ikan',
    'dori': 'dori',
    'salmon': 'salmon',
    'prawn': 'udang',
    'seafood': 'seafood',
    
    # Metode masak
    'grill': 'panggang',
    'grilled': 'panggang',
    'baked': 'panggang',
    'fried': 'goreng',
    'crispy': 'krispi',
    'katsu': 'katsu',
    'bbq': 'barbeque',
    'rolade': 'rolade',
    
    # Bumbu & saus
    'teriyaki': 'teriyaki',
    'blackpepper': 'lada hitam',
    'honey': 'madu',
    'curry': 'kari',
    'woku': 'woku',
    'balado': 'balado',
    'salted egg': 'telur asin',
    'tomyum': 'tomyum',
    'yakiniku': 'yakiniku',
    'bulgogi': 'bulgogi',
    'sambal matah': 'sambal matah',
    'sambal ijo': 'sambal ijo',
    'saus': 'saus',
    'sauce': 'saus',
    'mentega': 'mentega',
    'wijen': 'wijen',
    'korea': 'korea',
    'asam manis': 'asam manis',
    'asam pedas': 'asam pedas',
    
    # Hidangan
    'stick': 'stik',
    'ball': 'bola',
    'burger': 'burger',
    'patty': 'patty',
    'sempol': 'sempol',
    'pangsit': 'pangsit',
    'bakso': 'bakso',
    'soto': 'soto',
    'rawon': 'rawon',
    'rendang': 'rendang',
    'opor': 'opor',
    'gulai': 'gulai',
    'bistik': 'bistik',
    'steak': 'stik',
    'spaghetti': 'spageti',
    'pasta': 'pasta',
    'pizza': 'piza',
    'sandwich': 'sandwich',
    'kebab': 'kebab',
    'gyudon': 'gyudon',
    'ramen': 'ramen',
    'sushi roll': 'sushi roll',
    'okonomiyaki': 'okonomiyaki',
    
    # Lainnya
    'with': 'dengan',
    'brown rice': 'nasi merah',
    'special': 'spesial',
    'premium': 'premium',
    'deluxe': 'istimewa',
    'supreme': 'super',
    'signature': 'khas',
    'healthy': 'sehat',
    'diet': 'diet',
    'box': 'paket',
    'bowl': 'mangkuk',
    'lengkap': 'lengkap',
    'spesial': 'spesial',
    'egg': 'telur',
    'tortila': 'tortila',
    'herbs': 'rempah',
    'popcorn': 'popcorn',
    'chips': 'keripik',
    'oat': 'oat',
    'karage': 'karage',
    'creamy': 'krim',
    'mushroom': 'jamur',
}

# ================================================================
# FUNGSI TERJEMAHAN
# ================================================================

def translate_menu_name(name):
    """Terjemahkan nama menu ke bahasa Indonesia"""
    
    if pd.isna(name):
        return name
    
    name = str(name).strip().lower()
    
    # Multi-word phrases (process first)
    multiword = [
        ('salted egg', 'telur asin'),
        ('sambal matah', 'sambal matah'),
        ('sambal ijo', 'sambal ijo'),
        ('brown rice', 'nasi merah'),
        ('blackpepper', 'lada hitam'),
        ('black pepper', 'lada hitam'),
        ('asam manis', 'asam manis'),
        ('asam pedas', 'asam pedas'),
        ('sushi roll', 'sushi roll'),
        ('egg tortila', 'telur tortila'),
    ]
    
    for eng, ind in multiword:
        name = name.replace(eng, ind)
    
    # Single words
    words = name.split()
    result = []
    
    for word in words:
        # Remove special chars
        clean = re.sub(r'[^\w]', '', word)
        
        # Translate
        if clean in translation_dict:
            result.append(translation_dict[clean])
        elif clean:
            result.append(clean)
    
    # Title case
    return ' '.join(result).title()


# ================================================================
# MAIN PROCESS
# ================================================================

def main():
    INPUT_FILE = 'dataset/dataset450_final.csv'
    OUTPUT_FILE = 'dataset/dataset450_indonesia.csv'
    
    print("="*70)
    print("🇮🇩 TERJEMAHKAN NAMA MENU KE BAHASA INDONESIA")
    print("="*70)
    
    # Load
    df = pd.read_csv(INPUT_FILE)
    print(f"\n✅ Loaded {len(df)} menu")
    
    # Check Karbo_List format
    print("\n🔍 Cek format Karbo_List:")
    sample_karbo = df['Karbo_List'].iloc[0]
    print(f"   Sample: {sample_karbo}")
    print(f"   Type: {type(sample_karbo)}")
    
    # Backup & translate
    df['Nama_Menu_Original'] = df['Nama_Menu']
    df['Nama_Menu'] = df['Nama_Menu_Original'].apply(translate_menu_name)
    
    # Show changes
    print("\n" + "="*70)
    print("📋 SAMPLE TERJEMAHAN")
    print("="*70)
    
    changes = 0
    for idx in range(min(20, len(df))):
        original = df.iloc[idx]['Nama_Menu_Original']
        translated = df.iloc[idx]['Nama_Menu']
        
        if original.lower() != translated.lower():
            print(f"\n{idx+1}. {original}")
            print(f"   → {translated}")
            changes += 1
    
    print(f"\n📊 Total diterjemahkan: {changes}/20 sample")
    
    # Save
    df_final = df.drop(columns=['Nama_Menu_Original'])
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Tersimpan: {OUTPUT_FILE}")
    
    # Sample final
    print("\n" + "="*70)
    print("📋 HASIL AKHIR (10 pertama)")
    print("="*70)
    
    for idx in range(min(10, len(df_final))):
        row = df_final.iloc[idx]
        print(f"\n{idx+1}. {row['Nama_Menu']}")
        print(f"   Kategori: {row['Kategori']}")
        print(f"   Kalori: {row['Kalori']}")
        print(f"   Karbo: {row['Karbo_Clean']}")
    
    print("\n" + "="*70)
    print("✅ SELESAI!")
    print("="*70)
    print("\n💡 Langkah selanjutnya:")
    print("   1. Update app_hybrid.py:")
    print("      DATASET_PATH = 'dataset/dataset450_indonesia.csv'")
    print("   2. Run: streamlit run app_hybrid.py")
    print("\n⚠️  TIDAK PERLU RETRAIN MODEL!")
    print("="*70)


if __name__ == "__main__":
    main()