import pandas as pd
import ast
import re

# ================================================================
# COMPREHENSIVE KARBO CLEANING - FIXED VERSION
# ================================================================

def clean_karbo_final(text):
    """
    Clean karbohidrat dengan proper multi-word detection
    
    Input: "jagung nasi merah nasi putih"
    Output: ['jagung', 'nasi merah', 'nasi putih']
    """
    
    if pd.isna(text) or not str(text).strip():
        return []
    
    text = str(text).lower().strip()
    
    # ============================================
    # STEP 1: Multi-word Patterns (Priority Order)
    # ============================================
    # Detect dari yang paling panjang dulu untuk avoid overlap
    
    multiword_patterns = [
        # Rice variants (English)
        ('red grain rice', 'nasi merah'),
        ('brown rice', 'nasi merah'),
        ('white rice', 'nasi putih'),
        ('yellow rice', 'nasi kuning'),
        ('black rice', 'nasi hitam'),
        ('garlic rice', 'nasi putih'),
        
        # Rice variants (Indonesian)
        ('nasi merah', 'nasi merah'),
        ('nasi putih', 'nasi putih'),
        ('nasi coklat', 'nasi merah'),
        ('nasi hitam', 'nasi hitam'),
        ('nasi kuning', 'nasi kuning'),
        
        # Potato variants
        ('mashed potato', 'kentang'),
        ('french fries', 'kentang'),
        ('kentang tumbuk', 'kentang'),
        ('kentang goreng', 'kentang'),
        
        # Sweet potato
        ('ubi jalar', 'ubi'),
        ('sweet potato', 'ubi'),
        
        # Bread variants
        ('roti gandum', 'roti gandum'),
        ('roti putih', 'roti putih'),
        ('whole wheat bread', 'roti gandum'),
        ('white bread', 'roti putih'),
        
        # Corn
        ('sweet corn', 'jagung'),
        ('jagung manis', 'jagung'),
    ]
    
    # Replace multi-word dengan placeholder UNIK
    placeholders = {}
    counter = 0
    
    for pattern, normalized in multiword_patterns:
        if pattern in text:
            placeholder = f"___KARBO{counter}___"
            placeholders[placeholder] = normalized
            text = text.replace(pattern, placeholder)
            counter += 1
    
    # ============================================
    # STEP 2: Split dan Process Tokens
    # ============================================
    tokens = text.split()
    
    # Single word mapping
    singleword_map = {
        # Core carbs
        'rice': 'nasi',
        'nasi': 'nasi',
        'kentang': 'kentang',
        'potato': 'kentang',
        'potatoes': 'kentang',
        'corn': 'jagung',
        'jagung': 'jagung',
        'ubi': 'ubi',
        'yam': 'ubi',
        'bread': 'roti',
        'roti': 'roti',
        'pasta': 'pasta',
        'spaghetti': 'pasta',
        'penne': 'pasta',
        'macaroni': 'pasta',
        'mie': 'mie',
        'noodles': 'mie',
        'bihun': 'bihun',
        'vermicelli': 'bihun',
        
        # Ignore words (bukan karbo)
        'merah': None,
        'putih': None,
        'brown': None,
        'black': None,
        'yellow': None,
        'coklat': None,
        'kuning': None,
        'hitam': None,
        'grain': None,
        'garlic': None,
        'red': None,
        'white': None,
    }
    
    result = []
    
    for token in tokens:
        # Check if placeholder
        if token.startswith('___KARBO') and token.endswith('___'):
            normalized = placeholders.get(token)
            if normalized and normalized not in result:
                result.append(normalized)
        else:
            # Map single word
            mapped = singleword_map.get(token, token)
            if mapped is not None and len(mapped) > 2 and mapped not in result:
                result.append(mapped)
    
    return result


def process_dataset(input_path, output_path):
    """Process dataset dengan cleaning baru"""
    
    print("="*70)
    print("🔧 COMPREHENSIVE KARBO CLEANING - FIXED")
    print("="*70)
    
    # Load
    df = pd.read_csv(input_path)
    print(f"\n✅ Loaded {len(df)} rows from {input_path}")
    
    # Backup original
    df['Karbo_List_Original'] = df['Karbo_List'].copy()
    
    # Clean
    print("\n⏳ Cleaning Karbo_List...")
    df['Karbo_List'] = df['Sumber_Karbohidrat'].apply(clean_karbo_final)
    
    # Generate clean text version
    df['Karbo_Clean'] = df['Karbo_List'].apply(
        lambda x: ', '.join(x) if x else 'N/A'
    )
    
    # ============================================
    # VALIDATION & COMPARISON
    # ============================================
    
    print("\n" + "="*70)
    print("📊 SAMPLE CHANGES (First 20 with differences)")
    print("="*70)
    
    changes = 0
    for idx, row in df.iterrows():
        original_src = str(row['Sumber_Karbohidrat'])
        new_list = row['Karbo_List']
        
        try:
            old_list = ast.literal_eval(str(row['Karbo_List_Original']))
        except:
            old_list = []
        
        if old_list != new_list and changes < 20:
            print(f"\n#{idx+1}: {row['Nama_Menu']}")
            print(f"  Raw Source      : {original_src}")
            print(f"  Before Cleaning : {old_list}")
            print(f"  After Cleaning  : {new_list}")
            changes += 1
    
    # ============================================
    # STATISTICS
    # ============================================
    
    print("\n" + "="*70)
    print("📈 UNIQUE KARBOHIDRAT")
    print("="*70)
    
    # Collect all unique karbo
    all_karbo_before = set()
    for lst in df['Karbo_List_Original']:
        try:
            parsed = ast.literal_eval(str(lst))
            if isinstance(parsed, list):
                all_karbo_before.update([str(k).lower().strip() for k in parsed])
        except:
            pass
    
    all_karbo_after = set()
    for lst in df['Karbo_List']:
        if isinstance(lst, list):
            all_karbo_after.update(lst)
    
    print(f"\nBEFORE Cleaning ({len(all_karbo_before)} unique items):")
    print(sorted(all_karbo_before))
    
    print(f"\nAFTER Cleaning ({len(all_karbo_after)} unique items):")
    print(sorted(all_karbo_after))
    
    # ============================================
    # CHECK FOR EMPTY KARBO
    # ============================================
    
    empty_before = df['Karbo_List_Original'].apply(
        lambda x: len(x) == 0 if isinstance(x, list) else True
    ).sum()
    
    empty_after = df['Karbo_List'].apply(lambda x: len(x) == 0).sum()
    
    print(f"\n📊 Empty Karbo_List:")
    print(f"  Before: {empty_before}/{len(df)} ({empty_before/len(df)*100:.1f}%)")
    print(f"  After : {empty_after}/{len(df)} ({empty_after/len(df)*100:.1f}%)")
    
    # ============================================
    # VALIDATE SPECIFIC CASES
    # ============================================
    
    print("\n" + "="*70)
    print("🔍 VALIDATION: Critical Test Cases")
    print("="*70)
    
    test_cases = [
        ("jagung nasi merah nasi putih", ['jagung', 'nasi merah', 'nasi putih']),
        ("kentang ubi nasi merah nasi putih", ['kentang', 'ubi', 'nasi merah', 'nasi putih']),
        ("red grain rice", ['nasi merah']),
        ("garlic rice", ['nasi putih']),
        ("nasi coklat", ['nasi merah']),
        ("pasta", ['pasta']),
        ("jagung", ['jagung']),
        ("", []),
    ]
    
    all_pass = True
    for input_text, expected in test_cases:
        result = clean_karbo_final(input_text)
        status = "✅" if result == expected else "❌"
        
        if result != expected:
            all_pass = False
        
        print(f"\n{status} Input: '{input_text}'")
        print(f"   Expected: {expected}")
        print(f"   Got     : {result}")
    
    if all_pass:
        print("\n✅ All test cases PASSED!")
    else:
        print("\n⚠️  Some test cases FAILED!")
    
    # ============================================
    # SAVE
    # ============================================
    
    df_final = df.drop(columns=['Karbo_List_Original'])
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    print(f"✅ Saved to: {output_path}")
    
    # Show sample final results
    print("\n📋 Sample Final Results (First 10):")
    print("-"*70)
    sample = df_final[['Nama_Menu', 'Sumber_Karbohidrat', 'Karbo_Clean']].head(10)
    for idx, row in sample.iterrows():
        print(f"\n{idx+1}. {row['Nama_Menu']}")
        print(f"   Source: {row['Sumber_Karbohidrat']}")
        print(f"   Clean : {row['Karbo_Clean']}")
    
    return df_final


# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    
    INPUT_FILE = "dataset/dataset450_clean_mccbf.csv"
    OUTPUT_FILE = "dataset/dataset450_final.csv"
    
    print("\n")
    df_cleaned = process_dataset(INPUT_FILE, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("✅ CLEANING COMPLETED!")
    print("="*70)
    
    print("\n💡 NEXT STEPS:")
    print("="*70)
    print("1. ✅ Dataset cleaned → dataset/dataset450_final.csv")
    print("2. ✅ Update app_hybrid.py:")
    print("     DATASET_PATH = 'dataset/dataset450_final.csv'")
    print("3. ✅ Run: streamlit run app_hybrid.py")
    print("\n⚠️  NO NEED TO RETRAIN RF MODEL!")
    print("   RF model is for CATEGORY classification (ayam/ikan/sapi)")
    print("   Karbo_List is only used for MCCBF scoring")
    print("\n" + "="*70)