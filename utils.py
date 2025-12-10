# ============================================================
# SHARED UTILITIES - Preprocessing & Scoring
# ============================================================

import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

MODES = {
    'seimbang': {
        'w_similarity': 0.35,
        'w_kalori': 0.10,
        'w_karbo': 0.25,
        'w_keyword': 0.30,
        'description': '⚖️ Balance antara semua kriteria'
    },
    'fokus_deskripsi': {
        'w_similarity': 0.45,
        'w_kalori': 0.10,
        'w_karbo': 0.20,
        'w_keyword': 0.25,
        'description': '📝 Prioritas detail rasa & metode masak'
    },
    'fokus_kategori': {
        'w_similarity': 0.25,
        'w_kalori': 0.10,
        'w_karbo': 0.30,
        'w_keyword': 0.35,
        'description': '🍖 Prioritas jenis lauk/protein'
    }
}

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def extract_keywords(text):
    """
    Extract semantic keywords dengan LOGIKA PRIORITAS.
    Rule: Jika terdeteksi Daging, abaikan label Vegetarian.
    """
    text = str(text).lower().strip()
    
    keywords_map = {
        'ayam': ['ayam', 'chicken', 'poultry', 'bebek', 'dada', 'paha', 'sayap'],
        'sapi': ['sapi', 'beef', 'daging', 'steak', 'rendang', 'rawon', 'bakso', 
                 'iga', 'buntut', 'meat', 'burger', 'short ribs', 'ribs'],
        'ikan': ['ikan', 'fish', 'dori', 'salmon', 'tuna', 'kakap', 'lele', 
                 'udang', 'cumi', 'seafood', 'prawn'],
        'vegetarian': [
            'vegetarian', 'vegan', 'tahu', 'tofu', 'tempe', 'tempeh', 
            'jamur', 'mushroom', 'telur', 'egg', 'sayur', 'vegetable', 
            'salad', 'jagung', 'corn', 'bayam', 'kangkung', 'brokoli',
            'buncis', 'terong', 'gado-gado', 'pecel', 'karedok'
        ]
    }
    
    found_categories = set()
    
    # Cek keberadaan keyword
    for category, keywords in keywords_map.items():
        if any(k in text for k in keywords):
            found_categories.add(category)
            
    # LOGIKA PRIORITAS: Meat overrides Veggie
    is_meat_present = ('ayam' in found_categories or 'sapi' in found_categories 
                      or 'ikan' in found_categories)
    
    if is_meat_present and 'vegetarian' in found_categories:
        found_categories.remove('vegetarian')
        
    # Fitur Tambahan
    features = [f"protein_{cat}" for cat in found_categories]
    
    cooking = ['goreng', 'bakar', 'rebus', 'kukus', 'panggang', 'grill', 
               'fried', 'grilled', 'roasted']
    if any(method in text for method in cooking):
        features.append("cooked")
    
    flavors = ['saus', 'sauce', 'bumbu', 'pedas', 'manis', 'asam', 'teriyaki', 
               'blackpepper', 'lada hitam', 'balado', 'curry', 'kari']
    if any(flavor in text for flavor in flavors):
        features.append("flavored")
    
    return " ".join(features) if features else text


def preprocess_menu(menu_name):
    """Preprocess menu untuk vectorization"""
    corpus = str(menu_name).lower()
    semantic = extract_keywords(menu_name)
    return corpus + " " + semantic


def clean_karbo_list(x):
    """Parse dan clean karbohidrat list dari dataset
    
    Dataset format:
    - "jagung nasi merah nasi putih" → ["jagung", "nasi merah", "nasi putih"]
    - "kentang ubi nasi merah" → ["kentang", "ubi", "nasi merah"]
    """
    import ast
    
    if pd.isna(x) or not x:
        return []
    
    x_str = str(x).strip()
    
    # Try list format first (format lama)
    if x_str.startswith('['):
        try:
            lst = ast.literal_eval(x_str)
            if isinstance(lst, list):
                # Combine nasi+merah, nasi+putih ke satu item
                return _combine_nasi(lst)
        except:
            pass
    
    # Parse space-separated dengan smart grouping
    words = x_str.split()
    result = []
    i = 0
    
    while i < len(words):
        word = words[i].lower().strip()
        
        # Jika ini "nasi", cek apakah ada warna setelahnya
        if word == 'nasi' and i + 1 < len(words):
            next_word = words[i + 1].lower().strip()
            if next_word in ['merah', 'putih', 'kuning', 'hitam']:
                result.append(f"{word} {next_word}")
                i += 2
                continue
        
        # Jika ini "umbi", cek apakah ada "-umbian"
        if word == 'umbi' and i + 1 < len(words) and words[i + 1].lower().strip() == 'umbian':
            result.append('umbi umbian')
            i += 2
            continue
        
        result.append(word)
        i += 1
    
    return [k for k in result if k]


def _combine_nasi(lst):
    """Helper untuk combine nasi+color dari list format"""
    result = []
    i = 0
    lst_lower = [str(k).lower().strip() for k in lst]
    
    while i < len(lst_lower):
        item = lst_lower[i]
        
        if item == 'nasi' and i + 1 < len(lst_lower):
            next_item = lst_lower[i + 1]
            if next_item in ['merah', 'putih', 'kuning', 'hitam']:
                result.append(f"{item} {next_item}")
                i += 2
                continue
        
        result.append(item)
        i += 1
    
    return result


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def gaussian_calorie_score(menu_cal, target_cal, sigma=10):
    """Gaussian scoring untuk kalori"""
    if target_cal is None or target_cal <= 0:
        return 1.0
    try:
        diff = abs(float(target_cal) - float(menu_cal))
        score = math.exp(-(diff ** 2) / (2 * (sigma ** 2)))
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def karbo_score(menu_karbo_list, preferred_karbo):
    """Jaccard similarity untuk karbohidrat"""
    if not preferred_karbo:
        return 1.0
    if not menu_karbo_list:
        return 0.5
    
    menu_set = {k.lower().strip() for k in menu_karbo_list}
    pref_set = {k.lower().strip() for k in preferred_karbo}
    
    inter = len(menu_set & pref_set)
    union = len(menu_set | pref_set)
    
    return inter / union if union > 0 else 0.5


def keyword_boost_score(menu_desc, user_query):
    """Keyword boost scoring"""
    if not user_query or pd.isna(user_query):
        return 0.0
    
    important_keywords = {
        'pedas', 'manis', 'gurih', 'asam', 'asin',
        'panggang', 'bakar', 'goreng', 'kukus', 'rebus', 'tumis',
        'crispy', 'grill', 'renyah',
        'rendah', 'tinggi', 'tanpa', 'kuah', 'kering',
        'bening', 'lembut', 'empuk', 'segar',
        'protein', 'santan', 'lemak', 'minyak',
        'teriyaki', 'balado', 'sambal', 'woku', 'korea',
        'yakiniku', 'bulgogi', 'curry', 'soto', 'rawon',
        'sehat', 'diet', 'organik', 'fit', 'healthy'
    }
    
    menu_desc = str(menu_desc).lower() if not pd.isna(menu_desc) else ""
    user_kw = set(str(user_query).lower().split())
    menu_kw = set(menu_desc.split())
    
    matched = user_kw & menu_kw & important_keywords
    
    return min(0.35, len(matched) * 0.10)


# ============================================================
# MCCBF SCORING
# ============================================================

def compute_mccbf_scores(df, X_all, vectorizer, user_query, 
                         target_calorie=None, preferred_karbo=None, 
                         mode="seimbang"):
    """
    Hitung MCCBF score untuk semua menu.
    
    Returns:
        tuple: (mccbf_scores, components_dict)
    """
    weights = MODES.get(mode, MODES['seimbang'])
    
    # Similarity component
    enhanced_query = preprocess_menu(user_query)
    X_user = vectorizer.transform([enhanced_query])
    sim = cosine_similarity(X_user, X_all)[0]
    sim_norm = sim / sim.max() if sim.max() > 0 else sim
    
    # Calorie component
    cal_scores = df["Kalori"].apply(
        lambda c: gaussian_calorie_score(c, target_calorie)
    ).values
    
    # Karbo component
    karbo_scores = df["Karbo_List"].apply(
        lambda kl: karbo_score(kl, preferred_karbo)
    ).values
    
    # Keyword component
    keyword_boosts = df["Deskripsi_Menu"].apply(
        lambda desc: keyword_boost_score(desc, user_query)
    ).values
    
    # Weighted combination
    mccbf = (
        weights['w_similarity'] * sim_norm +
        weights['w_kalori'] * cal_scores +
        weights['w_karbo'] * karbo_scores +
        weights['w_keyword'] * keyword_boosts
    )
    
    components = {
        'similarity': sim_norm,
        'calorie': cal_scores,
        'karbo': karbo_scores,
        'keyword': keyword_boosts
    }
    
    return mccbf, components


def rf_category_scores(df, X_all, rf_model, preferred_categories=None):
    """RF scoring berdasarkan kategori preferensi"""
    proba = rf_model.predict_proba(X_all)
    classes = list(rf_model.classes_)
    
    if not preferred_categories:
        return proba.max(axis=1)
    
    idx_list = [classes.index(kat) for kat in preferred_categories 
                if kat in classes]
    
    if not idx_list:
        return proba.max(axis=1)
    
    return proba[:, idx_list].max(axis=1)


# ============================================================
# FORMATTING FUNCTIONS (untuk UI/Display)
# ============================================================

def format_karbo_display(karbo_list, user_prefs):
    """Format display karbohidrat dengan highlight"""
    if not karbo_list or len(karbo_list) == 0:
        return "_Tidak ada data_"
    
    user_prefs_lower = [k.lower() for k in user_prefs] if user_prefs else []
    
    karbo_display = []
    for item in karbo_list:
        item_title = item.title()
        
        is_match = False
        for pref in user_prefs_lower:
            if pref in item.lower() or item.lower() in pref:
                is_match = True
                break
        
        if is_match:
            karbo_display.append(f"`{item_title}` ✅")
        else:
            karbo_display.append(f"`{item_title}`")
    
    return ' '.join(karbo_display)