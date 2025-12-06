import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import ast
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Sistem Rekomendasi Menu Diet Sehat - Hybrid",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# KONSTANTA
# ========================================
DATASET_PATH = "dataset/dataset450_clean_mccbf.csv"
RFDATA_DIR = "rfdata"
ALPHA_DEFAULT = 0.7

MODES = {
    'seimbang': {
        'w_similarity': 0.35,
        'w_kalori': 0.10,
        'w_karbo': 0.25,
        'w_keyword': 0.30
    },
    'fokus_deskripsi': {
        'w_similarity': 0.45,
        'w_kalori': 0.10,
        'w_karbo': 0.20,
        'w_keyword': 0.25
    },
    'fokus_kategori': {
        'w_similarity': 0.25,
        'w_kalori': 0.10,
        'w_karbo': 0.30,
        'w_keyword': 0.35
    }
}

# ========================================
# PREPROCESSING FUNCTIONS
# ========================================
def extract_keywords(text):
    """Ekstrak semantic keywords"""
    text = str(text).lower().strip()
    
    protein_keywords = {
        'ayam': ['ayam', 'chicken', 'poultry', 'wing', 'breast', 'drumstick'],
        'sapi': ['sapi', 'beef', 'daging', 'steak', 'bistik', 'rendang', 'rawon', 'bakso', 'iga'],
        'ikan': ['ikan', 'fish', 'dori', 'salmon', 'tuna', 'kakap', 'gurame', 'nila', 'seafood'],
        'lainnya': ['vegetarian', 'vegan', 'salad', 'kentang', 'potato', 'sayur', 'vegetable']
    }
    
    features = []
    for category, keywords in protein_keywords.items():
        if any(kw in text for kw in keywords):
            features.append(f"protein_{category}")
    
    cooking = ['goreng', 'bakar', 'rebus', 'kukus', 'panggang', 'grill', 'fried', 'grilled']
    if any(method in text for method in cooking):
        features.append("cooked")
    
    flavors = ['saus', 'sauce', 'bumbu', 'pedas', 'manis', 'asam', 'teriyaki', 'blackpepper']
    if any(flavor in text for flavor in flavors):
        features.append("flavored")
    
    return " ".join(features) if features else text

def preprocess_menu(menu_name):
    """Preprocess menu untuk vectorization"""
    corpus = str(menu_name).lower()
    semantic = extract_keywords(menu_name)
    return corpus + " " + semantic

# ========================================
# LOAD MODEL (dengan caching)
# ========================================
@st.cache_resource
def load_hybrid_engine():
    """Load semua assets untuk hybrid engine"""
    try:
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Parse Karbo_List
        if "Karbo_List" in df.columns:
            def to_list(x):
                if isinstance(x, list):
                    return x
                try:
                    return ast.literal_eval(x)
                except:
                    return []
            df["Karbo_List"] = df["Karbo_List"].apply(to_list)
        else:
            df["Karbo_List"] = [[] for _ in range(len(df))]
        
        # Generate enhanced corpus
        df["enhanced_corpus"] = df["Nama_Menu"].apply(preprocess_menu)
        
        # Load RF model & vectorizer
        rf_model = joblib.load(os.path.join(RFDATA_DIR, "rf_model_production.pkl"))
        vectorizer = joblib.load(os.path.join(RFDATA_DIR, "tfidf_vectorizer_production.pkl"))
        
        # Precompute TF-IDF
        X_all = vectorizer.transform(df["enhanced_corpus"])
        
        return {
            'df': df,
            'rf_model': rf_model,
            'vectorizer': vectorizer,
            'X_all': X_all,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

# ========================================
# SCORING FUNCTIONS
# ========================================
def gaussian_calorie_score(menu_cal, target_cal, sigma=30):
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
    """Jaccard similarity untuk karbo"""
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

def mccbf_score(df, X_all, vectorizer, user_query, 
                target_calorie=None, preferred_karbo=None, mode="seimbang"):
    """Hitung MCCBF score"""
    weights = MODES.get(mode, MODES['seimbang'])
    
    # 1) Similarity score
    enhanced_query = preprocess_menu(user_query)
    X_user = vectorizer.transform([enhanced_query])
    sim = cosine_similarity(X_user, X_all)[0]
    sim_norm = sim / sim.max() if sim.max() > 0 else sim
    
    # 2) Calorie score
    cal_scores = df["Kalori"].apply(
        lambda c: gaussian_calorie_score(c, target_calorie)
    ).values
    
    # 3) Karbo score
    karbo_scores = df["Karbo_List"].apply(
        lambda kl: karbo_score(kl, preferred_karbo)
    ).values
    
    # 4) Keyword boost
    keyword_boosts = df["Deskripsi_Menu"].apply(
        lambda desc: keyword_boost_score(desc, user_query)
    ).values
    
    # Weighted sum
    mccbf = (
        weights['w_similarity'] * sim_norm +
        weights['w_kalori'] * cal_scores +
        weights['w_karbo'] * karbo_scores +
        weights['w_keyword'] * keyword_boosts
    )
    
    return mccbf, {
        'similarity': sim_norm,
        'calorie': cal_scores,
        'karbo': karbo_scores,
        'keyword': keyword_boosts
    }

def rf_category_scores(df, X_all, rf_model, preferred_categories=None):
    """RF scoring berdasarkan kategori"""
    proba = rf_model.predict_proba(X_all)
    classes = list(rf_model.classes_)
    
    if not preferred_categories:
        return proba.max(axis=1)
    
    idx_list = [classes.index(kat) for kat in preferred_categories if kat in classes]
    
    if not idx_list:
        return proba.max(axis=1)
    
    return proba[:, idx_list].max(axis=1)

def recommend_hybrid(engine, user_query, target_calorie=None,
                    preferred_categories=None, preferred_karbo=None,
                    mode="seimbang", alpha=0.7, top_n=10):
    """Hybrid recommender"""
    df = engine['df']
    rf_model = engine['rf_model']
    vectorizer = engine['vectorizer']
    X_all = engine['X_all']
    
    if preferred_categories is None:
        preferred_categories = []
    if preferred_karbo is None:
        preferred_karbo = []
    
    # MCCBF Score
    mccbf, mccbf_components = mccbf_score(
        df, X_all, vectorizer,
        user_query=user_query,
        target_calorie=target_calorie,
        preferred_karbo=preferred_karbo,
        mode=mode
    )
    
    # RF Score
    rf_scores = rf_category_scores(
        df, X_all, rf_model,
        preferred_categories=preferred_categories
    )
    
    # Hybrid Score
    hybrid = alpha * mccbf + (1 - alpha) * rf_scores
    
    # Build result
    result = df[[
        'Nama_Menu', 'Kategori', 'Kalori', 
        'Karbo_List', 'Deskripsi_Menu'
    ]].copy()
    
    result["mccbf_score"] = mccbf
    result["mccbf_similarity"] = mccbf_components['similarity']
    result["mccbf_calorie"] = mccbf_components['calorie']
    result["mccbf_karbo"] = mccbf_components['karbo']
    result["mccbf_keyword"] = mccbf_components['keyword']
    result["rf_score"] = rf_scores
    result["hybrid_score"] = hybrid
    
    # Sort & return top N
    result = result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    return result.head(top_n)

# ========================================
# LOAD ENGINE
# ========================================
engine = load_hybrid_engine()

# ========================================
# HEADER APLIKASI
# ========================================
st.title("🥗 Sistem Rekomendasi Menu Diet Sehat - HYBRID")
st.markdown("### Icel's Room Kitchen - Katering Diet Personal")
st.markdown("**🔬 Powered by: MCCBF + Random Forest Classifier**")
st.markdown("---")

# Info sistem
with st.expander("ℹ️ Tentang Sistem Hybrid"):
    st.write("""
    Sistem ini menggunakan **pendekatan hybrid** yang menggabungkan:
    
    ### 🔵 MCCBF (Multi-Criteria Content-Based Filtering)
    - 🔢 Gaussian Calorie Scoring (sigma=30)
    - 🍚 Jaccard Similarity untuk Karbohidrat
    - 📝 TF-IDF + Bigram untuk Deskripsi
    - 🏷️ Keyword Boost System
    
    ### 🟢 Random Forest Classifier
    - 🎯 Kategori Prediction dengan 100 trees
    - 📊 Trained pada 450 menu dari 3 top catering Indonesia
    - ✅ Accuracy: ~80% (Test Set)
    
    ### ⚖️ Hybrid Blending
    - **Alpha (α)**: Weight untuk MCCBF (default: 0.7)
    - **(1-α)**: Weight untuk RF
    - Formula: `Hybrid Score = α × MCCBF + (1-α) × RF`
    
    **Dataset:** 450 menu dari Icel's Room Kitchen, Katering Fit, dan Diet Catering Indonesia
    """)

# Check engine status
if engine['status'] == 'error':
    st.error(f"❌ Gagal load model: {engine['error']}")
    st.stop()

df = engine['df']

# ========================================
# SIDEBAR - INPUT USER
# ========================================
st.sidebar.header("🎯 Preferensi Anda")

# Info dataset
st.sidebar.caption(f"📊 **Dataset:** {len(df)} menu tersedia")

# 1️⃣ Target Kalori
kalori_min = int(df['Kalori'].min())
kalori_max = int(df['Kalori'].max())
kalori_mean = int(df['Kalori'].mean())

col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Min", f"{kalori_min}")
col2.metric("Avg", f"{kalori_mean}")
col3.metric("Max", f"{kalori_max}")

kalori_target = st.sidebar.slider(
    "Target Kalori (kcal)",
    min_value=kalori_min,
    max_value=kalori_max,
    value=kalori_mean,
    step=5
)

# 2️⃣ Kategori (untuk RF)
kategori_options = sorted(df['Kategori'].dropna().unique().tolist())
kategori_pref = st.sidebar.multiselect(
    "Pilih Kategori Lauk (untuk RF scoring)",
    options=kategori_options,
    default=[kategori_options[0]] if kategori_options else [],
    help="RF akan prioritaskan kategori ini"
)

# 3️⃣ Karbohidrat
if "Karbo_List" in df.columns:
    karbo_raw = []
    for lst in df["Karbo_List"]:
        if isinstance(lst, list):
            karbo_raw.extend([k.lower().strip() for k in lst])
    karbo_options = sorted(list(set(karbo_raw)))
else:
    karbo_options = []

pilihan_karbo = st.sidebar.multiselect(
    "Pilih Sumber Karbohidrat",
    options=karbo_options,
    default=[karbo_options[0]] if karbo_options else [],
    help="MCCBF akan prioritaskan karbohidrat ini"
)

# 4️⃣ Deskripsi Query
st.sidebar.markdown("**📝 Deskripsi Preferensi:**")

# Quick tags
col1, col2 = st.sidebar.columns(2)
if 'query' not in st.session_state:
    st.session_state.query = "ayam bakar pedas sehat"

with col1:
    if st.button("🟢 Rendah Lemak"):
        st.session_state.query = "rendah lemak sehat"
    if st.button("🟢 Tinggi Protein"):
        st.session_state.query = "tinggi protein"
    if st.button("🟢 Diet"):
        st.session_state.query = "diet sehat rendah kalori"

with col2:
    if st.button("🔴 Tanpa Santan"):
        st.session_state.query = "tanpa santan"
    if st.button("🔴 Tidak Pedas"):
        st.session_state.query = "tidak pedas"
    if st.button("🔴 Kukus"):
        st.session_state.query = "kukus sehat"

user_query = st.sidebar.text_area(
    "Atau Ketik Manual:",
    value=st.session_state.query,
    height=80,
    help="Deskripsikan preferensi menu Anda"
)

# 5️⃣ Mode & Alpha
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Hybrid")

mode = st.sidebar.radio(
    "Mode MCCBF",
    options=["seimbang", "fokus_deskripsi", "fokus_kategori"],
    help="Mode bobot untuk MCCBF component"
)

alpha = st.sidebar.slider(
    "Alpha (α) - Weight MCCBF",
    min_value=0.0,
    max_value=1.0,
    value=ALPHA_DEFAULT,
    step=0.05,
    help=f"α={ALPHA_DEFAULT} = 70% MCCBF + 30% RF"
)

st.sidebar.caption(f"📊 **Blending:** {alpha*100:.0f}% MCCBF + {(1-alpha)*100:.0f}% RF")

top_n = st.sidebar.slider(
    "Jumlah Rekomendasi",
    min_value=3,
    max_value=20,
    value=10
)

show_debug = st.sidebar.checkbox("🔍 Mode Debug")

# ========================================
# TOMBOL GENERATE
# ========================================
st.sidebar.markdown("---")
generate_btn = st.sidebar.button("🚀 Dapatkan Rekomendasi", type="primary")

# ========================================
# MAIN CONTENT
# ========================================
if generate_btn:
    with st.spinner("🔍 Computing hybrid scores..."):
        try:
            recommendations = recommend_hybrid(
                engine=engine,
                user_query=user_query,
                target_calorie=kalori_target,
                preferred_categories=kategori_pref,
                preferred_karbo=pilihan_karbo,
                mode=mode,
                alpha=alpha,
                top_n=top_n
            )
            
            st.success("✅ Rekomendasi berhasil dibuat!")
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔥 Target Kalori", f"{kalori_target} kcal")
            with col2:
                st.metric("🍖 Kategori (RF)", f"{len(kategori_pref)} jenis")
            with col3:
                st.metric("🍚 Karbo (MCCBF)", f"{len(pilihan_karbo)} jenis")
            with col4:
                st.metric("⚙️ Mode", mode.title())
            
            st.caption(f"📝 **Query:** {user_query}")
            st.caption(f"⚖️ **Blending:** α={alpha} ({alpha*100:.0f}% MCCBF + {(1-alpha)*100:.0f}% RF)")
            
            st.markdown("---")
            st.subheader(f"🏆 Top-{top_n} Rekomendasi Hybrid")
            
            if recommendations.empty:
                st.warning("⚠️ Tidak ada menu yang cocok.")
            else:
                for idx, row in recommendations.iterrows():
                    rank = idx + 1
                    
                    with st.container():
                        col_rank, col_info = st.columns([1, 9])
                        
                        with col_rank:
                            st.markdown(f"### #{rank}")
                        
                        with col_info:
                            st.markdown(f"**{row['Nama_Menu'].title()}**")
                            st.caption(f"Kategori: {row['Kategori'].title()} | Kalori: {row['Kalori']} kcal")
                            
                            if row['Deskripsi_Menu'] and str(row['Deskripsi_Menu']) != 'nan':
                                st.write(f"📝 {row['Deskripsi_Menu'].capitalize()}")
                            
                            # Karbo display
                            karbo_list = row['Karbo_List']
                            if karbo_list and len(karbo_list) > 0:
                                karbo_display = []
                                for item in karbo_list:
                                    if item.lower() in [k.lower() for k in pilihan_karbo]:
                                        karbo_display.append(f"`{item.title()}` ✅")
                                    else:
                                        karbo_display.append(f"`{item.title()}`")
                                st.markdown(f"🍚 **Karbohidrat:** {' '.join(karbo_display)}")
                            
                            # Scores
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.caption(f"🔵 MCCBF: {row['mccbf_score']:.3f}")
                            with col_b:
                                st.caption(f"🟢 RF: {row['rf_score']:.3f}")
                            with col_c:
                                st.caption(f"⭐ Hybrid: {row['hybrid_score']:.3f}")
                            
                            if show_debug:
                                st.caption("📊 **MCCBF Breakdown:**")
                                st.caption(f"   • Similarity: {row['mccbf_similarity']:.3f}")
                                st.caption(f"   • Calorie: {row['mccbf_calorie']:.3f}")
                                st.caption(f"   • Karbo: {row['mccbf_karbo']:.3f}")
                                st.caption(f"   • Keyword: {row['mccbf_keyword']:.3f}")
                        
                        st.markdown("---")
                
                # Download button
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name=f'rekomendasi_hybrid_alpha{alpha}.csv',
                    mime='text/csv',
                )
                
                # Debug table
                if show_debug:
                    st.markdown("---")
                    st.subheader("🔍 Debug: Score Details")
                    debug_df = recommendations[[
                        'Nama_Menu', 'hybrid_score', 'mccbf_score', 'rf_score',
                        'mccbf_similarity', 'mccbf_calorie', 'mccbf_karbo', 'mccbf_keyword'
                    ]]
                    st.dataframe(debug_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Masukkan preferensi di sidebar, lalu klik **Dapatkan Rekomendasi**")
    
    # Preview dataset
    st.subheader("📊 Preview Dataset (450 Menu)")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Total Menu:** {len(df)}")
    with col2:
        show_all = st.checkbox("Tampilkan Semua")
    
    display_df = df[['Nama_Menu', 'Kategori', 'Kalori']].copy()
    
    if show_all:
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.dataframe(display_df.head(10), use_container_width=True)
    
    # Statistik
    with st.expander("📊 Statistik Dataset"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Menu", len(df))
            st.metric("Kategori", df['Kategori'].nunique())
        
        with col2:
            st.metric("Kalori Min", f"{df['Kalori'].min():.0f}")
            st.metric("Kalori Max", f"{df['Kalori'].max():.0f}")
        
        with col3:
            st.metric("Kalori Avg", f"{df['Kalori'].mean():.0f}")
            st.metric("Sumber Data", "3 Catering")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("🔬 Hybrid Recommender System: MCCBF (70%) + Random Forest (30%)")
st.caption("📊 Dataset: 450 menu dari 3 top catering diet Indonesia")