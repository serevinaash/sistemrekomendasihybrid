import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import ast
import joblib
import re
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
# BEST MODEL CONFIGURATION
# ========================================
BEST_SPLIT = "70_30"           # ← Menggunakan split 70_30
ALPHA_BEST = 0.7                # ← 70% MCCBF + 30% RF
MODE_BEST = "seimbang"          # ← Mode terbaik

DATASET_PATH = "dataset/dataset450_indonesia.csv"
RFDATA_DIR = f"rftrain/{BEST_SPLIT}"  # Load model dari split terbaik

# ========================================
# MODES CONFIGURATION
# ========================================
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

# ========================================
# KARBO CLEANING FUNCTIONS
# ========================================
def clean_karbo_list(x):
    if pd.isna(x) or not x:
        return []
    try:
        lst = ast.literal_eval(str(x))
        if not isinstance(lst, list):
            lst = [lst]
    except:
        lst = str(x).split()
    return [k.lower().strip() for k in lst if k]

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

# ========================================
# PREPROCESSING FUNCTIONS (VERSI UPDATE)
# ========================================
def extract_keywords(text):
    """
    Ekstrak semantic keywords dengan LOGIKA PRIORITAS (SAMA DENGAN TRAINING).
    """
    text = str(text).lower().strip()
    
    # 1. Definisi Kamus Kata Kunci
    keywords_map = {
        'ayam': ['ayam', 'chicken', 'poultry', 'bebek', 'dada', 'paha', 'sayap'],
        'sapi': ['sapi', 'beef', 'daging', 'steak', 'rendang', 'rawon', 'bakso', 'iga', 'buntut', 'meat', 'burger', 'short ribs', 'ribs'],
        'ikan': ['ikan', 'fish', 'dori', 'salmon', 'tuna', 'kakap', 'lele', 'udang', 'cumi', 'seafood', 'prawn'],
        'vegetarian': [
            'vegetarian', 'vegan', 'tahu', 'tofu', 'tempe', 'tempeh', 
            'jamur', 'mushroom', 'telur', 'egg', 'sayur', 'vegetable', 
            'salad', 'jagung', 'corn', 'bayam', 'kangkung', 'brokoli',
            'buncis', 'terong', 'gado-gado', 'pecel', 'karedok'
        ]
    }
    
    found_categories = set()
    
    # 2. Cek keberadaan keyword
    for category, keywords in keywords_map.items():
        if any(k in text for k in keywords):
            found_categories.add(category)
            
    # 3. LOGIKA PRIORITAS (Meat overrides Veggie)
    is_meat_present = 'ayam' in found_categories or 'sapi' in found_categories or 'ikan' in found_categories
    
    if is_meat_present and 'vegetarian' in found_categories:
        found_categories.remove('vegetarian')
        
    # 4. Fitur Tambahan
    features = [f"protein_{cat}" for cat in found_categories]
    
    cooking = ['goreng', 'bakar', 'rebus', 'kukus', 'panggang', 'grill', 'fried', 'grilled', 'roasted']
    if any(method in text for method in cooking):
        features.append("cooked")
    
    flavors = ['saus', 'sauce', 'bumbu', 'pedas', 'manis', 'asam', 'teriyaki', 'blackpepper', 'lada hitam', 'balado', 'curry', 'kari']
    if any(flavor in text for flavor in flavors):
        features.append("flavored")
    
    if not features:
        return text 
        
    return " ".join(features)

# ========================================
# LOAD MODEL (dengan caching)
# ========================================
@st.cache_resource
def load_hybrid_engine():
    """Load semua assets untuk hybrid engine"""
    try:
        df = pd.read_csv(DATASET_PATH)
        
        if "Karbo_List" in df.columns:
            df["Karbo_List"] = df["Karbo_List"].apply(clean_karbo_list)
        else:
            df["Karbo_List"] = [[] for _ in range(len(df))]
        
        df["enhanced_corpus"] = df["Nama_Menu"].apply(preprocess_menu)
        
        rf_model = joblib.load(os.path.join(RFDATA_DIR, "rf_model.pkl"))
        vectorizer = joblib.load(os.path.join(RFDATA_DIR, "tfidf_vectorizer.pkl"))
        
        metrics_path = os.path.join(RFDATA_DIR, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        else:
            metrics = {}
        
        X_all = vectorizer.transform(df["enhanced_corpus"])
        
        return {
            'df': df,
            'rf_model': rf_model,
            'vectorizer': vectorizer,
            'X_all': X_all,
            'metrics': metrics,
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
def gaussian_calorie_score(menu_cal, max_cal):
    """
    Dummy function: kalori sekarang dikontrol via hard filter (bukan scoring).
    Supaya pipeline lama tetap jalan, kita kembalikan 1.0 untuk semua menu.
    """
    return 1.0



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
    
    enhanced_query = preprocess_menu(user_query)
    X_user = vectorizer.transform([enhanced_query])
    sim = cosine_similarity(X_user, X_all)[0]
    sim_norm = sim / sim.max() if sim.max() > 0 else sim
    
    cal_scores = df["Kalori"].apply(
        lambda c: gaussian_calorie_score(c, target_calorie)
    ).values
    
    karbo_scores = df["Karbo_List"].apply(
        lambda kl: karbo_score(kl, preferred_karbo)
    ).values
    
    keyword_boosts = df["Deskripsi_Menu"].apply(
        lambda desc: keyword_boost_score(desc, user_query)
    ).values
    
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
    """Hybrid recommender dengan hard-calorie filter"""

    df = engine['df']
    rf_model = engine['rf_model']
    vectorizer = engine['vectorizer']
    X_all = engine['X_all']

    # -------------------------
    # 🔥 HARD FILTER KALORI
    # -------------------------
    if target_calorie is not None:
        allowed_idx = df[df["Kalori"] <= target_calorie].index
    else:
        allowed_idx = df.index

    df_filtered = df.loc[allowed_idx].reset_index(drop=True)
    X_all_filtered = X_all[allowed_idx]

    # Jika seluruh menu terfilter habis
    if len(df_filtered) == 0:
        return pd.DataFrame()

    # Safety untuk list preferensi
    if preferred_categories is None:
        preferred_categories = []
    if preferred_karbo is None:
        preferred_karbo = []

    # -------------------------
    # HITUNG MCCBF (pakai df_filtered)
    # -------------------------
    mccbf, mccbf_components = mccbf_score(
        df_filtered, X_all_filtered, vectorizer,
        user_query=user_query,
        target_calorie=target_calorie,
        preferred_karbo=preferred_karbo,
        mode=mode
    )

    # -------------------------
    # HITUNG RF (pakai df_filtered)
    # -------------------------
    rf_scores = rf_category_scores(
        df_filtered, X_all_filtered, rf_model,
        preferred_categories=preferred_categories
    )

    # -------------------------
    # HYBRID SCORE
    # -------------------------
    hybrid = alpha * mccbf + (1 - alpha) * rf_scores

    # -------------------------
    # SUSUN DATAFRAME OUTPUT
    # -------------------------
    result = df_filtered[[
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
st.markdown("### Icel's Room Kitchen , Yellow Fit Kitchen , DietGo Kicthen- Katering Diet Personal")

metrics = engine.get('metrics', {})
st.markdown("---")

# Info sistem
with st.expander("ℹ️ Tentang Sistem Hybrid"):
    st.write(f"""
    Sistem ini menggunakan **pendekatan hybrid** yang menggabungkan:

    ### 🔵 MCCBF (Multi-Criteria Content-Based Filtering)
    - 🔢 Gaussian Calorie Scoring (sigma=30)
    - 🍚 Jaccard Similarity untuk Karbohidrat
    - 📝 TF-IDF + Bigram untuk Deskripsi
    - 🏷️ Keyword Boost System

    ### 🟢 Random Forest Classifier
    - 🎯 Kategori Prediction dengan 100 trees
    - 📊 Trained pada split: **{BEST_SPLIT.replace('_', '-')}**
    - ✅ Test Accuracy: **{metrics.get('Test_Accuracy', 0):.2%}**
    - 📈 Test F1-Score: **{metrics.get('Test_F1', 0):.4f}**

    ### ⚖️ Hybrid Blending (Optimal Configuration)
    - **Alpha (α)**: {ALPHA_BEST} (Best from evaluation)
    - **Formula**: `Hybrid Score = {ALPHA_BEST} × MCCBF + {1-ALPHA_BEST} × RF`
    - **Mode Default**: {MODE_BEST} (Best from evaluation)

    ### 🎨 Mode Flexibility
    - **Seimbang**: {MODES['seimbang']['description']}
    - **Fokus Deskripsi**: {MODES['fokus_deskripsi']['description']}
    - **Fokus Kategori**: {MODES['fokus_kategori']['description']}

    **Dataset:** {len(engine['df'])} menu dari Icel's Room Kitchen, Yellow Fit Kitchen, dan DietGo Kitchen

    ### ✨ Fitur:
    - 🔧 Auto-cleaning Karbohidrat
    - 🎯 Smart Normalisasi
    - ✅ Highlight Matching
    """)


# Check engine status
if engine['status'] == 'error':
    st.error(f"❌ Gagal load model: {engine['error']}")
    st.info(f"💡 Pastikan model sudah di-train dengan script 1_train_rf_tfidf.py untuk split {BEST_SPLIT}")
    st.stop()

df = engine['df']

# ========================================
# SIDEBAR - INPUT USER
# ========================================
st.sidebar.header("🎯 Preferensi Anda")

st.sidebar.caption(f"📊 **Dataset:** {len(df)} menu tersedia")
st.sidebar.caption(f"🏆 **Model:** {BEST_SPLIT.replace('_', '-')} (Best)")

# 1️⃣ Target Kalori
kalori_min = int(df['Kalori'].min())
kalori_max = int(df['Kalori'].max())
kalori_mean = int(df['Kalori'].mean())

col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Min", f"{kalori_min}")
col2.metric("Avg", f"{kalori_mean}")
col3.metric("Max", f"{kalori_max}")

kalori_target = st.sidebar.slider(
    "Kalori Maksimal (kcal)",
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

# 3️⃣ Karbohidrat (CLEANED)
all_karbo = set()
for lst in df["Karbo_List"]:
    if isinstance(lst, list):
        all_karbo.update(lst)

karbo_options = sorted(list(all_karbo))
st.sidebar.caption(f"🍚 **{len(karbo_options)} jenis karbohidrat tersedia**")

pilihan_karbo = st.sidebar.multiselect(
    "Pilih Sumber Karbohidrat",
    options=karbo_options,
    default=[karbo_options[0]] if karbo_options else [],
    help="MCCBF akan prioritaskan karbohidrat ini"
)
# 4️⃣ Deskripsi Query
st.sidebar.markdown("### 📝 Deskripsi Preferensi:")

# State untuk query
if "query" not in st.session_state:
    st.session_state.query = "ayam bakar pedas sehat"

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🟢 Rendah Lemak"):
        st.session_state.query = "rendah lemak sehat"
    if st.button("🟢 Tinggi Protein"):
        st.session_state.query = "tinggi protein"
    if st.button("🟢 Diet"):
        st.session_state.query = "diet sehat"

with col2:
    if st.button("🔴 Tanpa Santan"):
        st.session_state.query = "tanpa santan"
    if st.button("🔴 Tidak Pedas"):
        st.session_state.query = "tidak pedas"
    if st.button("🔴 Kukus"):
        st.session_state.query = "kukus sehat"

# Input manual user
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
    index=0,
    help=f"Mode bobot untuk MCCBF component\n\n" + 
         "\n".join([f"• {k}: {v['description']}" for k, v in MODES.items()])
)

with st.sidebar.expander("📊 Lihat Bobot Mode"):
    selected_mode = MODES[mode]
    st.write("**Weight Distribution:**")
    st.write(f"• Similarity: {selected_mode['w_similarity']*100:.0f}%")
    st.write(f"• Kalori: {selected_mode['w_kalori']*100:.0f}%")
    st.write(f"• Karbo: {selected_mode['w_karbo']*100:.0f}%")
    st.write(f"• Keyword: {selected_mode['w_keyword']*100:.0f}%")


AUTO_ALPHA = {
    "seimbang": 0.70,
    "fokus_deskripsi": 0.65,
    "fokus_kategori": 0.60
}

alpha = AUTO_ALPHA.get(mode, ALPHA_BEST)


top_n = st.sidebar.slider(
    "Jumlah Rekomendasi",
    min_value=3,
    max_value=20,
    value=10
)

show_debug = st.sidebar.checkbox("🔍 Mode Debug")

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
             
            if pilihan_karbo:
                st.caption(f"🍚 **Karbohidrat dipilih:** {', '.join([k.title() for k in pilihan_karbo])}")
            
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
                            
                            karbo_display = format_karbo_display(row['Karbo_List'], pilihan_karbo)
                            st.markdown(f"🍚 **Karbohidrat:** {karbo_display}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            
                            if show_debug:
                                with st.expander("🔍 Score Breakdown"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**MCCBF Components:**")
                                        st.write(f"• Similarity: {row['mccbf_similarity']:.4f}")
                                        st.write(f"• Calorie: {row['mccbf_calorie']:.4f}")
                                        st.write(f"• Karbo: {row['mccbf_karbo']:.4f}")
                                        st.write(f"• Keyword: {row['mccbf_keyword']:.4f}")
                                    
                                    with col2:
                                        st.markdown("**Calculation:**")
                                        st.write(f"• MCCBF Total: {row['mccbf_score']:.4f}")
                                        st.write(f"• RF Probability: {row['rf_score']:.4f}")
                                        st.write(f"• α × MCCBF: {alpha * row['mccbf_score']:.4f}")
                                        st.write(f"• (1-α) × RF: {(1-alpha) * row['rf_score']:.4f}")
                                        st.write(f"• **Final: {row['hybrid_score']:.4f}**")
                        
                        st.markdown("---")
                
                # Export recommendations
                st.markdown("### 📥 Export Rekomendasi")
                
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"rekomendasi_hybrid_{mode}_alpha{alpha}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                with st.expander("📊 Statistik Rekomendasi"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Hybrid Score:**")
                        st.write(f"• Mean: {recommendations['hybrid_score'].mean():.4f}")
                        st.write(f"• Std: {recommendations['hybrid_score'].std():.4f}")
                        st.write(f"• Min: {recommendations['hybrid_score'].min():.4f}")
                        st.write(f"• Max: {recommendations['hybrid_score'].max():.4f}")
                    
                    with col2:
                        st.markdown("**Kalori:**")
                        st.write(f"• Mean: {recommendations['Kalori'].mean():.1f} kcal")
                        st.write(f"• Std: {recommendations['Kalori'].std():.1f}")
                        st.write(f"• Min: {recommendations['Kalori'].min():.0f}")
                        st.write(f"• Max: {recommendations['Kalori'].max():.0f}")
                    
                    with col3:
                        st.markdown("**Kategori Distribution:**")
                        kategori_counts = recommendations['Kategori'].value_counts()
                        for kat, count in kategori_counts.items():
                            st.write(f"• {kat.title()}: {count}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if show_debug:
                import traceback
                st.code(traceback.format_exc())

else:
    # Landing page
    st.info("👈 **Silakan atur preferensi Anda di sidebar dan klik 'Dapatkan Rekomendasi'**")
    
    st.markdown("### 📋 Preview Dataset")
    
    sample_df = df[['Nama_Menu', 'Kategori', 'Kalori', 'Karbo_List', 'Deskripsi_Menu']].head(10)
    
    display_df = sample_df.copy()
    display_df['Karbo_List'] = display_df['Karbo_List'].apply(
        lambda x: ', '.join([k.title() for k in x]) if x else '-'
    )
    display_df['Deskripsi_Menu'] = display_df['Deskripsi_Menu'].apply(
        lambda x: (str(x)[:80] + '...') if pd.notna(x) and len(str(x)) > 80 else str(x)
    )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nama_Menu": st.column_config.TextColumn("Menu", width="medium"),
            "Kategori": st.column_config.TextColumn("Kategori", width="small"),
            "Kalori": st.column_config.NumberColumn("Kalori (kcal)", width="small"),
            "Karbo_List": st.column_config.TextColumn("Karbohidrat", width="medium"),
            "Deskripsi_Menu": st.column_config.TextColumn("Deskripsi", width="large")
        }
    )
    
    # Quick stats
    st.markdown("### 📈 Statistik Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Menu", len(df))
    with col2:
        st.metric("Kategori", df['Kategori'].nunique())
    with col3:
        st.metric("Jenis Karbo", len(karbo_options))
    with col4:
        avg_cal = df['Kalori'].mean()
        st.metric("Avg Kalori", f"{avg_cal:.0f}")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Kategori:**")
        kategori_dist = df['Kategori'].value_counts()
        st.bar_chart(kategori_dist)
    
    with col2:
        st.markdown("**Distribusi Kalori:**")
        st.line_chart(df['Kalori'].sort_values().reset_index(drop=True))

# ========================================
# FOOTER
# ========================================
st.markdown("""
<hr>
<div style='text-align:center; padding: 25px; color:#888;'>
    <h4 style='margin-bottom: 5px;'>🥗 Sistem Rekomendasi Menu Diet Sehat - HYBRID</h4>
    <p style='margin: 0;'>Serevina — Katering Diet Personal</p>
    <p style='font-size: 0.85em; margin: 8px 0 0 0;'>
        Powered by MCCBF + Random Forest<br>
        Split Model Terbaik: 70-30 | Alpha: 0.7 | Mode: Seimbang
    </p>
    <p style='font-size: 0.7em; margin-top: 10px; color:#aaa;'>
        © 2025 Sistem Rekomendasi Hybrid | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
