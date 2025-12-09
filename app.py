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
# BEST MODEL CONFIGURATION (SPLIT 70-30)
# ========================================
BEST_SPLIT = "70_30"
ALPHA_BEST = 0.7
MODE_BEST = "seimbang"

DATASET_PATH = "dataset/dataset450_indonesia_fixed.csv"
RFDATA_DIR = f"rftrain/{BEST_SPLIT}"

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
    if not karbo_list or len(karbo_list) == 0:
        return "_Tidak ada data_"
    
    user_prefs_lower = [k.lower() for k in user_prefs] if user_prefs else []
    karbo_display = []
    
    for item in karbo_list:
        item_title = item.title()
        is_match = any(pref in item.lower() or item.lower() in pref for pref in user_prefs_lower)
        
        if is_match:
            karbo_display.append(f"`{item_title}` ✅")
        else:
            karbo_display.append(f"`{item_title}`")
    
    return ' '.join(karbo_display)

# ========================================
# PREPROCESSING FUNCTIONS
# ========================================
def extract_keywords(text):
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
    corpus = str(menu_name).lower()
    semantic = extract_keywords(menu_name)
    return corpus + " " + semantic

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_hybrid_engine():
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
        return {'status': 'error', 'error': str(e)}

# ========================================
# SCORING FUNCTIONS
# ========================================
def gaussian_calorie_score(menu_cal, target_cal, sigma=10):
    if target_cal is None or target_cal <= 0:
        return 1.0
    try:
        diff = abs(float(target_cal) - float(menu_cal))
        score = math.exp(-(diff ** 2) / (2 * (sigma ** 2)))
        return max(0.0, min(1.0, score))
    except:
        return 0.5

def karbo_score(menu_karbo_list, preferred_karbo):
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
    if not user_query or pd.isna(user_query):
        return 0.0
    
    important_keywords = {
        'pedas', 'manis', 'gurih', 'asam', 'asin',
        'panggang', 'bakar', 'goreng', 'kukus', 'rebus', 'tumis',
        'crispy', 'grill', 'renyah', 'rendah', 'tinggi', 'tanpa',
        'kuah', 'kering', 'bening', 'lembut', 'empuk', 'segar',
        'protein', 'santan', 'lemak', 'minyak', 'teriyaki', 'balado',
        'sambal', 'woku', 'korea', 'yakiniku', 'bulgogi', 'curry',
        'soto', 'rawon', 'sehat', 'diet', 'organik', 'fit', 'healthy'
    }
    
    menu_desc = str(menu_desc).lower() if not pd.isna(menu_desc) else ""
    user_kw = set(str(user_query).lower().split())
    menu_kw = set(menu_desc.split())
    
    matched = user_kw & menu_kw & important_keywords
    return min(0.35, len(matched) * 0.10)

def mccbf_score(df, X_all, vectorizer, user_query, target_calorie=None, preferred_karbo=None, mode="seimbang"):
    weights = MODES.get(mode, MODES['seimbang'])
    
    enhanced_query = preprocess_menu(user_query)
    X_user = vectorizer.transform([enhanced_query])
    sim = cosine_similarity(X_user, X_all)[0]
    sim_norm = sim / sim.max() if sim.max() > 0 else sim
    
    cal_scores = df["Kalori"].apply(lambda c: gaussian_calorie_score(c, target_calorie)).values
    karbo_scores = df["Karbo_List"].apply(lambda kl: karbo_score(kl, preferred_karbo)).values
    keyword_boosts = df["Deskripsi_Menu"].apply(lambda desc: keyword_boost_score(desc, user_query)).values
    
    mccbf = (weights['w_similarity'] * sim_norm + weights['w_kalori'] * cal_scores + 
             weights['w_karbo'] * karbo_scores + weights['w_keyword'] * keyword_boosts)
    
    return mccbf, {
        'similarity': sim_norm,
        'calorie': cal_scores,
        'karbo': karbo_scores,
        'keyword': keyword_boosts
    }

def rf_category_scores(df, X_all, rf_model, preferred_categories=None):
    proba = rf_model.predict_proba(X_all)
    classes = list(rf_model.classes_)
    
    if not preferred_categories:
        return proba.max(axis=1)
    
    idx_list = [classes.index(kat) for kat in preferred_categories if kat in classes]
    return proba[:, idx_list].max(axis=1) if idx_list else proba.max(axis=1)

def recommend_hybrid(engine, user_query, target_calorie=None, preferred_categories=None, 
                     preferred_karbo=None, mode="seimbang", alpha=0.7, top_n=10):
    df = engine['df']
    rf_model = engine['rf_model']
    vectorizer = engine['vectorizer']
    X_all = engine['X_all']

    if target_calorie is not None:
        allowed_idx = df[df["Kalori"] <= target_calorie].index
    else:
        allowed_idx = df.index

    df_filtered = df.loc[allowed_idx].reset_index(drop=True)
    X_all_filtered = X_all[allowed_idx]

    if len(df_filtered) == 0:
        return pd.DataFrame()

    if preferred_categories is None:
        preferred_categories = []
    if preferred_karbo is None:
        preferred_karbo = []

    mccbf, mccbf_components = mccbf_score(df_filtered, X_all_filtered, vectorizer,
                                           user_query, target_calorie, preferred_karbo, mode)
    rf_scores = rf_category_scores(df_filtered, X_all_filtered, rf_model, preferred_categories)
    hybrid = alpha * mccbf + (1 - alpha) * rf_scores

    result = df_filtered[['Nama_Menu', 'Kategori', 'Kalori', 'Karbo_List', 'Deskripsi_Menu']].copy()
    result["mccbf_score"] = mccbf
    result["mccbf_similarity"] = mccbf_components['similarity']
    result["mccbf_calorie"] = mccbf_components['calorie']
    result["mccbf_karbo"] = mccbf_components['karbo']
    result["mccbf_keyword"] = mccbf_components['keyword']
    result["rf_score"] = rf_scores
    result["hybrid_score"] = hybrid

    return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True).head(top_n)

# ========================================
# LOAD ENGINE & HEADER
# ========================================
engine = load_hybrid_engine()

st.title("🥗 Sistem Rekomendasi Menu Diet Sehat - HYBRID")
st.markdown("### Icel's Room Kitchen, Yellow Fit Kitchen, DietGo Kitchen")

metrics = engine.get('metrics', {})
st.markdown("---")

with st.expander("ℹ️ Tentang Sistem Hybrid"):
    st.write(f"""
    ### 🔵 MCCBF (Multi-Criteria Content-Based Filtering)
    - Gaussian Calorie Scoring (sigma=10)
    - Jaccard Similarity untuk Karbohidrat
    - TF-IDF + Bigram untuk Deskripsi
    - Keyword Boost System
    
    ### 🟢 Random Forest Classifier
    - 100 trees, max_depth=10
    - Split: **70-30** | Test Accuracy: **{metrics.get('Test_Accuracy', 0):.2%}**
    
    ### ⚖️ Hybrid Formula
    `Hybrid Score = 0.7 × MCCBF + 0.3 × RF`
    """)

if engine['status'] == 'error':
    st.error(f"❌ Error: {engine['error']}")
    st.stop()

df = engine['df']

# ========================================
# SIDEBAR INPUT
# ========================================
st.sidebar.header("🎯 Preferensi Anda")
st.sidebar.caption(f"📊 Dataset: {len(df)} menu | 🏆 Split: 70-30")

kalori_min, kalori_max, kalori_mean = int(df['Kalori'].min()), int(df['Kalori'].max()), int(df['Kalori'].mean())
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Min", f"{kalori_min}")
col2.metric("Avg", f"{kalori_mean}")
col3.metric("Max", f"{kalori_max}")

kalori_target = st.sidebar.slider("Kalori Maksimal (kcal)", kalori_min, kalori_max, kalori_mean, 5)

kategori_options = sorted(df['Kategori'].dropna().unique().tolist())
kategori_pref = st.sidebar.multiselect("Pilih Kategori Lauk", kategori_options, 
                                       default=[kategori_options[0]] if kategori_options else [])

all_karbo = set()
for lst in df["Karbo_List"]:
    if isinstance(lst, list):
        all_karbo.update(lst)

karbo_options = sorted(list(all_karbo))
st.sidebar.caption(f"🍚 {len(karbo_options)} jenis karbohidrat")

pilihan_karbo = st.sidebar.multiselect("Pilih Sumber Karbohidrat", karbo_options,
                                       default=[karbo_options[0]] if karbo_options else [])

st.sidebar.markdown("### 📝 Deskripsi Preferensi:")
if "query" not in st.session_state:
    st.session_state.query = "ayam bakar pedas sehat"

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🟢 Rendah Lemak"):
        st.session_state.query = "rendah lemak sehat"
    if st.button("🟢 Tinggi Protein"):
        st.session_state.query = "tinggi protein"

with col2:
    if st.button("🔴 Tanpa Santan"):
        st.session_state.query = "tanpa santan"
    if st.button("🔴 Tidak Pedas"):
        st.session_state.query = "tidak pedas"

user_query = st.sidebar.text_area("Atau Ketik Manual:", st.session_state.query, height=80)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Hybrid")

mode = st.sidebar.radio("Mode MCCBF", ["seimbang", "fokus_deskripsi", "fokus_kategori"], index=0)

with st.sidebar.expander("📊 Lihat Bobot Mode"):
    selected_mode = MODES[mode]
    st.write(f"• Similarity: {selected_mode['w_similarity']*100:.0f}%")
    st.write(f"• Kalori: {selected_mode['w_kalori']*100:.0f}%")
    st.write(f"• Karbo: {selected_mode['w_karbo']*100:.0f}%")
    st.write(f"• Keyword: {selected_mode['w_keyword']*100:.0f}%")

AUTO_ALPHA = {"seimbang": 0.70, "fokus_deskripsi": 0.65, "fokus_kategori": 0.60}
alpha = AUTO_ALPHA.get(mode, ALPHA_BEST)

top_n = st.sidebar.slider("Jumlah Rekomendasi", 3, 20, 10)
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
                engine, user_query, kalori_target, kategori_pref, pilihan_karbo, mode, alpha, top_n
            )
            
            st.success("✅ Rekomendasi berhasil dibuat!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔥 Target Kalori", f"{kalori_target} kcal")
            col2.metric("🍖 Kategori (RF)", len(kategori_pref))
            col3.metric("🍚 Karbo (MCCBF)", len(pilihan_karbo))
            col4.metric("⚙️ Mode", mode.title().replace('_', ' '))
            
            st.caption(f"📝 **Query:** {user_query}")
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
                                st.write(f"📝 {row['Deskripsi_Menu']}")
                            
                            karbo_display = format_karbo_display(row['Karbo_List'], pilihan_karbo)
                            st.markdown(f"🍚 **Karbohidrat:** {karbo_display}")
                            
                            if show_debug:
                                with st.expander("🔍 Score Breakdown"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write("**MCCBF Components:**")
                                        st.write(f"• Similarity: {row['mccbf_similarity']:.4f}")
                                        st.write(f"• Calorie: {row['mccbf_calorie']:.4f}")
                                        st.write(f"• Karbo: {row['mccbf_karbo']:.4f}")
                                        st.write(f"• Keyword: {row['mccbf_keyword']:.4f}")
                                    with col_b:
                                        st.write("**Final Scores:**")
                                        st.write(f"• MCCBF: {row['mccbf_score']:.4f}")
                                        st.write(f"• RF: {row['rf_score']:.4f}")
                                        st.write(f"• Hybrid: {row['hybrid_score']:.4f}")
                        st.markdown("---")
                
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", csv, f"rekomendasi_{mode}_{alpha}.csv", "text/csv")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 **Atur preferensi dan klik 'Dapatkan Rekomendasi'**")
    st.markdown("### 📋 Preview Dataset (10 Menu)")
    
    sample_df = df[['Nama_Menu', 'Kategori', 'Kalori', 'Karbo_List', 'Deskripsi_Menu']].head(10).copy()
    sample_df['Karbo_List'] = sample_df['Karbo_List'].apply(
        lambda x: ', '.join([k.title() for k in x]) if x else '-'
    )
    
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 📈 Dataset Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Menu", len(df))
    col2.metric("Kategori", df['Kategori'].nunique())
    col3.metric("Jenis Karbo", len(karbo_options))
    col4.metric("Avg Kalori", f"{df['Kalori'].mean():.0f}")

st.markdown("---")
st.markdown("🥗 **Sistem Rekomendasi Hybrid MCCBF+RF** | Split: 70-30 | α=0.7")