import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from utils import (
    MODES, preprocess_menu, clean_karbo_list, format_karbo_display,
    compute_mccbf_scores, rf_category_scores
)

# ======================================== 
# PAGE CONFIG
# ======================================== 
st.set_page_config(
    page_title="Menu Diet Rekomendasi",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================== 
# STYLING
# ======================================== 
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .main { padding: 1rem; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 8px; }
    .recommendation-card { 
        background: white; 
        border-left: 4px solid #4CAF50; 
        padding: 1rem; 
        margin: 0.5rem 0;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rank-badge { 
        background: #4CAF50; 
        color: white; 
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================== 
# MODEL LOADING
# ======================================== 
BEST_SPLIT = "70_30"
ALPHA_BEST = 0.7
MODE_BEST = "seimbang"
DATASET_PATH = "dataset/dataset450_indonesia.csv"
RFDATA_DIR = f"rftrain/{BEST_SPLIT}"

@st.cache_resource
def load_hybrid_engine():
    try:
        df = pd.read_csv(DATASET_PATH)
        karbo_col = "Sumber_Karbohidrat" if "Sumber_Karbohidrat" in df.columns else "Karbo_List"
        if karbo_col in df.columns:
            df["Karbo_List"] = df[karbo_col].apply(clean_karbo_list)
        else:
            df["Karbo_List"] = [[] for _ in range(len(df))]
        
        df["enhanced_corpus"] = df["Nama_Menu"].apply(preprocess_menu)
        rf_model = joblib.load(os.path.join(RFDATA_DIR, "rf_model.pkl"))
        vectorizer = joblib.load(os.path.join(RFDATA_DIR, "tfidf_vectorizer.pkl"))
        metrics_path = os.path.join(RFDATA_DIR, "metrics_summary.csv")
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict() if os.path.exists(metrics_path) else {}
        X_all = vectorizer.transform(df["enhanced_corpus"])
        
        return {
            'df': df, 'rf_model': rf_model, 'vectorizer': vectorizer,
            'X_all': X_all, 'metrics': metrics, 'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def recommend_hybrid(engine, user_query, target_calorie=None, preferred_categories=None,
                     preferred_karbo=None, mode="seimbang", alpha=0.7, top_n=10):
    df = engine['df']
    rf_model = engine['rf_model']
    vectorizer = engine['vectorizer']
    X_all = engine['X_all']
    
    allowed_idx = df[df["Kalori"] <= target_calorie].index if target_calorie else df.index
    df_filtered = df.loc[allowed_idx].reset_index(drop=True)
    X_all_filtered = X_all[allowed_idx]
    
    if len(df_filtered) == 0:
        return pd.DataFrame()
    
    preferred_categories = preferred_categories or []
    preferred_karbo = preferred_karbo or []
    
    mccbf, mccbf_components = compute_mccbf_scores(
        df_filtered, X_all_filtered, vectorizer, user_query=user_query,
        target_calorie=target_calorie, preferred_karbo=preferred_karbo, mode=mode
    )
    rf_scores = rf_category_scores(df_filtered, X_all_filtered, rf_model, preferred_categories=preferred_categories)
    hybrid = alpha * mccbf + (1 - alpha) * rf_scores
    
    result = df_filtered[['Nama_Menu', 'Kategori', 'Kalori', 'Karbo_List', 'Deskripsi_Menu']].copy()
    result["hybrid_score"] = hybrid
    result["mccbf_score"] = mccbf
    result["rf_score"] = rf_scores
    result["mccbf_similarity"] = mccbf_components['similarity']
    result["mccbf_calorie"] = mccbf_components['calorie']
    result["mccbf_karbo"] = mccbf_components['karbo']
    
    return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True).head(top_n)

# ======================================== 
# INITIALIZE
# ======================================== 
engine = load_hybrid_engine()

if engine['status'] == 'error':
    st.error(f"❌ Model error: {engine['error']}")
    st.stop()

df = engine['df']

# ======================================== 
# HEADER
# ======================================== 
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🥗 Menu Diet Sehat</h1>
        <p style="color: #666;">Temukan menu terbaik sesuai preferensi Anda</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# ======================================== 
# MAIN INPUT SECTION
# ======================================== 
col1, col2, col3 = st.columns([2, 2, 1], gap="large")

with col1:
    st.markdown("### 📝 Deskripsi")
    preset_queries = {
        "🟢 Rendah Lemak": "rendah lemak sehat",
        "🟢 Tinggi Protein": "tinggi protein",
        "🔴 Tanpa Santan": "tanpa santan",
        "🔴 Kukus": "kukus sehat"
    }
    
    col_a, col_b = st.columns(2)
    with col_a:
        for label, query in list(preset_queries.items())[:2]:
            if st.button(label, use_container_width=True):
                st.session_state.query = query
    with col_b:
        for label, query in list(preset_queries.items())[2:]:
            if st.button(label, use_container_width=True):
                st.session_state.query = query
    
    if "query" not in st.session_state:
        st.session_state.query = "sehat lezat"
    
    user_query = st.text_input(
        "Atau ketik sendiri:", 
        value=st.session_state.query,
        placeholder="misal: ayam bakar pedas"
    )

with col2:
    st.markdown("### 🔥 Kalori Maksimal")
    kalori_min, kalori_max = int(df['Kalori'].min()), int(df['Kalori'].max())
    kalori_mean = int(df['Kalori'].mean())
    kalori_target = st.slider(
        "kcal",
        min_value=kalori_min,
        max_value=kalori_max,
        value=kalori_mean,
        step=5,
        label_visibility="collapsed"
    )
    st.caption(f"Range: {kalori_min} - {kalori_max} kcal")

with col3:
    st.markdown("### 🍖 Kategori")
    kategori_options = sorted(df['Kategori'].dropna().unique().tolist())
    kategori_pref = st.multiselect(
        "Pilih kategori",
        options=kategori_options,
        default=[kategori_options[0]] if kategori_options else [],
        label_visibility="collapsed",
        max_selections=3
    )

# ======================================== 
# KARBOHIDRAT SECTION
# ======================================== 
st.markdown("---")
st.markdown("### 🥔 Sumber Karbohidrat")

all_karbo = set()
for lst in df["Karbo_List"]:
    if isinstance(lst, list):
        all_karbo.update(lst)
karbo_options = sorted(list(all_karbo))

cols = st.columns(min(5, len(karbo_options)))
pilihan_karbo = []

for idx, karbo in enumerate(sorted(karbo_options)):
    with cols[idx % len(cols)]:
        if st.checkbox(karbo.title(), key=f"karbo_{karbo}"):
            pilihan_karbo.append(karbo)

# ======================================== 
# ADVANCED SETTINGS
# ======================================== 
with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Mode Bobot:",
            options=["seimbang", "fokus_deskripsi", "fokus_kategori"],
            captions=[MODES[m]['description'] for m in ["seimbang", "fokus_deskripsi", "fokus_kategori"]]
        )
    
    with col2:
        show_debug = st.checkbox("🔍 Debug Info")
    
    top_n = st.slider("Jumlah Rekomendasi:", 3, 20, 10)

# ======================================== 
# GENERATE BUTTON
# ======================================== 
st.markdown("---")
generate_btn = st.button("🚀 Cari Menu", type="primary", use_container_width=True)

# ======================================== 
# RESULTS
# ======================================== 
if generate_btn:
    AUTO_ALPHA = {"seimbang": 0.70, "fokus_deskripsi": 0.65, "fokus_kategori": 0.60}
    alpha = AUTO_ALPHA.get(mode, ALPHA_BEST)
    
    with st.spinner("Mencari menu terbaik..."):
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
    
    if recommendations.empty:
        st.warning("⚠️ Tidak ada menu yang sesuai kriteria. Coba ubah filter.")
    else:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hasil", len(recommendations))
        with col2:
            st.metric("Rata-rata Kalori", f"{recommendations['Kalori'].mean():.0f} kcal")
        with col3:
            st.metric("Min Kalori", f"{recommendations['Kalori'].min()} kcal")
        with col4:
            st.metric("Max Kalori", f"{recommendations['Kalori'].max()} kcal")
        
        st.markdown("---")
        st.markdown(f"### 🏆 Hasil Pencarian ({len(recommendations)} menu)")
        
        # Results
        for idx, row in recommendations.iterrows():
            with st.container():
                col_rank, col_content = st.columns([0.5, 9.5])
                
                with col_rank:
                    st.markdown(f"<div class='rank-badge'>#{idx + 1}</div>", unsafe_allow_html=True)
                
                with col_content:
                    st.markdown(f"**{row['Nama_Menu'].title()}**")
                    
                    info_cols = st.columns([2, 2])
                    with info_cols[0]:
                        st.caption(f"🍖 {row['Kategori'].title()}")
                    with info_cols[1]:
                        st.caption(f"🔥 {row['Kalori']} kcal")
                    
                    if row['Deskripsi_Menu'] and str(row['Deskripsi_Menu']) != 'nan':
                        st.caption(f"{row['Deskripsi_Menu'].capitalize()}")
                    
                    karbo_display = format_karbo_display(row['Karbo_List'], pilihan_karbo)
                    st.caption(f"🥔 {karbo_display}")
                    
                    if show_debug:
                        with st.expander("Score Details", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"MCCBF: {row['mccbf_score']:.4f}")
                                st.write(f"  • Similarity: {row['mccbf_similarity']:.4f}")
                                st.write(f"  • Calorie: {row['mccbf_calorie']:.4f}")
                            with col2:
                                st.write(f"RF Score: {row['rf_score']:.4f}")
                                st.write(f"Hybrid: {row['hybrid_score']:.4f}")
                
                st.markdown("")
        
        # Download
        st.markdown("---")
        csv = recommendations.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"rekomendasi_{mode}.csv",
            mime="text/csv"
        )

else:
    # Landing
    st.info("👆 Atur preferensi Anda dan klik 'Cari Menu'")
    
    with st.expander("📋 Preview 10 Menu Terbaru", expanded=False):
        sample_df = df[['Nama_Menu', 'Kategori', 'Kalori', 'Deskripsi_Menu']].head(10).copy()
        sample_df['Kalori'] = sample_df['Kalori'].astype(int)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

# ======================================== 
# FOOTER
# ======================================== 
st.markdown("""
    <hr>
    <div style="text-align: center; color: #999; font-size: 12px; padding: 2rem 0;">
        <p>🥗 Menu Diet Sehat | Icel's Room • Yellow Fit • DietGo Kitchen</p>
        <p>Powered by Hybrid Recommendation System</p>
    </div>
""", unsafe_allow_html=True)