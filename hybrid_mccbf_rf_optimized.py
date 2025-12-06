import os
import math
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1. CONFIG
# ==========================

DATASET_PATH = "dataset/dataset450_final.csv"
RFDATA_DIR = "rfdata"
OUTPUT_DIR = "datahybrid"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alpha blending: MCCBF vs RF
ALPHA = 0.7  # 70% MCCBF, 30% RF (bisa di-tune)

# Mode weights (dari grid search MCCBF engine)
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

# ==========================
# 2. PREPROCESSING FUNCTION (Recreate from training)
# ==========================

def extract_keywords(text):
    """
    Recreate preprocessing function dari training RF
    """
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


# ==========================
# 3. LOAD ASSETS
# ==========================

def load_assets():
    """Load dataset, RF model, dan vectorizer"""
    
    print(f"📂 Loading assets from {RFDATA_DIR}/...")
    
    # Dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded dataset: {len(df)} rows")
    
    # Parse Karbo_List
    # SAFE PARSING UNTUK LIST BARU HASIL CLEANER
    if "Karbo_List" in df.columns:
        def parse_list(x):
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []
        df["Karbo_List"] = df["Karbo_List"].apply(parse_list)
    else:
        df["Karbo_List"] = [[] for _ in range(len(df))]

    
    # RF Model & Vectorizer
    rf_model = joblib.load(os.path.join(RFDATA_DIR, "rf_model_production.pkl"))
    vectorizer = joblib.load(os.path.join(RFDATA_DIR, "tfidf_vectorizer_production.pkl"))
    print("✅ Loaded RF model & vectorizer")
    
    # Generate enhanced_corpus
    print("⏳ Preprocessing menu corpus...")
    df["enhanced_corpus"] = df["Nama_Menu"].apply(preprocess_menu)
    
    # Precompute TF-IDF untuk semua menu
    X_all = vectorizer.transform(df["enhanced_corpus"])
    print(f"✅ Generated TF-IDF matrix: {X_all.shape}")
    
    return df, rf_model, vectorizer, X_all


# ==========================
# 4. MCCBF SCORING COMPONENTS
# ==========================

def gaussian_calorie_score(menu_cal, target_cal, sigma=30):
    """Gaussian scoring untuk kalori (OPTIMAL: sigma=30)"""
    if target_cal is None or target_cal <= 0:
        return 1.0
    
    try:
        diff = abs(float(target_cal) - float(menu_cal))
        score = math.exp(-(diff ** 2) / (2 * (sigma ** 2)))
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def karbo_score(menu_karbo_list, preferred_karbo):
    """Jaccard similarity untuk karbo preference"""
    if not preferred_karbo:
        return 1.0
    
    if not menu_karbo_list:
        return 0.5
    
    menu_set = {k.lower().strip() for k in menu_karbo_list}
    pref_set = {k.lower().strip() for k in preferred_karbo}
    
    inter = len(menu_set & pref_set)
    union = len(menu_set | pref_set)
    
    if union == 0:
        return 0.5
    
    return inter / union


def keyword_boost_score(menu_desc, user_query):
    """Keyword boost (OPTIMIZED: 0.10 per keyword, max 0.35)"""
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
    """Hitung MCCBF score dengan mode weights"""
    
    weights = MODES.get(mode, MODES['seimbang'])
    
    # 1) Similarity score (TF-IDF cosine)
    enhanced_query = preprocess_menu(user_query)
    X_user = vectorizer.transform([enhanced_query])
    sim = cosine_similarity(X_user, X_all)[0]
    
    # Normalize to [0, 1]
    if sim.max() > 0:
        sim_norm = sim / sim.max()
    else:
        sim_norm = sim
    
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


# ==========================
# 5. RF CATEGORY SCORING
# ==========================

def rf_category_scores(df, X_all, rf_model, preferred_categories=None):
    """RF scoring berdasarkan kategori preference"""
    
    proba = rf_model.predict_proba(X_all)
    classes = list(rf_model.classes_)
    
    if not preferred_categories:
        return proba.max(axis=1)
    
    idx_list = [classes.index(kat) for kat in preferred_categories if kat in classes]
    
    if not idx_list:
        return proba.max(axis=1)
    
    scores = proba[:, idx_list].max(axis=1)
    
    return scores


# ==========================
# 6. HYBRID RECOMMENDER
# ==========================

def recommend_hybrid(
    user_query,
    target_calorie=None,
    preferred_categories=None,
    preferred_karbo=None,
    mode="seimbang",
    alpha=ALPHA,
    top_n=10,
    min_confidence=0.0,
    save_results=True
):
    """
    Hybrid Recommender: MCCBF + RF
    
    Parameters
    ----------
    user_query : str
        Query deskripsi menu
    target_calorie : int/float/None
        Target kalori
    preferred_categories : list[str]/None
        Kategori preferensi
    preferred_karbo : list[str]/None
        Karbo preferensi
    mode : str
        Mode MCCBF: "seimbang", "fokus_deskripsi", "fokus_kategori"
    alpha : float
        Blending weight (0-1)
    top_n : int
        Jumlah rekomendasi
    min_confidence : float
        Filter minimum hybrid score
    save_results : bool
        Save ke CSV/grafik
    
    Returns
    -------
    DataFrame with recommendations + scores
    """
    
    if preferred_categories is None:
        preferred_categories = []
    if preferred_karbo is None:
        preferred_karbo = []
    
    # Load assets
    df, rf_model, vectorizer, X_all = load_assets()
    
    # 1) MCCBF Score
    print(f"\n⏳ Computing MCCBF scores (mode: {mode})...")
    mccbf, mccbf_components = mccbf_score(
        df, X_all, vectorizer,
        user_query=user_query,
        target_calorie=target_calorie,
        preferred_karbo=preferred_karbo,
        mode=mode
    )
    
    # 2) RF Score
    print("⏳ Computing RF category scores...")
    rf_scores = rf_category_scores(
        df, X_all, rf_model,
        preferred_categories=preferred_categories
    )
    
    # 3) Hybrid Score
    print(f"⏳ Blending scores (alpha={alpha})...")
    hybrid = alpha * mccbf + (1 - alpha) * rf_scores
    
    # 4) Build result dataframe
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
    
    # 5) Filter & Sort
    result = result[result["hybrid_score"] >= min_confidence]
    result = result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    
    top_results = result.head(top_n)
    
    # 6) Save results if requested
    if save_results:
        save_hybrid_results(top_results, user_query, mode, alpha)
    
    return top_results


# ==========================
# 7. SAVE RESULTS
# ==========================

def save_hybrid_results(results_df, query, mode, alpha):
    """Save hasil rekomendasi ke CSV dan visualisasi"""
    
    print(f"\n💾 Saving results to {OUTPUT_DIR}/...")
    
    # 1) Save full results
    results_df.to_csv(f"{OUTPUT_DIR}/recommendations.csv", index=False)
    print(f"✅ Saved: {OUTPUT_DIR}/recommendations.csv")
    
    # 2) Save summary
    summary = {
        'Query': [query],
        'Mode': [mode],
        'Alpha': [alpha],
        'Top_Score': [results_df['hybrid_score'].iloc[0] if len(results_df) > 0 else 0],
        'Avg_Score': [results_df['hybrid_score'].mean()],
        'Num_Results': [len(results_df)]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTPUT_DIR}/recommendation_summary.csv", index=False)
    print(f"✅ Saved: {OUTPUT_DIR}/recommendation_summary.csv")
    
    # 3) Visualizations
    if len(results_df) > 0:
        create_visualizations(results_df, query, mode, alpha)


def create_visualizations(results_df, query, mode, alpha):
    """Create grafik visualisasi"""
    
    top_n = min(10, len(results_df))
    top_df = results_df.head(top_n)
    
    # Figure 1: Score Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hybrid vs MCCBF vs RF
    ax1 = axes[0, 0]
    x_pos = np.arange(top_n)
    width = 0.25
    
    ax1.barh(x_pos - width, top_df['hybrid_score'], width, 
             label='Hybrid', color='purple', alpha=0.8)
    ax1.barh(x_pos, top_df['mccbf_score'], width,
             label='MCCBF', color='blue', alpha=0.8)
    ax1.barh(x_pos + width, top_df['rf_score'], width,
             label='RF', color='green', alpha=0.8)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([f"{i+1}. {name[:25]}..." for i, name in enumerate(top_df['Nama_Menu'])])
    ax1.set_xlabel('Score', fontsize=11)
    ax1.set_title(f'Top {top_n} Recommendations: Score Comparison\nAlpha={alpha} | Mode={mode}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: MCCBF Component Breakdown (Top 5)
    ax2 = axes[0, 1]
    top5 = top_df.head(5)
    components = ['mccbf_similarity', 'mccbf_calorie', 'mccbf_karbo', 'mccbf_keyword']
    comp_labels = ['Similarity', 'Calorie', 'Karbo', 'Keyword']
    
    x_pos2 = np.arange(len(top5))
    width2 = 0.2
    
    for i, (comp, label) in enumerate(zip(components, comp_labels)):
        offset = (i - 1.5) * width2
        ax2.bar(x_pos2 + offset, top5[comp], width2, label=label, alpha=0.8)
    
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels([f"#{i+1}" for i in range(len(top5))])
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('MCCBF Component Breakdown (Top 5)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Kategori Distribution
    ax3 = axes[1, 0]
    kategori_counts = top_df['Kategori'].value_counts()
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(kategori_counts)))
    
    ax3.pie(kategori_counts.values, labels=kategori_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax3.set_title('Kategori Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Kalori Distribution
    ax4 = axes[1, 1]
    ax4.hist(top_df['Kalori'], bins=10, color='coral', alpha=0.7, edgecolor='black')
    ax4.axvline(top_df['Kalori'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {top_df["Kalori"].mean():.0f}')
    ax4.set_xlabel('Kalori (kcal)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Kalori Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hybrid_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/hybrid_visualization.png")
    
    # Figure 2: Score Heatmap
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    score_matrix = top_df[['mccbf_similarity', 'mccbf_calorie', 
                            'mccbf_karbo', 'mccbf_keyword', 
                            'rf_score', 'hybrid_score']].T
    
    sns.heatmap(score_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f"#{i+1}" for i in range(len(top_df))],
                yticklabels=['Similarity', 'Calorie', 'Karbo', 'Keyword', 'RF', 'Hybrid'],
                cbar_kws={'label': 'Score'})
    
    ax.set_title(f'Score Heatmap - Top {len(top_df)} Recommendations', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/score_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/score_heatmap.png")


# ==========================
# 8. DEMO & CLI
# ==========================

if __name__ == "__main__":
    
    print("="*70)
    print("🍽️  HYBRID RECOMMENDER: MCCBF + RF")
    print("="*70)
    
    # Example 1: Simple query
    print("\n📌 Example 1: Basic Query")
    print("-"*70)
    
    recs1 = recommend_hybrid(
        user_query="ayam bakar pedas sehat",
        target_calorie=400,
        preferred_categories=["ayam"],
        preferred_karbo=["nasi merah"],
        mode="seimbang",
        top_n=5,
        save_results=False  # Don't save untuk example
    )
    
    print("\nTop 5 Recommendations:")
    print(recs1[["Nama_Menu", "Kategori", "Kalori", "hybrid_score"]].to_string(index=False))
    
    # Example 2: Different mode
    print("\n\n📌 Example 2: Fokus Deskripsi Mode")
    print("-"*70)
    
    recs2 = recommend_hybrid(
        user_query="ikan goreng crispy rendah kalori",
        target_calorie=300,
        preferred_categories=["ikan"],
        mode="fokus_deskripsi",
        top_n=5,
        save_results=False
    )
    
    print("\nTop 5 Recommendations:")
    print(recs2[["Nama_Menu", "Kategori", "Kalori", "hybrid_score"]].to_string(index=False))
    
    # Example 3: Full demo with save
    print("\n\n📌 Example 3: Full Demo (Saved to datahybrid/)")
    print("-"*70)
    
    recs3 = recommend_hybrid(
        user_query="sapi rendang pedas enak",
        target_calorie=500,
        preferred_categories=["sapi"],
        preferred_karbo=["nasi putih"],
        mode="seimbang",
        alpha=0.7,
        top_n=10,
        save_results=True  # Save results!
    )
    
    print("\nTop 10 Recommendations:")
    print(recs3[["Nama_Menu", "Kategori", "Kalori", "hybrid_score"]].to_string(index=False))
    
    print("\n\n📊 Score Breakdown (Top 3):")
    print("-"*70)
    for idx in range(min(3, len(recs3))):
        row = recs3.iloc[idx]
        print(f"\n#{idx+1}: {row['Nama_Menu']}")
        print(f"   Hybrid Score : {row['hybrid_score']:.4f}")
        print(f"   ├─ MCCBF     : {row['mccbf_score']:.4f}")
        print(f"   │  ├─ Similarity : {row['mccbf_similarity']:.4f}")
        print(f"   │  ├─ Calorie    : {row['mccbf_calorie']:.4f}")
        print(f"   │  ├─ Karbo      : {row['mccbf_karbo']:.4f}")
        print(f"   │  └─ Keyword    : {row['mccbf_keyword']:.4f}")
        print(f"   └─ RF        : {row['rf_score']:.4f}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED!")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")
    print("="*70)