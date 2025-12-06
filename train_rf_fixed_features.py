import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = "dataset/dataset450_clean_mccbf.csv"
OUTPUT_DIR = "rfdata"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Created directory: {OUTPUT_DIR}/")

# ============================================================
# 1. IMPROVED PREPROCESSING
# ============================================================

def extract_keywords(text):
    """
    Ekstrak keyword semantic, bukan literal words
    """
    text = text.lower().strip()
    
    # Mapping semantic keywords
    protein_keywords = {
        'ayam': ['ayam', 'chicken', 'poultry', 'wing', 'breast', 'drumstick'],
        'sapi': ['sapi', 'beef', 'daging', 'steak', 'bistik', 'rendang', 'rawon', 'bakso', 'iga'],
        'ikan': ['ikan', 'fish', 'dori', 'salmon', 'tuna', 'kakap', 'gurame', 'nila', 'seafood'],
        'lainnya': ['vegetarian', 'vegan', 'salad', 'kentang', 'potato', 'sayur', 'vegetable']
    }
    
    # Check presence of keywords
    features = []
    for category, keywords in protein_keywords.items():
        if any(kw in text for kw in keywords):
            features.append(f"protein_{category}")
    
    # Cooking method
    cooking = ['goreng', 'bakar', 'rebus', 'kukus', 'panggang', 'grill', 'fried', 'grilled']
    if any(method in text for method in cooking):
        features.append("cooked")
    
    # Sauce/flavor
    flavors = ['saus', 'sauce', 'bumbu', 'pedas', 'manis', 'asam', 'teriyaki', 'blackpepper']
    if any(flavor in text for flavor in flavors):
        features.append("flavored")
    
    return " ".join(features) if features else text


def create_enhanced_corpus(df):
    """
    Gabungkan original corpus dengan semantic features
    """
    df['semantic_features'] = df['Nama_Menu'].apply(extract_keywords)
    df['enhanced_corpus'] = df['corpus'] + " " + df['semantic_features']
    return df


# ============================================================
# 2. Load & Preprocess
# ============================================================

print("\n" + "="*70)
print("🔧 LOAD & PREPROCESS DATASET")
print("="*70)

df = pd.read_csv(DATASET)
df = create_enhanced_corpus(df)

print(f"\nTotal data: {len(df)}")
print(f"\nKategori distribution:")
print(df['Kategori'].value_counts())

print("\n📝 Sample enhanced corpus:")
for i in range(3):
    print(f"\n  Original: {df.iloc[i]['Nama_Menu']}")
    print(f"  Enhanced: {df.iloc[i]['enhanced_corpus']}")

# ============================================================
# 3. SIMPLE RANDOM SPLIT (untuk baseline)
# ============================================================

print("\n" + "="*70)
print("✂️  TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    df['enhanced_corpus'], 
    df['Kategori'],
    test_size=0.25,
    random_state=42,
    stratify=df['Kategori']
)

# Get indices for saving full data
train_indices = X_train.index
test_indices = X_test.index

# Create train & test dataframes
train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

print(f"\nTrain size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test size : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

print("\nTrain distribution:")
print(train_df['Kategori'].value_counts())
print("\nTest distribution:")
print(test_df['Kategori'].value_counts())

# Save train & test sets
train_df.to_csv(f"{OUTPUT_DIR}/train_set.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test_set.csv", index=False)
print(f"\n✅ Saved: {OUTPUT_DIR}/train_set.csv")
print(f"✅ Saved: {OUTPUT_DIR}/test_set.csv")

# ============================================================
# 4. TF-IDF dengan ngram yang lebih baik
# ============================================================

print("\n" + "="*70)
print("📊 TF-IDF VECTORIZATION")
print("="*70)

vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),     # Unigram + bigram
    min_df=2,
    max_df=0.8,
    sublinear_tf=True       # Log scaling
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape: {X_train_vec.shape}")
print(f"Number of features: {X_train_vec.shape[1]}")

# ============================================================
# 5. Train Model (Simplified)
# ============================================================

print("\n" + "="*70)
print("🤖 TRAINING RANDOM FOREST MODEL")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # Deeper untuk capture complexity
    min_samples_split=5,    # Lebih fleksibel
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nModel parameters:")
print(f"  n_estimators     : {rf.n_estimators}")
print(f"  max_depth        : {rf.max_depth}")
print(f"  min_samples_split: {rf.min_samples_split}")
print(f"  min_samples_leaf : {rf.min_samples_leaf}")

print("\n⏳ Training in progress...")
rf.fit(X_train_vec, y_train)
print("✅ Training completed!")

# ============================================================
# 6. Evaluate on Test Set
# ============================================================

print("\n" + "="*70)
print("📊 MODEL EVALUATION")
print("="*70)

y_train_pred = rf.predict(X_train_vec)
y_test_pred = rf.predict(X_test_vec)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\n{'Metric':<20} {'Train':<12} {'Test':<12} {'Gap'}")
print("-"*60)
print(f"{'Accuracy':<20} {train_acc:<12.4f} {test_acc:<12.4f} {train_acc-test_acc:.4f}")
print(f"{'F1-Score (macro)':<20} {train_f1:<12.4f} {test_f1:<12.4f} {train_f1-test_f1:.4f}")

# Classification Report
print("\n📋 Detailed Classification Report (Test Set):")
print("\n" + classification_report(y_test, y_test_pred))

# Save classification report
report_dict = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
print(f"✅ Saved: {OUTPUT_DIR}/classification_report.csv")

# ============================================================
# 7. Confusion Matrix
# ============================================================

print("\n" + "="*70)
print("📈 CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_test_pred)
labels = sorted(df['Kategori'].unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix\nTest Accuracy: {test_acc:.2%} | F1-Score: {test_f1:.4f}', 
          fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/confusion_matrix.png")

# ============================================================
# 8. Performance Metrics Graph
# ============================================================

print("\n" + "="*70)
print("📊 PERFORMANCE METRICS VISUALIZATION")
print("="*70)

# Create performance comparison
metrics_data = {
    'Metric': ['Accuracy', 'Accuracy', 'F1-Score', 'F1-Score'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    'Score': [train_acc, test_acc, train_f1, test_f1]
}
metrics_df = pd.DataFrame(metrics_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart comparison
x_pos = np.arange(2)
width = 0.35

axes[0].bar(x_pos - width/2, [train_acc, train_f1], width, 
            label='Train', color='skyblue', alpha=0.8)
axes[0].bar(x_pos + width/2, [test_acc, test_f1], width,
            label='Test', color='coral', alpha=0.8)
axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_title('Train vs Test Performance', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(['Accuracy', 'F1-Score'])
axes[0].legend()
axes[0].set_ylim([0, 1.1])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (train_val, test_val) in enumerate([(train_acc, test_acc), (train_f1, test_f1)]):
    axes[0].text(i - width/2, train_val + 0.02, f'{train_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    axes[0].text(i + width/2, test_val + 0.02, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=9)

# Plot 2: Per-class performance
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, labels=labels
)

x_pos2 = np.arange(len(labels))
width2 = 0.25

axes[1].bar(x_pos2 - width2, precision, width2, label='Precision', alpha=0.8)
axes[1].bar(x_pos2, recall, width2, label='Recall', alpha=0.8)
axes[1].bar(x_pos2 + width2, f1, width2, label='F1-Score', alpha=0.8)
axes[1].set_ylabel('Score', fontsize=11)
axes[1].set_title('Per-Class Performance (Test Set)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos2)
axes[1].set_xticklabels(labels)
axes[1].legend()
axes[1].set_ylim([0, 1.1])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/performance_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/performance_metrics.png")

# ============================================================
# 9. MANUAL TEST CASES (CRITICAL!)
# ============================================================

print("\n" + "="*70)
print("🧪 MANUAL TEST CASES (Unseen Data)")
print("="*70)

test_cases = [
    ("nasi ayam bakar madu", "ayam"),
    ("beef rendang pedas", "sapi"),
    ("ikan bakar rica rica", "ikan"),
    ("salad caesar vegetarian", "lainnya"),
    ("chicken beef wrap", "ayam"),
    ("sup iga sapi special", "sapi"),
    ("dori goreng tepung", "ikan"),
    ("nasi goreng seafood", "ikan"),
    ("ayam suwir pedas", "ayam"),
    ("rawon daging sapi", "sapi"),
    ("crispy fish fillet", "ikan"),
    ("kentang goreng crispy", "lainnya"),
    ("bakso sapi kuah", "sapi"),
    ("gurame goreng crispy", "ikan"),
    ("nasi putih biasa", "lainnya"),
]

# Preprocess test cases
def preprocess_new_menu(menu_name):
    corpus = menu_name  # Assume already cleaned
    semantic = extract_keywords(menu_name)
    return corpus + " " + semantic

correct = 0
uncertain = 0
manual_results = []

print(f"\n{'No':<4} {'Menu':<35} {'True':<10} {'Pred':<10} {'Conf':<6} {'Status'}")
print("-"*80)

for idx, (menu, true_label) in enumerate(test_cases, 1):
    enhanced_menu = preprocess_new_menu(menu)
    X_new = vectorizer.transform([enhanced_menu])
    
    pred_label = rf.predict(X_new)[0]
    pred_proba = rf.predict_proba(X_new)[0]
    confidence = pred_proba.max()
    
    is_correct = pred_label == true_label
    status = "✅" if is_correct else "❌"
    
    if is_correct:
        correct += 1
    
    if confidence < 0.6:
        status += " ⚠️"
        uncertain += 1
    
    print(f"{idx:<4} {menu:<35} {true_label:<10} {pred_label:<10} {confidence:.2f}   {status}")
    
    manual_results.append({
        'No': idx,
        'Menu': menu,
        'True_Label': true_label,
        'Predicted_Label': pred_label,
        'Confidence': confidence,
        'Correct': is_correct
    })

manual_accuracy = correct / len(test_cases)
print("\n" + "-"*80)
print(f"📊 Manual Test Accuracy: {manual_accuracy:.1%} ({correct}/{len(test_cases)})")
print(f"⚠️  Uncertain predictions (conf < 0.6): {uncertain}")

# Save manual test results
manual_df = pd.DataFrame(manual_results)
manual_df.to_csv(f"{OUTPUT_DIR}/manual_test_results.csv", index=False)
print(f"✅ Saved: {OUTPUT_DIR}/manual_test_results.csv")

# ============================================================
# 10. Feature Importance
# ============================================================

print("\n" + "="*70)
print("🔑 FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_names = vectorizer.get_feature_names_out()
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]

print(f"\n{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Bar'}")
print("-"*70)
for i, idx in enumerate(indices, 1):
    bar = '█' * int(importances[idx] * 100)
    print(f"{i:<6} {feature_names[idx]:<25} {importances[idx]:<12.6f} {bar}")

# Save feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
print(f"\n✅ Saved: {OUTPUT_DIR}/feature_importance.csv")

# Plot feature importance
plt.figure(figsize=(12, 8))
top_n = 20
top_indices = indices[:top_n]
plt.barh(range(top_n), importances[top_indices], color='steelblue', alpha=0.8)
plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
plt.xlabel('Importance Score', fontsize=12)
plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/feature_importance.png")

# ============================================================
# 11. Save Model & Summary
# ============================================================

print("\n" + "="*70)
print("💾 SAVING MODEL & SUMMARY")
print("="*70)

joblib.dump(rf, f"{OUTPUT_DIR}/rf_model_production.pkl")
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer_production.pkl")

# Save preprocessing function
import pickle
with open(f"{OUTPUT_DIR}/preprocess_function.pkl", "wb") as f:
    pickle.dump(preprocess_new_menu, f)

print(f"✅ Saved: {OUTPUT_DIR}/rf_model_production.pkl")
print(f"✅ Saved: {OUTPUT_DIR}/tfidf_vectorizer_production.pkl")
print(f"✅ Saved: {OUTPUT_DIR}/preprocess_function.pkl")

# Create summary report
summary = {
    'Metric': [
        'Total Data',
        'Train Size',
        'Test Size',
        'Train Accuracy',
        'Test Accuracy',
        'Train F1-Score',
        'Test F1-Score',
        'Overfitting Gap (Acc)',
        'Overfitting Gap (F1)',
        'Manual Test Accuracy',
        'Number of Features',
        'Model Type'
    ],
    'Value': [
        len(df),
        len(train_df),
        len(test_df),
        f"{train_acc:.4f}",
        f"{test_acc:.4f}",
        f"{train_f1:.4f}",
        f"{test_f1:.4f}",
        f"{train_acc - test_acc:.4f}",
        f"{train_f1 - test_f1:.4f}",
        f"{manual_accuracy:.4f}",
        X_train_vec.shape[1],
        "Random Forest"
    ]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUTPUT_DIR}/model_summary.csv", index=False)
print(f"✅ Saved: {OUTPUT_DIR}/model_summary.csv")

# ============================================================
# 12. FINAL VERDICT
# ============================================================

print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)

gap = train_f1 - test_f1

if manual_accuracy >= 0.80 and gap < 0.15:
    verdict = "✅ MODEL READY FOR PRODUCTION!"
    status = "EXCELLENT"
elif manual_accuracy >= 0.70 and gap < 0.20:
    verdict = "⚠️  MODEL NEEDS IMPROVEMENT"
    status = "ACCEPTABLE"
else:
    verdict = "❌ MODEL NOT READY"
    status = "NEEDS WORK"

print(f"\n{verdict}")
print(f"Status: {status}")
print(f"\n📊 Key Metrics:")
print(f"   • Test Accuracy    : {test_acc:.2%}")
print(f"   • Test F1-Score    : {test_f1:.4f}")
print(f"   • Manual Test Acc  : {manual_accuracy:.1%}")
print(f"   • Overfitting Gap  : {gap:.4f}")

print("\n💡 Production Recommendations:")
if manual_accuracy >= 0.80:
    print("   ✅ Deploy dengan confidence threshold (reject if < 0.6)")
    print("   ✅ Monitor predictions in production")
    print("   ✅ Set up feedback loop untuk continuous improvement")
else:
    print("   ⚠️  Consider hybrid approach (rule-based + ML)")
    print("   ⚠️  Collect more diverse training data")
    print("   ⚠️  Fine-tune semantic features")

print(f"\n📁 All outputs saved in: {OUTPUT_DIR}/")
print("\n" + "="*70)
print("🎉 TRAINING COMPLETED!")
print("="*70)