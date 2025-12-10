import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import dari utils
from utils import (
    MODES, preprocess_menu, clean_karbo_list,
    compute_mccbf_scores, rf_category_scores
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/dataset450_indonesia.csv"
RF_MODEL_DIR = "rftrain"
OUTPUT_DIR = "hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_RATIOS = [
    (0.70, 0.30, "70-30"),
    (0.80, 0.20, "80-20"),
    (0.90, 0.10, "90-10")
]

# HYBRID CONFIGURATION
ALPHA = 0.7
MODE = "seimbang"

# ============================================================
# HYBRID EVALUATOR CLASS
# ============================================================

class HybridEvaluator:
    def __init__(self, df, split_name, train_size):
        self.df = df.copy()
        self.split_name = split_name
        self.train_size = train_size
        self.test_size = 1 - train_size
        self.results = {}
        
    def load_rf_model(self):
        """Load RF model dari training sebelumnya"""
        split_folder = os.path.join(RF_MODEL_DIR, self.split_name.replace('-', '_'))
        
        model_path = os.path.join(split_folder, "rf_model.pkl")
        vectorizer_path = os.path.join(split_folder, "tfidf_vectorizer.pkl")
        indices_path = os.path.join(split_folder, "split_indices.csv")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RF model not found: {model_path}\nPlease run script 1 first!")
        
        print(f"\n📂 Loading RF model from: {split_folder}")
        self.rf_model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Load indices
        indices_df = pd.read_csv(indices_path)
        self.train_indices = indices_df['train_indices'].dropna().astype(int).tolist()
        self.test_indices = indices_df['test_indices'].dropna().astype(int).tolist()
        
        print(f"✅ Loaded RF model and vectorizer")
        
    def prepare_data(self):
        """Prepare data sesuai split yang sama dengan RF"""
        print(f"\n{'='*70}")
        print(f"🔬 EVALUATING HYBRID MCCBF+RF - SPLIT: {self.split_name}")
        print(f"{'='*70}")
        
        # Parse Karbo_List
        self.df["Karbo_List"] = self.df["Karbo_List"].apply(clean_karbo_list)
        
        # Enhanced corpus
        self.df["enhanced_corpus"] = self.df["Nama_Menu"].apply(preprocess_menu)
        
        # Use same split as RF
        self.train_df = self.df.iloc[self.train_indices].copy()
        self.test_df = self.df.iloc[self.test_indices].copy()
        
        print(f"Train: {len(self.train_df)} ({self.train_size*100:.0f}%)")
        print(f"Test:  {len(self.test_df)} ({self.test_size*100:.0f}%)")
        
    def evaluate_hybrid(self):
        """Evaluate Hybrid MCCBF+RF"""
        print(f"\n⏳ Computing hybrid predictions...")
        
        # Precompute TF-IDF untuk semua data
        X_all = self.vectorizer.transform(self.df["enhanced_corpus"])
        
        # Generate test queries dari test set
        test_queries = []
        for idx in self.test_indices:
            row = self.df.iloc[idx]
            test_queries.append({
                'query': row['Nama_Menu'].lower(),
                'true_label': row['Kategori'],
                'target_calorie': None,
                'preferred_karbo': None
            })

        # Predict dengan hybrid
        y_hybrid_pred = []
        hybrid_scores_list = []
        mccbf_scores_list = []
        rf_scores_list = []
        
        for i, test_case in enumerate(test_queries):
            if (i + 1) % 10 == 0:
                print(f"   Processing {i+1}/{len(test_queries)}...")
            
            # MCCBF Score
            mccbf_scores, _ = compute_mccbf_scores(
                self.df, X_all, self.vectorizer,
                user_query=test_case['query'],
                target_calorie=test_case['target_calorie'],
                preferred_karbo=test_case['preferred_karbo'],
                mode=MODE
            )
            
            # RF Score
            rf_proba = self.rf_model.predict_proba(X_all)
            rf_scores = rf_proba.max(axis=1)
            
            # Hybrid score
            hybrid = ALPHA * mccbf_scores + (1 - ALPHA) * rf_scores
            
            # Get best prediction
            best_idx = hybrid.argmax()
            pred_label = self.df.iloc[best_idx]['Kategori']
            y_hybrid_pred.append(pred_label)
            hybrid_scores_list.append(hybrid[best_idx])
            mccbf_scores_list.append(mccbf_scores[best_idx])
            rf_scores_list.append(rf_scores[best_idx])
        
        y_hybrid_pred = np.array(y_hybrid_pred)
        y_test_true = self.test_df['Kategori'].values
        
        # Calculate metrics
        self.results = {
            'split': self.split_name,
            'test_size': len(self.test_df),
            'test_acc': accuracy_score(y_test_true, y_hybrid_pred),
            'test_f1': f1_score(y_test_true, y_hybrid_pred, average='macro'),
            'test_precision': precision_score(y_test_true, y_hybrid_pred, average='macro'),
            'test_recall': recall_score(y_test_true, y_hybrid_pred, average='macro'),
            'y_test_true': y_test_true,
            'y_test_pred': y_hybrid_pred,
            'cm': confusion_matrix(y_test_true, y_hybrid_pred),
            'avg_hybrid_score': np.mean(hybrid_scores_list),
            'avg_mccbf_score': np.mean(mccbf_scores_list),
            'avg_rf_score': np.mean(rf_scores_list)
        }
        
        print(f"✅ Hybrid evaluation completed!")
        self.print_results()
        
    def print_results(self):
        """Print evaluation results"""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION RESULTS - {self.split_name}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<25} {'Score'}")
        print("-"*50)
        print(f"{'Test Accuracy':<25} {self.results['test_acc']:.4f}")
        print(f"{'Test F1-Score':<25} {self.results['test_f1']:.4f}")
        print(f"{'Test Precision':<25} {self.results['test_precision']:.4f}")
        print(f"{'Test Recall':<25} {self.results['test_recall']:.4f}")
        print(f"\n{'Average Scores:':<25}")
        print(f"{'  - Hybrid Score':<25} {self.results['avg_hybrid_score']:.4f}")
        print(f"{'  - MCCBF Score':<25} {self.results['avg_mccbf_score']:.4f}")
        print(f"{'  - RF Score':<25} {self.results['avg_rf_score']:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report (Test Set):")
        print(classification_report(self.results['y_test_true'], self.results['y_test_pred']))
        
    def save_results(self):
        """Save evaluation results"""
        split_folder = os.path.join(OUTPUT_DIR, self.split_name.replace('-', '_'))
        os.makedirs(split_folder, exist_ok=True)
        
        print(f"\n💾 Saving results to: {split_folder}")
        
        # 1. Save confusion matrix
        self.plot_confusion_matrix(split_folder)
        
        # 2. Save metrics summary
        metrics_df = pd.DataFrame([{
            'Split': self.results['split'],
            'Test_Size': self.results['test_size'],
            'Test_Accuracy': self.results['test_acc'],
            'Test_F1': self.results['test_f1'],
            'Test_Precision': self.results['test_precision'],
            'Test_Recall': self.results['test_recall'],
            'Avg_Hybrid_Score': self.results['avg_hybrid_score'],
            'Avg_MCCBF_Score': self.results['avg_mccbf_score'],
            'Avg_RF_Score': self.results['avg_rf_score'],
            'Alpha': ALPHA,
            'Mode': MODE
        }])
        
        metrics_path = os.path.join(split_folder, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   - {metrics_path}")
        
        # 3. Save classification report
        report_dict = classification_report(self.results['y_test_true'], 
                                           self.results['y_test_pred'], 
                                           output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(split_folder, "classification_report.csv")
        report_df.to_csv(report_path)
        print(f"   - {report_path}")
        
        return self.results
        
    def plot_confusion_matrix(self, split_folder):
        """Plot and save confusion matrix"""
        labels = sorted(self.df['Kategori'].unique())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.results['cm'], annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Hybrid MCCBF+RF ({self.split_name})\n' +
                 f'Accuracy: {self.results["test_acc"]:.2%} | F1-Score: {self.results["test_f1"]:.4f}\n' +
                 f'Alpha: {ALPHA} | Mode: {MODE}',
                 fontsize=13, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        cm_path = os.path.join(split_folder, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   - {cm_path}")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def main():
    print("="*70)
    print("🔬 HYBRID MCCBF+RF EVALUATION")
    print(f"Alpha: {ALPHA} | Mode: {MODE}")
    print("="*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded {len(df)} rows")
    
    # Evaluate for all splits
    all_results = []
    
    for train_size, test_size, split_name in SPLIT_RATIOS:
        evaluator = HybridEvaluator(df, split_name, train_size)
        evaluator.load_rf_model()
        evaluator.prepare_data()
        evaluator.evaluate_hybrid()
        results = evaluator.save_results()
        all_results.append(results)
        
        print(f"\n✅ Completed: {split_name}")
        print("="*70)
    
    # Create overall summary
    create_overall_summary(all_results)
    
    print("\n" + "="*70)
    print("🎉 HYBRID EVALUATION COMPLETED!")
    print(f"📁 All results saved in: {OUTPUT_DIR}/")
    print("="*70)


def create_overall_summary(all_results):
    """Create overall summary across all splits"""
    print(f"\n{'='*70}")
    print("📊 OVERALL SUMMARY - ALL SPLITS")
    print(f"{'='*70}")
    
    # Create summary dataframe
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Split': r['split'],
            'Test_Size': r['test_size'],
            'Test_Accuracy': r['test_acc'],
            'Test_F1': r['test_f1'],
            'Test_Precision': r['test_precision'],
            'Test_Recall': r['test_recall'],
            'Avg_Hybrid_Score': r['avg_hybrid_score'],
            'Avg_MCCBF_Score': r['avg_mccbf_score'],
            'Avg_RF_Score': r['avg_rf_score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "overall_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + summary_df.to_string(index=False))
    print(f"\n✅ Saved: {summary_path}")
    
    # Plot comparison
    plot_comparison(summary_df)


def plot_comparison(summary_df):
    """Plot comparison across splits"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    splits = summary_df['Split'].values
    x_pos = np.arange(len(splits))
    
    # Plot 1: Test Accuracy
    axes[0, 0].bar(x_pos, summary_df['Test_Accuracy'], color='coral', alpha=0.8)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(splits)
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Across Splits', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    for i, val in enumerate(summary_df['Test_Accuracy']):
        axes[0, 0].text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=10)
    
    # Plot 2: Test F1-Score
    axes[0, 1].bar(x_pos, summary_df['Test_F1'], color='lightcoral', alpha=0.8)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(splits)
    axes[0, 1].set_ylabel('Test F1-Score', fontsize=11)
    axes[0, 1].set_title('Test F1-Score Across Splits', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    for i, val in enumerate(summary_df['Test_F1']):
        axes[0, 1].text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=10)
    
    # Plot 3: Score Components
    width = 0.25
    x_pos2 = np.arange(len(splits))
    
    axes[1, 0].bar(x_pos2 - width, summary_df['Avg_MCCBF_Score'], width,
                   label='MCCBF', color='skyblue', alpha=0.8)
    axes[1, 0].bar(x_pos2, summary_df['Avg_RF_Score'], width,
                   label='RF', color='lightgreen', alpha=0.8)
    axes[1, 0].bar(x_pos2 + width, summary_df['Avg_Hybrid_Score'], width,
                   label='Hybrid', color='coral', alpha=0.8)
    
    axes[1, 0].set_xticks(x_pos2)
    axes[1, 0].set_xticklabels(splits)
    axes[1, 0].set_ylabel('Average Score', fontsize=11)
    axes[1, 0].set_title(f'Score Components (Alpha={ALPHA})', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: All Metrics
    width = 0.2
    x_pos3 = np.arange(len(splits))
    
    axes[1, 1].bar(x_pos3 - 1.5*width, summary_df['Test_Accuracy'], width,
                   label='Accuracy', alpha=0.8)
    axes[1, 1].bar(x_pos3 - 0.5*width, summary_df['Test_F1'], width,
                   label='F1', alpha=0.8)
    axes[1, 1].bar(x_pos3 + 0.5*width, summary_df['Test_Precision'], width,
                   label='Precision', alpha=0.8)
    axes[1, 1].bar(x_pos3 + 1.5*width, summary_df['Test_Recall'], width,
                   label='Recall', alpha=0.8)
    
    axes[1, 1].set_xticks(x_pos3)
    axes[1, 1].set_xticklabels(splits)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('All Test Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "splits_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {comparison_path}")


if __name__ == "__main__":
    main()