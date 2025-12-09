import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/dataset450_indonesia.csv"
OUTPUT_DIR = "rftrain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_RATIOS = [
    (0.70, 0.30, "70-30"),
    (0.80, 0.20, "80-20"),
    (0.90, 0.10, "90-10")
]

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def extract_keywords(text):
    """Extract semantic keywords"""
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


# ============================================================
# RF TRAINER CLASS
# ============================================================

class RFTrainer:
    def __init__(self, df, split_name, train_size):
        self.df = df.copy()
        self.split_name = split_name
        self.train_size = train_size
        self.test_size = 1 - train_size
        self.results = {}
        
    def prepare_data(self):
        """Prepare train-test split"""
        print(f"\n{'='*70}")
        print(f"📊 TRAINING RF+TF-IDF - SPLIT: {self.split_name}")
        print(f"{'='*70}")
        
        # Enhanced corpus
        self.df["enhanced_corpus"] = self.df["Nama_Menu"].apply(preprocess_menu)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.df['enhanced_corpus'],
            self.df['Kategori'],
            train_size=self.train_size,
            random_state=42,
            stratify=self.df['Kategori']
        )
        
        self.train_indices = X_train.index
        self.test_indices = X_test.index
        self.train_df = self.df.iloc[self.train_indices].copy()
        self.test_df = self.df.iloc[self.test_indices].copy()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = self.train_df['Kategori']
        self.y_test = self.test_df['Kategori']
        
        print(f"Train: {len(self.train_df)} ({self.train_size*100:.0f}%)")
        print(f"Test:  {len(self.test_df)} ({self.test_size*100:.0f}%)")
        print(f"\nTrain distribution:")
        print(self.y_train.value_counts())
        print(f"\nTest distribution:")
        print(self.y_test.value_counts())
        
    def train_model(self):
        """Train Random Forest + TF-IDF"""
        print(f"\n⏳ Training RF+TF-IDF Model...")
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"TF-IDF matrix shape: {X_train_vec.shape}")
        
        # Train RF
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train_vec, self.y_train)
        print("✅ Training completed!")
        
        # Predictions
        y_train_pred = self.rf_model.predict(X_train_vec)
        y_test_pred = self.rf_model.predict(X_test_vec)
        
        # Calculate metrics
        self.results = {
            'split': self.split_name,
            'train_size': len(self.train_df),
            'test_size': len(self.test_df),
            'train_acc': accuracy_score(self.y_train, y_train_pred),
            'test_acc': accuracy_score(self.y_test, y_test_pred),
            'train_f1': f1_score(self.y_train, y_train_pred, average='macro'),
            'test_f1': f1_score(self.y_test, y_test_pred, average='macro'),
            'train_precision': precision_score(self.y_train, y_train_pred, average='macro'),
            'test_precision': precision_score(self.y_test, y_test_pred, average='macro'),
            'train_recall': recall_score(self.y_train, y_train_pred, average='macro'),
            'test_recall': recall_score(self.y_test, y_test_pred, average='macro'),
            'overfitting_gap_acc': accuracy_score(self.y_train, y_train_pred) - accuracy_score(self.y_test, y_test_pred),
            'overfitting_gap_f1': f1_score(self.y_train, y_train_pred, average='macro') - f1_score(self.y_test, y_test_pred, average='macro'),
            'y_test_true': self.y_test.values,
            'y_test_pred': y_test_pred,
            'cm': confusion_matrix(self.y_test, y_test_pred)
        }
        
        # Store for later use
        self.X_train_vec = X_train_vec
        self.X_test_vec = X_test_vec
        
        self.print_results()
        
    def print_results(self):
        """Print evaluation results"""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION RESULTS - {self.split_name}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<25} {'Train':<12} {'Test':<12} {'Gap'}")
        print("-"*70)
        print(f"{'Accuracy':<25} {self.results['train_acc']:<12.4f} {self.results['test_acc']:<12.4f} {self.results['overfitting_gap_acc']:.4f}")
        print(f"{'F1-Score (macro)':<25} {self.results['train_f1']:<12.4f} {self.results['test_f1']:<12.4f} {self.results['overfitting_gap_f1']:.4f}")
        print(f"{'Precision (macro)':<25} {self.results['train_precision']:<12.4f} {self.results['test_precision']:<12.4f}")
        print(f"{'Recall (macro)':<25} {self.results['train_recall']:<12.4f} {self.results['test_recall']:<12.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report (Test Set):")
        print(classification_report(self.y_test, self.results['y_test_pred']))
        
    def save_model(self):
        """Save trained model and vectorizer"""
        split_folder = os.path.join(OUTPUT_DIR, self.split_name.replace('-', '_'))
        os.makedirs(split_folder, exist_ok=True)
        
        # Save model
        model_path = os.path.join(split_folder, "rf_model.pkl")
        vectorizer_path = os.path.join(split_folder, "tfidf_vectorizer.pkl")
        
        joblib.dump(self.rf_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\n💾 Model saved:")
        print(f"   - {model_path}")
        print(f"   - {vectorizer_path}")
        
        return split_folder
        
    def save_results(self):
        """Save evaluation results"""
        split_folder = os.path.join(OUTPUT_DIR, self.split_name.replace('-', '_'))
        os.makedirs(split_folder, exist_ok=True)
        
        # 1. Save confusion matrix
        self.plot_confusion_matrix(split_folder)
        
        # 2. Save metrics summary
        metrics_df = pd.DataFrame([{
            'Split': self.results['split'],
            'Train_Size': self.results['train_size'],
            'Test_Size': self.results['test_size'],
            'Train_Accuracy': self.results['train_acc'],
            'Test_Accuracy': self.results['test_acc'],
            'Train_F1': self.results['train_f1'],
            'Test_F1': self.results['test_f1'],
            'Train_Precision': self.results['train_precision'],
            'Test_Precision': self.results['test_precision'],
            'Train_Recall': self.results['train_recall'],
            'Test_Recall': self.results['test_recall'],
            'Overfitting_Gap_Acc': self.results['overfitting_gap_acc'],
            'Overfitting_Gap_F1': self.results['overfitting_gap_f1']
        }])
        
        metrics_path = os.path.join(split_folder, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   - {metrics_path}")
        
        # 3. Save classification report
        report_dict = classification_report(self.y_test, self.results['y_test_pred'], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(split_folder, "classification_report.csv")
        report_df.to_csv(report_path)
        print(f"   - {report_path}")
        
        # 4. Save train/test indices
        indices_df = pd.DataFrame({
            'train_indices': list(self.train_indices),
            'test_indices': list(self.test_indices) + [None] * (len(self.train_indices) - len(self.test_indices))
        })
        indices_path = os.path.join(split_folder, "split_indices.csv")
        indices_df.to_csv(indices_path, index=False)
        print(f"   - {indices_path}")
        
        return self.results
        
    def plot_confusion_matrix(self, split_folder):
        """Plot and save confusion matrix"""
        labels = sorted(self.df['Kategori'].unique())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.results['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - RF+TF-IDF ({self.split_name})\n' +
                 f'Accuracy: {self.results["test_acc"]:.2%} | F1-Score: {self.results["test_f1"]:.4f}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        cm_path = os.path.join(split_folder, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   - {cm_path}")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("="*70)
    print("🤖 RF+TF-IDF TRAINING & EVALUATION")
    print("="*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded {len(df)} rows")
    print(f"\nKategori distribution:")
    print(df['Kategori'].value_counts())
    
    # Train for all splits
    all_results = []
    
    for train_size, test_size, split_name in SPLIT_RATIOS:
        trainer = RFTrainer(df, split_name, train_size)
        trainer.prepare_data()
        trainer.train_model()
        trainer.save_model()
        results = trainer.save_results()
        all_results.append(results)
        
        print(f"\n✅ Completed: {split_name}")
        print("="*70)
    
    # Create overall summary
    create_overall_summary(all_results)
    
    print("\n" + "="*70)
    print("🎉 RF+TF-IDF TRAINING COMPLETED!")
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
            'Train_Size': r['train_size'],
            'Test_Size': r['test_size'],
            'Test_Accuracy': r['test_acc'],
            'Test_F1': r['test_f1'],
            'Test_Precision': r['test_precision'],
            'Test_Recall': r['test_recall'],
            'Overfitting_Gap_Acc': r['overfitting_gap_acc'],
            'Overfitting_Gap_F1': r['overfitting_gap_f1']
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
    axes[0, 0].bar(x_pos, summary_df['Test_Accuracy'], color='steelblue', alpha=0.8)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(splits)
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Across Splits', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    for i, val in enumerate(summary_df['Test_Accuracy']):
        axes[0, 0].text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=10)
    
    # Plot 2: Test F1-Score
    axes[0, 1].bar(x_pos, summary_df['Test_F1'], color='coral', alpha=0.8)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(splits)
    axes[0, 1].set_ylabel('Test F1-Score', fontsize=11)
    axes[0, 1].set_title('Test F1-Score Across Splits', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    for i, val in enumerate(summary_df['Test_F1']):
        axes[0, 1].text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=10)
    
    # Plot 3: Overfitting Gap
    axes[1, 0].bar(x_pos, summary_df['Overfitting_Gap_F1'], 
                   color=['green' if x < 0.15 else 'orange' if x < 0.25 else 'red' 
                          for x in summary_df['Overfitting_Gap_F1']], alpha=0.7)
    axes[1, 0].axhline(y=0.15, color='orange', linestyle='--', linewidth=1, label='Threshold (0.15)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(splits)
    axes[1, 0].set_ylabel('Overfitting Gap (F1)', fontsize=11)
    axes[1, 0].set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, val in enumerate(summary_df['Overfitting_Gap_F1']):
        axes[1, 0].text(i, val + 0.01, f'{val:.4f}', ha='center', fontsize=10)
    
    # Plot 4: All Metrics Comparison
    width = 0.25
    x_pos2 = np.arange(len(splits))
    
    axes[1, 1].bar(x_pos2 - width, summary_df['Test_Accuracy'], width, 
                   label='Accuracy', alpha=0.8)
    axes[1, 1].bar(x_pos2, summary_df['Test_Precision'], width,
                   label='Precision', alpha=0.8)
    axes[1, 1].bar(x_pos2 + width, summary_df['Test_Recall'], width,
                   label='Recall', alpha=0.8)
    
    axes[1, 1].set_xticks(x_pos2)
    axes[1, 1].set_xticklabels(splits)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Test Metrics Comparison', fontsize=12, fontweight='bold')
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