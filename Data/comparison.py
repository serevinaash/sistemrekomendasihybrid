import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

RF_DIR = "rftrain"
HYBRID_DIR = "hybrid"
OUTPUT_DIR = "comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_NAMES = ["70_30", "80_20", "90_10"]

# ============================================================
# LOAD RESULTS
# ============================================================

def load_all_results():
    """Load results dari RF dan Hybrid"""
    print("="*70)
    print("📊 RF vs HYBRID COMPARISON ANALYSIS")
    print("="*70)
    
    rf_results = []
    hybrid_results = []
    
    for split in SPLIT_NAMES:
        # Load RF results
        rf_path = os.path.join(RF_DIR, split, "metrics_summary.csv")
        if os.path.exists(rf_path):
            rf_df = pd.read_csv(rf_path)
            rf_results.append(rf_df.iloc[0].to_dict())
            print(f"✅ Loaded RF results: {split}")
        else:
            print(f"❌ RF results not found: {rf_path}")
            
        # Load Hybrid results
        hybrid_path = os.path.join(HYBRID_DIR, split, "metrics_summary.csv")
        if os.path.exists(hybrid_path):
            hybrid_df = pd.read_csv(hybrid_path)
            hybrid_results.append(hybrid_df.iloc[0].to_dict())
            print(f"✅ Loaded Hybrid results: {split}")
        else:
            print(f"❌ Hybrid results not found: {hybrid_path}")
    
    if len(rf_results) != len(SPLIT_NAMES) or len(hybrid_results) != len(SPLIT_NAMES):
        raise ValueError("Not all results are available! Please run scripts 1 and 2 first.")
    
    return pd.DataFrame(rf_results), pd.DataFrame(hybrid_results)


# ============================================================
# CREATE COMPARISON DATAFRAME
# ============================================================

def create_comparison_df(rf_df, hybrid_df):
    """Create comprehensive comparison dataframe"""
    
    comparison_data = []
    
    for i, split in enumerate(SPLIT_NAMES):
        split_label = split.replace('_', '-')
        
        comparison_data.append({
            'Split': split_label,
            'RF_Test_Acc': rf_df.iloc[i]['Test_Accuracy'],
            'Hybrid_Test_Acc': hybrid_df.iloc[i]['Test_Accuracy'],
            'RF_Test_F1': rf_df.iloc[i]['Test_F1'],
            'Hybrid_Test_F1': hybrid_df.iloc[i]['Test_F1'],
            'RF_Test_Precision': rf_df.iloc[i]['Test_Precision'],
            'Hybrid_Test_Precision': hybrid_df.iloc[i]['Test_Precision'],
            'RF_Test_Recall': rf_df.iloc[i]['Test_Recall'],
            'Hybrid_Test_Recall': hybrid_df.iloc[i]['Test_Recall'],
            'Improvement_Acc': hybrid_df.iloc[i]['Test_Accuracy'] - rf_df.iloc[i]['Test_Accuracy'],
            'Improvement_F1': hybrid_df.iloc[i]['Test_F1'] - rf_df.iloc[i]['Test_F1'],
            'Improvement_Precision': hybrid_df.iloc[i]['Test_Precision'] - rf_df.iloc[i]['Test_Precision'],
            'Improvement_Recall': hybrid_df.iloc[i]['Test_Recall'] - rf_df.iloc[i]['Test_Recall'],
            'RF_Overfitting_Gap': rf_df.iloc[i].get('Overfitting_Gap_F1', 0),
        })
    
    return pd.DataFrame(comparison_data)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_main_comparison(comp_df):
    """Main comparison: Accuracy and F1-Score"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    splits = comp_df['Split'].values
    x_pos = np.arange(len(splits))
    width = 0.35
    
    # Plot 1: Accuracy Comparison
    bars1 = axes[0].bar(x_pos - width/2, comp_df['RF_Test_Acc'], width,
                        label='RF+TF-IDF', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x_pos + width/2, comp_df['Hybrid_Test_Acc'], width,
                        label='Hybrid MCCBF+RF', color='coral', alpha=0.8, edgecolor='black')
    
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(splits, fontsize=11)
    axes[0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1.05])
    
    # Add value labels
    for i in range(len(splits)):
        rf_val = comp_df.iloc[i]['RF_Test_Acc']
        hybrid_val = comp_df.iloc[i]['Hybrid_Test_Acc']
        axes[0].text(i - width/2, rf_val + 0.02, f'{rf_val:.4f}',
                    ha='center', fontsize=10, fontweight='bold')
        axes[0].text(i + width/2, hybrid_val + 0.02, f'{hybrid_val:.4f}',
                    ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: F1-Score Comparison
    bars3 = axes[1].bar(x_pos - width/2, comp_df['RF_Test_F1'], width,
                        label='RF+TF-IDF', color='steelblue', alpha=0.8, edgecolor='black')
    bars4 = axes[1].bar(x_pos + width/2, comp_df['Hybrid_Test_F1'], width,
                        label='Hybrid MCCBF+RF', color='coral', alpha=0.8, edgecolor='black')
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(splits, fontsize=11)
    axes[1].set_ylabel('Test F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Test F1-Score Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])
    
    # Add value labels
    for i in range(len(splits)):
        rf_val = comp_df.iloc[i]['RF_Test_F1']
        hybrid_val = comp_df.iloc[i]['Hybrid_Test_F1']
        axes[1].text(i - width/2, rf_val + 0.02, f'{rf_val:.4f}',
                    ha='center', fontsize=10, fontweight='bold')
        axes[1].text(i + width/2, hybrid_val + 0.02, f'{hybrid_val:.4f}',
                    ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/main_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/main_comparison.png")


def plot_improvement_analysis(comp_df):
    """Improvement analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    splits = comp_df['Split'].values
    x_pos = np.arange(len(splits))
    
    # Plot 1: Accuracy Improvement
    colors_acc = ['green' if x > 0 else 'red' for x in comp_df['Improvement_Acc']]
    bars1 = axes[0, 0].bar(x_pos, comp_df['Improvement_Acc'], 
                           color=colors_acc, alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(splits, fontsize=11)
    axes[0, 0].set_ylabel('Improvement (Δ)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Accuracy Improvement (Hybrid - RF)', fontsize=13, fontweight='bold', pad=15)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, val in enumerate(comp_df['Improvement_Acc']):
        color = 'green' if val > 0 else 'red'
        axes[0, 0].text(i, val + (0.005 if val > 0 else -0.005),
                       f'{val:+.4f}',
                       ha='center', fontsize=10, fontweight='bold', color=color)
    
    # Plot 2: F1-Score Improvement
    colors_f1 = ['green' if x > 0 else 'red' for x in comp_df['Improvement_F1']]
    bars2 = axes[0, 1].bar(x_pos, comp_df['Improvement_F1'],
                           color=colors_f1, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(splits, fontsize=11)
    axes[0, 1].set_ylabel('Improvement (Δ)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('F1-Score Improvement (Hybrid - RF)', fontsize=13, fontweight='bold', pad=15)
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, val in enumerate(comp_df['Improvement_F1']):
        color = 'green' if val > 0 else 'red'
        axes[0, 1].text(i, val + (0.005 if val > 0 else -0.005),
                       f'{val:+.4f}',
                       ha='center', fontsize=10, fontweight='bold', color=color)
    
    # Plot 3: All Improvements
    width = 0.2
    x_pos2 = np.arange(len(splits))
    
    axes[1, 0].bar(x_pos2 - 1.5*width, comp_df['Improvement_Acc'], width,
                   label='Accuracy', alpha=0.8, edgecolor='black')
    axes[1, 0].bar(x_pos2 - 0.5*width, comp_df['Improvement_F1'], width,
                   label='F1-Score', alpha=0.8, edgecolor='black')
    axes[1, 0].bar(x_pos2 + 0.5*width, comp_df['Improvement_Precision'], width,
                   label='Precision', alpha=0.8, edgecolor='black')
    axes[1, 0].bar(x_pos2 + 1.5*width, comp_df['Improvement_Recall'], width,
                   label='Recall', alpha=0.8, edgecolor='black')
    
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xticks(x_pos2)
    axes[1, 0].set_xticklabels(splits, fontsize=11)
    axes[1, 0].set_ylabel('Improvement (Δ)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('All Metrics Improvement', fontsize=13, fontweight='bold', pad=15)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Percentage Improvement
    pct_improvement_acc = (comp_df['Improvement_Acc'] / comp_df['RF_Test_Acc']) * 100
    pct_improvement_f1 = (comp_df['Improvement_F1'] / comp_df['RF_Test_F1']) * 100
    
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, pct_improvement_acc, width,
                   label='Accuracy (%)', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x_pos + width/2, pct_improvement_f1, width,
                   label='F1-Score (%)', alpha=0.8, edgecolor='black')
    
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(splits, fontsize=11)
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Percentage Improvement', fontsize=13, fontweight='bold', pad=15)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i in range(len(splits)):
        axes[1, 1].text(i - width/2, pct_improvement_acc.iloc[i] + 0.2,
                       f'{pct_improvement_acc.iloc[i]:+.2f}%',
                       ha='center', fontsize=9, fontweight='bold')
        axes[1, 1].text(i + width/2, pct_improvement_f1.iloc[i] + 0.2,
                       f'{pct_improvement_f1.iloc[i]:+.2f}%',
                       ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/improvement_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/improvement_analysis.png")


def plot_all_metrics_heatmap(comp_df):
    """Heatmap of all metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Prepare data for RF
    rf_data = comp_df[['RF_Test_Acc', 'RF_Test_F1', 'RF_Test_Precision', 'RF_Test_Recall']].T
    rf_data.columns = comp_df['Split']
    
    # Prepare data for Hybrid
    hybrid_data = comp_df[['Hybrid_Test_Acc', 'Hybrid_Test_F1', 'Hybrid_Test_Precision', 'Hybrid_Test_Recall']].T
    hybrid_data.columns = comp_df['Split']
    
    # Plot 1: RF Heatmap
    sns.heatmap(rf_data, annot=True, fmt='.4f', cmap='Blues',
                ax=axes[0], cbar_kws={'label': 'Score'},
                vmin=0.5, vmax=1.0, linewidths=1, linecolor='white')
    axes[0].set_title('RF+TF-IDF Metrics', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Metrics', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Split', fontsize=12, fontweight='bold')
    axes[0].set_yticklabels(['Accuracy', 'F1-Score', 'Precision', 'Recall'], rotation=0)
    
    # Plot 2: Hybrid Heatmap
    sns.heatmap(hybrid_data, annot=True, fmt='.4f', cmap='Oranges',
                ax=axes[1], cbar_kws={'label': 'Score'},
                vmin=0.5, vmax=1.0, linewidths=1, linecolor='white')
    axes[1].set_title('Hybrid MCCBF+RF Metrics', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Metrics', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Split', fontsize=12, fontweight='bold')
    axes[1].set_yticklabels(['Accuracy', 'F1-Score', 'Precision', 'Recall'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/metrics_heatmap.png")


def plot_confusion_matrices_comparison():
    """Plot side-by-side confusion matrices for all splits"""
    
    print("\n📊 Loading confusion matrices...")
    
    for split in SPLIT_NAMES:
        split_label = split.replace('_', '-')
        
        # Load confusion matrices from images is not possible, so we skip
        # Instead, we note that CMs are already saved individually
        print(f"   ℹ️  Confusion matrices for {split_label} are in:")
        print(f"      - {RF_DIR}/{split}/confusion_matrix.png")
        print(f"      - {HYBRID_DIR}/{split}/confusion_matrix.png")
    
    print("\n   💡 Tip: View individual confusion matrices in rftrain/ and hybrid/ folders")


def plot_winner_summary(comp_df):
    """Winner summary for each split"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Split', 'Winner (Acc)', 'Winner (F1)', 'Best RF Metric', 'Best Hybrid Metric', 'Overall Winner']
    
    for i, row in comp_df.iterrows():
        winner_acc = 'Hybrid' if row['Improvement_Acc'] > 0 else 'RF'
        winner_f1 = 'Hybrid' if row['Improvement_F1'] > 0 else 'RF'
        best_rf = f"{max(row['RF_Test_Acc'], row['RF_Test_F1']):.4f}"
        best_hybrid = f"{max(row['Hybrid_Test_Acc'], row['Hybrid_Test_F1']):.4f}"
        
        # Determine overall winner
        if row['Improvement_Acc'] > 0 and row['Improvement_F1'] > 0:
            overall = '🏆 Hybrid'
        elif row['Improvement_Acc'] < 0 and row['Improvement_F1'] < 0:
            overall = '🏆 RF'
        else:
            overall = '⚖️ Mixed'
        
        table_data.append([
            row['Split'],
            winner_acc,
            winner_f1,
            best_rf,
            best_hybrid,
            overall
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='upper center',
                    colColours=['lightgray']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Color cells
    for i in range(len(table_data)):
        # Winner columns
        if table_data[i][1] == 'Hybrid':
            table[(i+1, 1)].set_facecolor('lightgreen')
        else:
            table[(i+1, 1)].set_facecolor('lightblue')
            
        if table_data[i][2] == 'Hybrid':
            table[(i+1, 2)].set_facecolor('lightgreen')
        else:
            table[(i+1, 2)].set_facecolor('lightblue')
        
        # Overall winner
        if 'Hybrid' in table_data[i][5]:
            table[(i+1, 5)].set_facecolor('lightgreen')
        elif 'RF' in table_data[i][5]:
            table[(i+1, 5)].set_facecolor('lightblue')
        else:
            table[(i+1, 5)].set_facecolor('lightyellow')
    
    plt.title('Performance Winner Summary', fontsize=16, fontweight='bold', pad=40)
    plt.savefig(f"{OUTPUT_DIR}/winner_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/winner_summary.png")


def create_detailed_report(comp_df):
    """Create detailed text report"""
    
    report_path = os.path.join(OUTPUT_DIR, "detailed_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RF+TF-IDF vs HYBRID MCCBF+RF - DETAILED COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        for i, row in comp_df.iterrows():
            f.write(f"{'='*80}\n")
            f.write(f"SPLIT: {row['Split']}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("RF+TF-IDF Results:\n")
            f.write(f"  - Test Accuracy:  {row['RF_Test_Acc']:.4f}\n")
            f.write(f"  - Test F1-Score:  {row['RF_Test_F1']:.4f}\n")
            f.write(f"  - Test Precision: {row['RF_Test_Precision']:.4f}\n")
            f.write(f"  - Test Recall:    {row['RF_Test_Recall']:.4f}\n")
            f.write(f"  - Overfitting Gap: {row['RF_Overfitting_Gap']:.4f}\n\n")
            
            f.write("Hybrid MCCBF+RF Results:\n")
            f.write(f"  - Test Accuracy:  {row['Hybrid_Test_Acc']:.4f}\n")
            f.write(f"  - Test F1-Score:  {row['Hybrid_Test_F1']:.4f}\n")
            f.write(f"  - Test Precision: {row['Hybrid_Test_Precision']:.4f}\n")
            f.write(f"  - Test Recall:    {row['Hybrid_Test_Recall']:.4f}\n\n")
            
            f.write("Improvements (Hybrid - RF):\n")
            f.write(f"  - Accuracy:  {row['Improvement_Acc']:+.4f} ({(row['Improvement_Acc']/row['RF_Test_Acc']*100):+.2f}%)\n")
            f.write(f"  - F1-Score:  {row['Improvement_F1']:+.4f} ({(row['Improvement_F1']/row['RF_Test_F1']*100):+.2f}%)\n")
            f.write(f"  - Precision: {row['Improvement_Precision']:+.4f}\n")
            f.write(f"  - Recall:    {row['Improvement_Recall']:+.4f}\n\n")
            
            # Verdict
            if row['Improvement_Acc'] > 0 and row['Improvement_F1'] > 0:
                verdict = "✅ HYBRID WINS - Better in both Accuracy and F1-Score"
            elif row['Improvement_Acc'] < 0 and row['Improvement_F1'] < 0:
                verdict = "❌ RF WINS - Better in both Accuracy and F1-Score"
            else:
                verdict = "⚖️ MIXED RESULTS - Each model has strengths"
            
            f.write(f"VERDICT: {verdict}\n\n")
        
        # Overall summary
        f.write("="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        avg_imp_acc = comp_df['Improvement_Acc'].mean()
        avg_imp_f1 = comp_df['Improvement_F1'].mean()
        
        f.write(f"Average Improvement (Hybrid over RF):\n")
        f.write(f"  - Accuracy:  {avg_imp_acc:+.4f}\n")
        f.write(f"  - F1-Score:  {avg_imp_f1:+.4f}\n\n")
        
        if avg_imp_acc > 0 and avg_imp_f1 > 0:
            f.write("FINAL VERDICT: 🏆 Hybrid MCCBF+RF is consistently better\n")
        elif avg_imp_acc < 0 and avg_imp_f1 < 0:
            f.write("FINAL VERDICT: 🏆 RF+TF-IDF is consistently better\n")
        else:
            f.write("FINAL VERDICT: ⚖️ Performance varies by split\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Saved: {report_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    # Load results
    rf_df, hybrid_df = load_all_results()
    
    # Create comparison dataframe
    print(f"\n{'='*70}")
    print("📊 Creating comparison dataframe...")
    comp_df = create_comparison_df(rf_df, hybrid_df)
    
    # Save comparison CSV
    comp_path = os.path.join(OUTPUT_DIR, "full_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"✅ Saved: {comp_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("📋 COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    print(comp_df.to_string(index=False))
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("📊 Generating visualizations...")
    print(f"{'='*70}\n")
    
    plot_main_comparison(comp_df)
    plot_improvement_analysis(comp_df)
    plot_all_metrics_heatmap(comp_df)
    plot_confusion_matrices_comparison()
    plot_winner_summary(comp_df)
    create_detailed_report(comp_df)
    
    print(f"\n{'='*70}")
    print("🎉 COMPARISON ANALYSIS COMPLETED!")
    print(f"📁 All results saved in: {OUTPUT_DIR}/")
    print(f"{'='*70}")
    
    # Print key findings
    print(f"\n{'='*70}")
    print("🔑 KEY FINDINGS")
    print(f"{'='*70}\n")
    
    avg_imp_acc = comp_df['Improvement_Acc'].mean()
    avg_imp_f1 = comp_df['Improvement_F1'].mean()
    
    print(f"Average Improvement (Hybrid over RF):")
    print(f"  • Accuracy:  {avg_imp_acc:+.4f} ({(avg_imp_acc/comp_df['RF_Test_Acc'].mean()*100):+.2f}%)")
    print(f"  • F1-Score:  {avg_imp_f1:+.4f} ({(avg_imp_f1/comp_df['RF_Test_F1'].mean()*100):+.2f}%)")
    
    best_split_acc = comp_df.loc[comp_df['Improvement_Acc'].idxmax(), 'Split']
    best_split_f1 = comp_df.loc[comp_df['Improvement_F1'].idxmax(), 'Split']
    
    print(f"\nBest Split for Hybrid:")
    print(f"  • Accuracy:  {best_split_acc}")
    print(f"  • F1-Score:  {best_split_f1}")
    
    if avg_imp_acc > 0 and avg_imp_f1 > 0:
        print(f"\n✅ CONCLUSION: Hybrid MCCBF+RF consistently outperforms RF+TF-IDF")
    elif avg_imp_acc < 0 and avg_imp_f1 < 0:
        print(f"\n❌ CONCLUSION: RF+TF-IDF consistently outperforms Hybrid MCCBF+RF")
    else:
        print(f"\n⚖️ CONCLUSION: Mixed results - model choice depends on use case")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()