import pandas as pd

df = pd.read_csv('results/user_01_isolation_forest_results.csv')

def print_top(sort_col, tie_breaker=None):
    if tie_breaker:
        sorted_df = df.sort_values(by=[sort_col, tie_breaker], ascending=[False, False])
    else:
        sorted_df = df.sort_values(by=[sort_col], ascending=[False])
    print(sorted_df.head(10)[['n_estimators', 'max_samples', 'contamination', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    return sorted_df.iloc[0]

print("=== 1. Top 10 by Validation F1 ===")
best_f1 = print_top('f1', 'recall')

print("\n=== 2. Top 10 by Validation Recall ===")
best_recall = print_top('recall', 'f1')

print("\n=== 3. Top 10 by Validation ROC-AUC ===")
best_roc = print_top('roc_auc')

print("\n--- Best Configurations ---")
print(f"Best by F1: n_estimators={int(best_f1['n_estimators'])}, max_samples={best_f1['max_samples']}, contamination={best_f1['contamination']} (F1={best_f1['f1']:.4f})")
print(f"Best by Recall: n_estimators={int(best_recall['n_estimators'])}, max_samples={best_recall['max_samples']}, contamination={best_recall['contamination']} (Recall={best_recall['recall']:.4f})")
print(f"Best by ROC-AUC: n_estimators={int(best_roc['n_estimators'])}, max_samples={best_roc['max_samples']}, contamination={best_roc['contamination']} (ROC-AUC={best_roc['roc_auc']:.4f})")
