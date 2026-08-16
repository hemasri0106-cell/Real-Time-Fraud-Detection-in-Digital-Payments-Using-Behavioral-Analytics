import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    y = df['label'].values
    X = df.drop(columns=['label']).values
    
    n_samples = len(df)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_val, y_val, X_test, y_test

def main():
    # Load data
    DATA_PATH = "data/processed/user_01_preprocessed.csv"
    X_val, y_val, X_test, y_test = prepare_data(DATA_PATH)
    
    # Load model
    model = joblib.load('models/user_01_isolation_forest.joblib')
    
    # 1. Generate anomaly scores for validation set
    # negate so higher = more anomalous
    val_scores = -model.score_samples(X_val)
    
    # 3. Test a range of thresholds
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 500)
    
    results = []
    for thresh in thresholds:
        # 2. Don't use contamination. Use raw score threshold.
        # If score >= threshold, then fraud (1), else normal (0)
        y_val_pred = (val_scores >= thresh).astype(int)
        
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
    results_df = pd.DataFrame(results)
    
    # 6. Print top 10 thresholds and validation metrics
    print("=== Top 10 Thresholds by Validation F1 ===")
    top_10 = results_df.sort_values(by=['f1', 'recall'], ascending=[False, False]).head(10)
    print(top_10.to_string(index=False))
    
    # 5. Select the best threshold
    best_row = top_10.iloc[0]
    best_threshold = best_row['threshold']
    print(f"\nSelected Best Threshold: {best_threshold:.4f} (Val F1: {best_row['f1']:.4f})")
    
    # 7. Save the selected threshold
    with open('models/user_01_threshold.json', 'w') as f:
        json.dump({'best_threshold': float(best_threshold)}, f)
    
    # 8. Apply frozen threshold to test set
    test_scores = -model.score_samples(X_test)
    y_test_pred = (test_scores >= best_threshold).astype(int)
    
    # Calculate metrics
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    if len(np.unique(y_test)) > 1:
        test_roc_auc = roc_auc_score(y_test, test_scores)
    else:
        test_roc_auc = np.nan
        
    cm = confusion_matrix(y_test, y_test_pred)
    
    # 9. Report
    print("\n=== Final Test Set Evaluation ===")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
if __name__ == "__main__":
    main()
