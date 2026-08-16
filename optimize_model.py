import os
import itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def prepare_data(data_path):
    """Loads preprocessed data and performs chronological split."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Verify label column exists
    if 'label' not in df.columns:
        raise ValueError("Label column missing from dataset.")
    
    y = df['label'].values
    X = df.drop(columns=['label']).values
    
    # 1. Chronological split: 80% train, 10% val, 10% test
    # Data is not shuffled to preserve temporal order
    n_samples = len(df)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(y_true, y_pred, y_scores):
    """Calculates precision, recall, f1, and roc_auc."""
    # IsolationForest outputs -1 for outliers (fraud) and 1 for inliers (normal)
    # We map -1 to 1 (fraud) and 1 to 0 (normal) to match standard evaluation metrics
    y_pred_mapped = np.where(y_pred == -1, 1, 0)
    
    # score_samples returns negative anomaly score (lower means more anomalous)
    # Negate it so that higher scores represent higher anomaly probability for ROC-AUC
    y_scores_mapped = -y_scores
    
    precision = precision_score(y_true, y_pred_mapped, zero_division=0)
    recall = recall_score(y_true, y_pred_mapped, zero_division=0)
    f1 = f1_score(y_true, y_pred_mapped, zero_division=0)
    
    # ROC-AUC calculation requires at least one positive and one negative example
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_scores_mapped)
    else:
        roc_auc = np.nan
        
    return precision, recall, f1, roc_auc

def grid_search(X_train, X_val, y_val):
    """Runs a grid search over specified Isolation Forest parameters."""
    n_estimators = [100, 200, 300]
    max_samples = [0.5, 0.75, 1.0]
    contamination = [0.01, 0.02, 0.05]
    
    # 2. Generate all 27 combinations
    param_grid = list(itertools.product(n_estimators, max_samples, contamination))
    print(f"\nRunning grid search over {len(param_grid)} combinations...")
    
    results = []
    
    for n_est, max_samp, contam in param_grid:
        model = IsolationForest(
            n_estimators=n_est,
            max_samples=max_samp,
            contamination=contam,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train)
        
        y_pred = model.predict(X_val)
        y_scores = model.score_samples(X_val)
        
        # 3. Calculate metrics on validation set
        precision, recall, f1, roc_auc = evaluate_model(y_val, y_pred, y_scores)
        
        results.append({
            'n_estimators': n_est,
            'max_samples': max_samp,
            'contamination': contam,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
    # 4. Create results DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def main():
    DATA_PATH = "data/processed/user_01_preprocessed.csv"
    RESULTS_DIR = "results"
    MODELS_DIR = "models"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Prepare data (chronological split)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(DATA_PATH)
    
    # Run Grid Search
    results_df = grid_search(X_train, X_val, y_val)
    
    # 5. Save results to CSV
    results_path = os.path.join(RESULTS_DIR, "user_01_isolation_forest_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved grid search results to {results_path}")
    
    # 6. Select best model based primarily on F1, tie-breaker: Recall
    best_row = results_df.sort_values(by=['f1', 'recall'], ascending=[False, False]).iloc[0]
    best_params = {
        'n_estimators': int(best_row['n_estimators']),
        'max_samples': float(best_row['max_samples']),
        'contamination': float(best_row['contamination'])
    }
    
    print("\n--- Best Validation Parameters ---")
    print(best_params)
    print(f"Validation F1: {best_row['f1']:.4f}")
    
    # 7. Retrain on train + validation data
    X_train_val = np.vstack((X_train, X_val))
    print(f"\nRetraining best model on train + validation data ({len(X_train_val)} samples)...")
    
    final_model = IsolationForest(
        n_estimators=best_params['n_estimators'],
        max_samples=best_params['max_samples'],
        contamination=best_params['contamination'],
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train_val)
    
    # 8. Evaluate final model on test set
    y_test_pred = final_model.predict(X_test)
    y_test_scores = final_model.score_samples(X_test)
    
    test_precision, test_recall, test_f1, test_roc_auc = evaluate_model(y_test, y_test_pred, y_test_scores)
    
    # Re-map predictions for confusion matrix calculation
    y_test_pred_mapped = np.where(y_test_pred == -1, 1, 0)
    cm = confusion_matrix(y_test, y_test_pred_mapped)
    
    print("\n--- Final Test Set Evaluation ---")
    print(f"Best Parameters: {best_params}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    if not np.isnan(test_roc_auc):
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    else:
        print("Test ROC-AUC: N/A (Only 1 class present in test set)")
    print("Confusion Matrix:")
    print(cm)
    
    # 9. Save final model
    model_path = os.path.join(MODELS_DIR, "user_01_isolation_forest.joblib")
    joblib.dump(final_model, model_path)
    print(f"\nFinal model saved to {model_path}")

if __name__ == "__main__":
    main()
