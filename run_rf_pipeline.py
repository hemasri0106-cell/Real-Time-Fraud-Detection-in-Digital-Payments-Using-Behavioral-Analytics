import pandas as pd
import numpy as np
import os
import joblib
import json
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def load_and_preprocess():
    print("--- 1. Loading Data and Preprocessing ---")
    df = pd.read_csv('data/user_datasets/user_01_transactions_v2.csv')
    y = df['label'].values
    X = df.drop(columns=['label', 'transaction_id', 'user_id', 'timestamp'])
    
    n_samples = len(df)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    # Handle boolean conversion
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X_train.loc[:, col] = X_train[col].astype(int)
        X_val.loc[:, col] = X_val[col].astype(int)
        X_test.loc[:, col] = X_test[col].astype(int)
        
    preprocessor = joblib.load('models/preprocessor_user_01_v2.joblib')
    
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    with open('models/feature_names_user_01_v2.json', 'r') as f:
        feature_names = json.load(f)
        
    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test, feature_names

def run_grid_search(X_train, y_train, X_val, y_val):
    print("--- 2. Running Random Forest Grid Search ---")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ["sqrt", "log2"],
        'class_weight': ["balanced", "balanced_subsample"]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total configurations to evaluate: {len(combinations)}")
    
    results = []
    
    for i, params in enumerate(combinations):
        if i % 50 == 0:
            print(f"Evaluated {i}/{len(combinations)} configurations...")
            
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        
        res = params.copy()
        res.update({'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr_auc})
        results.append(res)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/user_01_v2_random_forest_results.csv', index=False)
    
    # Select best
    best_row = results_df.sort_values(by=['f1', 'recall', 'pr_auc'], ascending=[False, False, False]).iloc[0]
    print(f"\nGrid search completed.")
    print(f"Best Validation F1: {best_row['f1']:.4f}")
    
    best_params = {k: best_row[k] for k in keys}
    if pd.isna(best_params['max_depth']):
        best_params['max_depth'] = None
    for k in ['n_estimators', 'min_samples_split', 'min_samples_leaf']:
        best_params[k] = int(best_params[k])
        
    return best_params, best_row

def final_training_and_eval(best_params, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    print("\n--- 3. Final Training and Evaluation ---")
    
    import scipy.sparse as sp
    if sp.issparse(X_train):
        X_train_val = sp.vstack([X_train, X_val])
    else:
        X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"Retraining final model on {X_train_val.shape[0]} samples...")
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)
    
    joblib.dump(final_model, 'models/user_01_v2_random_forest.joblib')
    
    # Evaluate on Test
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    fraud_in_test = sum(y_test)
    fraud_detected = cm[1, 1] if cm.shape == (2,2) else 0
    
    print("\n[Final Test Metrics]")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Fraud in Test: {fraud_in_test}")
    print(f"Fraud Detected: {fraud_detected}")
    print("Confusion Matrix:")
    print(cm)
    
    # Feature Importance
    importances = final_model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    fi_df.head(20).to_csv('results/user_01_v2_random_forest_feature_importance.csv', index=False)
    print("\nTop 10 Feature Importances:")
    print(fi_df.head(10).to_string(index=False))
    
    print("\n--- 4. Comparison with Isolation Forest ---")
    print("Isolation Forest Test F1: 0.3431")
    print(f"Random Forest Test F1:    {f1:.4f}")
    if f1 > 0.3431:
        print(f"Random Forest improved F1 by {f1 - 0.3431:.4f}!")
    else:
        print("Random Forest did not improve upon Isolation Forest.")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, fn = load_and_preprocess()
    best_params, best_row = run_grid_search(X_train, y_train, X_val, y_val)
    
    print("\n[Best Validation Metrics]")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall:    {best_row['recall']:.4f}")
    print(f"F1:        {best_row['f1']:.4f}")
    print(f"ROC-AUC:   {best_row['roc_auc']:.4f}")
    print(f"PR-AUC:    {best_row['pr_auc']:.4f}")
    print(f"Parameters: {best_params}")
    
    final_training_and_eval(best_params, X_train, y_train, X_val, y_val, X_test, y_test, fn)
