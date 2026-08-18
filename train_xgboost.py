import pandas as pd
import numpy as np
import os
import joblib
import json
import itertools
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def load_and_preprocess(user_id):
    print(f"\n--- Loading Data and Preprocessing for {user_id} ---")
    df = pd.read_csv(f'data/user_datasets/{user_id}_transactions_v2.csv')
    y = df['label'].values
    X = df.drop(columns=['label', 'transaction_id', 'user_id', 'timestamp'])
    
    n_samples = len(df)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_train, y_train = X.iloc[:train_end].copy(), y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end].copy(), y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:].copy(), y[val_end:]
    
    # Handle boolean conversion
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X_train[col] = X_train[col].astype(int)
        X_val[col] = X_val[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        
    preprocessor = joblib.load(f'models/preprocessor_{user_id}_v2.joblib')
    
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    with open(f'models/feature_names_{user_id}_v2.json', 'r') as f:
        feature_names = json.load(f)
        
    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test, feature_names

def run_grid_search(user_id, X_train, y_train, X_val, y_val):
    print(f"\n--- Running XGBoost Grid Search for {user_id} ---")
    
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total configurations to evaluate: {len(combinations)}")
    
    results = []
    
    for i, params in enumerate(combinations):
        if i % 20 == 0:
            print(f"Evaluated {i}/{len(combinations)} configurations...")
            
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            early_stopping_rounds=20
        )
        
        # XGBoost handles early stopping via the fit method
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        
        res = params.copy()
        res.update({
            'precision': prec, 
            'recall': rec, 
            'f1': f1, 
            'roc_auc': roc, 
            'pr_auc': pr_auc,
            'best_iteration': model.best_iteration
        })
        results.append(res)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{user_id}_v2_xgboost_results.csv', index=False)
    
    # Select best based on F1, Recall, PR-AUC, ROC-AUC
    best_row = results_df.sort_values(by=['f1', 'recall', 'pr_auc', 'roc_auc'], ascending=[False, False, False, False]).iloc[0]
    print(f"\nGrid search completed for {user_id}.")
    print(f"Best Validation F1: {best_row['f1']:.4f}")
    
    best_params = {k: best_row[k] for k in keys}
    for k in ['n_estimators', 'max_depth', 'min_child_weight']:
        best_params[k] = int(best_params[k])
    
    return best_params, int(best_row['best_iteration']), scale_pos_weight, best_row

def final_training_and_eval(user_id, best_params, best_iteration, scale_pos_weight, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    print(f"\n--- Final Training and Evaluation for {user_id} ---")
    
    import scipy.sparse as sp
    if sp.issparse(X_train):
        X_train_val = sp.vstack([X_train, X_val])
    else:
        X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"Retraining final model on {X_train_val.shape[0]} samples with n_estimators={best_iteration + 1}...")
    
    # Use best_iteration + 1 as n_estimators because best_iteration is 0-indexed
    final_params = best_params.copy()
    final_params['n_estimators'] = best_iteration + 1
    
    final_model = XGBClassifier(
        **final_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    final_model.fit(X_train_val, y_train_val)
    
    joblib.dump(final_model, f'models/{user_id}_v2_xgboost.joblib')
    
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
    total_test = len(y_test)
    
    print(f"\n[{user_id} Final Test Metrics]")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Total Test Transactions: {total_test}")
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
    
    fi_df.head(20).to_csv(f'results/{user_id}_v2_xgboost_feature_importance.csv', index=False)
    
    return {
        'user_id': user_id,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'pr_auc': pr_auc
    }

def generate_comparison(xgb_results):
    print("\n--- Comparing with Random Forest ---")
    if not os.path.exists('results/final_rf_metrics.csv'):
        print("RF metrics not found, skipping comparison.")
        return
        
    rf_df = pd.read_csv('results/final_rf_metrics.csv')
    
    comparison_data = []
    
    for xgb_res in xgb_results:
        user_id = xgb_res['user_id']
        rf_row = rf_df[rf_df['user_id'] == user_id]
        
        if not rf_row.empty:
            rf_row = rf_row.iloc[0]
            
            # XGBoost
            comparison_data.append({
                'User': user_id,
                'Model': 'XGBoost',
                'Precision': xgb_res['precision'],
                'Recall': xgb_res['recall'],
                'F1': xgb_res['f1'],
                'ROC-AUC': xgb_res['roc_auc'],
                'PR-AUC': xgb_res['pr_auc']
            })
            
            # Random Forest
            comparison_data.append({
                'User': user_id,
                'Model': 'Random Forest',
                'Precision': rf_row['test_precision'],
                'Recall': rf_row['test_recall'],
                'F1': rf_row['test_f1'],
                'ROC-AUC': rf_row['test_roc_auc'],
                'PR-AUC': rf_row['test_pr_auc']
            })
            
            # Flag better model
            if xgb_res['f1'] > rf_row['test_f1']:
                print(f"{user_id}: XGBoost is BETTER (F1: {xgb_res['f1']:.4f} vs {rf_row['test_f1']:.4f})")
            elif xgb_res['f1'] < rf_row['test_f1']:
                print(f"{user_id}: Random Forest is BETTER (F1: {rf_row['test_f1']:.4f} vs {xgb_res['f1']:.4f})")
            else:
                print(f"{user_id}: TIED on F1 ({xgb_res['f1']:.4f})")
                
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv('results/xgboost_vs_random_forest_comparison.csv', index=False)
    print("\nComparison saved to results/xgboost_vs_random_forest_comparison.csv")

if __name__ == "__main__":
    users = [f"user_{str(i).zfill(2)}" for i in range(3, 11)]
    all_xgb_results = []
    all_best_params = []
    all_best_iterations = []
    
    for uid in users:
        X_train, y_train, X_val, y_val, X_test, y_test, fn = load_and_preprocess(uid)
        best_params, best_iteration, scale_pos_weight, best_row = run_grid_search(uid, X_train, y_train, X_val, y_val)
        
        print(f"\n[{uid} Best Validation Metrics]")
        print(f"F1: {best_row['f1']:.4f}")
        print(f"Parameters: {best_params}")
        print(f"Best Iteration: {best_iteration}")
        
        all_best_params.append(best_params)
        all_best_iterations.append(best_iteration)
        
        test_metrics = final_training_and_eval(uid, best_params, best_iteration, scale_pos_weight, X_train, y_train, X_val, y_val, X_test, y_test, fn)
        all_xgb_results.append(test_metrics)
        
    generate_comparison(all_xgb_results)
    
    # Save results and best hyperparameters to a text file
    with open('results/xgboost_users_03_to_10_summary.txt', 'w') as f:
        f.write("XGBoost Results and Best Hyperparameters for Users 03-10\n")
        f.write("="*60 + "\n\n")
        for res, params, iteration in zip(all_xgb_results, all_best_params, all_best_iterations):
            f.write(f"--- {res['user_id']} ---\n")
            f.write(f"Test F1:      {res['f1']:.4f}\n")
            f.write(f"Test Recall:  {res['recall']:.4f}\n")
            f.write(f"Test Prec:    {res['precision']:.4f}\n")
            f.write(f"Test ROC-AUC: {res['roc_auc']:.4f}\n")
            f.write(f"Test PR-AUC:  {res['pr_auc']:.4f}\n")
            f.write("Best Hyperparameters:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"  best_iteration (early stopping): {iteration}\n\n")
        f.write("End of Summary.\n")
        print("\nSaved summary to results/xgboost_users_03_to_10_summary.txt")

