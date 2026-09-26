import pandas as pd
import numpy as np
import os
import joblib
import json
import itertools
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

np.random.seed(42)

def generate_data(num_samples=20000):
    print("--- PART 1: Data Generation ---")
    
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=int(x)) for x in np.cumsum(np.random.exponential(scale=30, size=num_samples))]
    
    # Base user stats
    user_id = 'user_01'
    
    data = []
    
    known_devices = ['dev_1', 'dev_2']
    known_locations = ['loc_1', 'loc_2', 'loc_3']
    known_merchants = [f'merch_{i}' for i in range(20)]
    categories = ['grocery', 'retail', 'dining', 'entertainment', 'travel', 'electronics']
    payment_methods = ['credit_card', 'debit_card', 'digital_wallet']
    
    fraud_patterns = {
        'large_amount': 0,
        'burst_frequency': 0,
        'new_device_loc': 0,
        'ato_known_device': 0,
        'weak_signals': 0
    }
    
    # Track state for rolling features
    last_tx_time = None
    daily_count = 0
    current_day = None
    recent_amounts = []
    
    for i in range(num_samples):
        ts = timestamps[i]
        is_fraud = np.random.random() < 0.08 # ~8% fraud
        
        # State updates
        if current_day != ts.date():
            current_day = ts.date()
            daily_count = 0
        daily_count += 1
        
        if last_tx_time:
            gap_mins = (ts - last_tx_time).total_seconds() / 60.0
        else:
            gap_mins = 1440.0
            
        last_tx_time = ts
        
        avg_amt_7d = np.mean(recent_amounts) if recent_amounts else 50.0
        std_amt_7d = np.std(recent_amounts) if len(recent_amounts) > 1 else 10.0
        
        if not is_fraud:
            amount = np.clip(np.random.normal(50, 20), 1, 300)
            merchant = np.random.choice(known_merchants)
            category = np.random.choice(categories, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
            method = np.random.choice(payment_methods)
            device = np.random.choice(known_devices)
            location = np.random.choice(known_locations)
            is_new_dev = 0
            is_new_loc = 0
            is_new_merch = np.random.choice([0, 1], p=[0.95, 0.05])
            dist = np.random.exponential(scale=5)
            
            # small chance of unusual behavior in normal
            if np.random.random() < 0.02: amount = np.random.uniform(300, 800)
            if np.random.random() < 0.05: gap_mins = np.random.uniform(1, 5)
            
            pattern_name = 'normal'
            
        else:
            pattern = np.random.choice(list(fraud_patterns.keys()))
            fraud_patterns[pattern] += 1
            
            if pattern == 'large_amount':
                amount = np.random.uniform(1000, 5000)
                merchant = np.random.choice(known_merchants)
                category = 'electronics'
                method = np.random.choice(payment_methods)
                device = np.random.choice(known_devices)
                location = np.random.choice(known_locations)
                is_new_dev, is_new_loc, is_new_merch = 0, 0, 0
                dist = np.random.exponential(scale=5)
            elif pattern == 'burst_frequency':
                amount = np.random.normal(50, 20)
                merchant = np.random.choice(known_merchants)
                category = 'digital_goods' if np.random.random() < 0.5 else 'retail'
                method = np.random.choice(payment_methods)
                device = np.random.choice(known_devices)
                location = np.random.choice(known_locations)
                is_new_dev, is_new_loc, is_new_merch = 0, 0, 0
                dist = np.random.exponential(scale=5)
                gap_mins = np.random.uniform(0.1, 2.0)
                daily_count += np.random.randint(5, 20)
            elif pattern == 'new_device_loc':
                amount = np.random.uniform(200, 1000)
                merchant = f"new_merch_{i}"
                category = np.random.choice(categories)
                method = 'credit_card'
                device = f"new_dev_{i}"
                location = f"new_loc_{i}"
                is_new_dev, is_new_loc, is_new_merch = 1, 1, 1
                dist = np.random.uniform(500, 2000)
            elif pattern == 'ato_known_device':
                amount = np.random.uniform(500, 2000)
                merchant = f"new_merch_{i}"
                category = 'travel'
                method = np.random.choice(payment_methods)
                device = np.random.choice(known_devices)
                location = np.random.choice(known_locations)
                is_new_dev, is_new_loc, is_new_merch = 0, 0, 1
                dist = np.random.exponential(scale=5)
                gap_mins = np.random.uniform(1, 10) # slightly fast
            elif pattern == 'weak_signals':
                amount = np.random.uniform(200, 400)
                merchant = np.random.choice(known_merchants)
                category = 'retail'
                method = 'digital_wallet'
                device = np.random.choice(known_devices)
                location = np.random.choice(known_locations)
                is_new_dev, is_new_loc, is_new_merch = 0, 0, 0
                dist = np.random.uniform(50, 200)
                gap_mins = np.random.uniform(10, 30)

        recent_amounts.append(amount)
        if len(recent_amounts) > 50:
            recent_amounts.pop(0)
            
        data.append({
            'transaction_id': f"tx_{i}",
            'user_id': user_id,
            'timestamp': ts.strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_amount': amount,
            'merchant_category': category,
            'merchant_id': merchant,
            'payment_method': method,
            'device_id': device,
            'device_type': 'mobile' if 'dev_1' in device else 'desktop',
            'city': location,
            'hour_of_day': ts.hour,
            'day_of_week': ts.weekday(),
            'is_weekend': 1 if ts.weekday() >= 5 else 0,
            'transaction_gap_minutes': gap_mins,
            'daily_transaction_count': daily_count,
            'average_amount_last_7_days': avg_amt_7d,
            'std_amount_last_7_days': std_amt_7d,
            'merchant_visit_frequency': np.random.uniform(0, 1),
            'device_usage_frequency': np.random.uniform(0.5, 1.0) if not is_new_dev else 0.0,
            'location_visit_frequency': np.random.uniform(0.5, 1.0) if not is_new_loc else 0.0,
            'new_device': is_new_dev,
            'new_location': is_new_loc,
            'new_merchant': is_new_merch,
            'distance_from_last_transaction_km': dist,
            'label': 1 if is_fraud else 0
        })

    df = pd.DataFrame(data)
    
    os.makedirs('data/user_datasets', exist_ok=True)
    df.to_csv('data/user_datasets/user_01_transactions_v2.csv', index=False)
    
    fraud_count = df['label'].sum()
    normal_count = len(df) - fraud_count
    print(f"Total rows: {len(df)}")
    print(f"Normal count: {normal_count} ({normal_count/len(df)*100:.2f}%)")
    print(f"Fraud count: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("Fraud patterns:", fraud_patterns)
    
    return df

def preprocess_data(df):
    print("\n--- PART 2: Preprocessing ---")
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

    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit ONLY on training data
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out().tolist()
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(preprocessor, 'models/preprocessor_user_01_v2.joblib')
    with open('models/feature_names_user_01_v2.json', 'w') as f:
        json.dump(feature_names, f)
        
    df_trans = pd.DataFrame(X_train_trans, columns=feature_names)
    df_trans['label'] = y_train
    df_trans.to_csv('data/processed/user_01_v2_preprocessed.csv', index=False)
    
    print("Preprocessing completed and saved.")
    
    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test

def optimize_model(X_train, X_val, y_val):
    print("\n--- PART 3: Isolation Forest Grid Search ---")
    n_estimators = [100, 200, 300]
    max_samples = [0.5, 0.75, 1.0]
    contamination = [0.01, 0.02, 0.05, 0.10]
    
    param_grid = list(itertools.product(n_estimators, max_samples, contamination))
    
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
        y_scores = -model.score_samples(X_val) # higher = more anomalous
        
        y_pred_mapped = np.where(y_pred == -1, 1, 0)
        
        precision = precision_score(y_val, y_pred_mapped, zero_division=0)
        recall = recall_score(y_val, y_pred_mapped, zero_division=0)
        f1 = f1_score(y_val, y_pred_mapped, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_scores) if len(np.unique(y_val)) > 1 else np.nan
        
        results.append({
            'n_estimators': n_est,
            'max_samples': max_samp,
            'contamination': contam,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/user_01_v2_isolation_forest_results.csv', index=False)
    
    # Select best model
    best_row = results_df.sort_values(by=['f1', 'recall'], ascending=[False, False]).iloc[0]
    
    best_model = IsolationForest(
        n_estimators=int(best_row['n_estimators']),
        max_samples=float(best_row['max_samples']),
        contamination=float(best_row['contamination']),
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train)
    joblib.dump(best_model, 'models/user_01_v2_isolation_forest.joblib')
    
    print(f"Grid search completed. Best parameters: n_est={int(best_row['n_estimators'])}, max_samp={best_row['max_samples']}, contam={best_row['contamination']}")
    print(f"Validation F1: {best_row['f1']:.4f}")
    
    return best_model

def tune_threshold(model, X_val, y_val):
    print("\n--- PART 4: Threshold Tuning ---")
    val_scores = -model.score_samples(X_val)
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 500)
    
    best_f1 = -1
    best_thresh = None
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (val_scores >= thresh).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1 or (f1 == best_f1 and rec > best_metrics.get('recall', -1)):
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {'precision': prec, 'recall': rec, 'f1': f1}
            
    with open('models/user_01_v2_threshold.json', 'w') as f:
        json.dump({'best_threshold': float(best_thresh)}, f)
        
    print(f"Selected Threshold: {best_thresh:.4f}")
    print(f"Validation F1 at threshold: {best_metrics['f1']:.4f}")
    
    return best_thresh, best_metrics

def evaluate_test(model, threshold, X_test, y_test, val_metrics):
    print("\n--- PART 5: Final Test Evaluation ---")
    test_scores = -model.score_samples(X_test)
    y_pred = (test_scores >= threshold).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_scores) if len(np.unique(y_test)) > 1 else np.nan
    cm = confusion_matrix(y_test, y_pred)
    
    fraud_in_test = sum(y_test)
    fraud_detected = cm[1, 1] if cm.shape == (2,2) else 0
    
    print(f"\n[Validation Metrics] Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
    
    print(f"\n[Test Metrics]")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Fraud in Test: {fraud_in_test}")
    print(f"Fraud Detected: {fraud_detected}")
    print("Confusion Matrix:")
    print(cm)
    
    print("\n--- PART 6: Final Assessment ---")
    if f1 >= 0.90:
        print("Assessment: SUCCESS")
        print("The model achieved strong generalization with F1 >= 0.90.")
    else:
        print("Assessment: BOTTLENECK")
        print("The Isolation Forest failed to reach the target F1 of 0.90.")
        print("Explanation: The new synthetic dataset includes complex, varied fraud behaviors (like ATOs with known devices, transaction bursts, and weak signal combinations). Isolation Forest is an unsupervised anomaly detector that only looks for data points far from the norm. When fraud closely mimics normal behavior but in unique combinations, it overlaps massively with normal transactions. Unsupervised learning cannot draw precise decision boundaries around complex, adversarial classes. A supervised model (like XGBoost) is required to capture these interactions effectively.")

def main():
    df = generate_data(20000)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df)
    model = optimize_model(X_train, X_val, y_val)
    threshold, val_metrics = tune_threshold(model, X_val, y_val)
    evaluate_test(model, threshold, X_test, y_test, val_metrics)

if __name__ == "__main__":
    main()
