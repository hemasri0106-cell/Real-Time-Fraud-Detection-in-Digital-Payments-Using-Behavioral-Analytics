import pandas as pd
import numpy as np
import os
import joblib
import json
import itertools
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import scipy.sparse as sp

np.random.seed(42)

def generate_data(user_id, num_samples=20000):
    print(f"\n--- Generating Data for {user_id} ---")
    
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=int(x)) for x in np.cumsum(np.random.exponential(scale=30, size=num_samples))]
    
    data = []
    
    known_devices = [f'dev_{user_id}_1', f'dev_{user_id}_2']
    known_locations = [f'loc_{user_id}_1', f'loc_{user_id}_2', f'loc_{user_id}_3']
    known_merchants = [f'merch_{user_id}_{i}' for i in range(20)]
    categories = ['grocery', 'retail', 'dining', 'entertainment', 'travel', 'electronics']
    payment_methods = ['credit_card', 'debit_card', 'digital_wallet']
    
    last_tx_time = None
    daily_count = 0
    current_day = None
    recent_amounts = []
    
    for i in range(num_samples):
        ts = timestamps[i]
        is_fraud = np.random.random() < 0.08 # ~8% fraud
        
        if current_day != ts.date():
            current_day = ts.date()
            daily_count = 0
        daily_count += 1
        
        if last_tx_time: gap_mins = (ts - last_tx_time).total_seconds() / 60.0
        else: gap_mins = 1440.0
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
            is_new_dev, is_new_loc, is_new_merch = 0, 0, np.random.choice([0, 1], p=[0.95, 0.05])
            dist = np.random.exponential(scale=5)
            
            if np.random.random() < 0.02: amount = np.random.uniform(300, 800)
            if np.random.random() < 0.05: gap_mins = np.random.uniform(1, 5)
        else:
            pattern = np.random.choice(['large_amount', 'burst_frequency', 'new_device_loc', 'ato_known_device', 'weak_signals'])
            
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
                gap_mins = np.random.uniform(1, 10)
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
        if len(recent_amounts) > 50: recent_amounts.pop(0)
            
        data.append({
            'transaction_id': f"tx_{i}",
            'user_id': user_id,
            'timestamp': ts.strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_amount': amount,
            'merchant_category': category,
            'merchant_id': merchant,
            'payment_method': method,
            'device_id': device,
            'device_type': 'mobile' if 'dev' in device else 'desktop',
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
    df.to_csv(f'data/user_datasets/{user_id}_transactions_v2.csv', index=False)
    print(f"Saved dataset for {user_id}: {len(df)} rows, {df['label'].sum()} frauds.")
    return df

def preprocess_and_train(df, user_id):
    print(f"\n--- Preprocessing & Training for {user_id} ---")
    y = df['label'].values
    X = df.drop(columns=['label', 'transaction_id', 'user_id', 'timestamp'])
    
    n_samples = len(df)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_train, y_train = X.iloc[:train_end].copy(), y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end].copy(), y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:].copy(), y[val_end:]
    
    # Avoid warning by specifying bool specifically
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X_train.loc[:, col] = X_train[col].astype(int)
        X_val.loc[:, col] = X_val[col].astype(int)
        X_test.loc[:, col] = X_test[col].astype(int)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    # explicitly include str per the pandas 3 deprecation warning we saw earlier
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

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
        ], remainder='drop'
    )

    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out().tolist()
    joblib.dump(preprocessor, f'models/preprocessor_{user_id}_v2.joblib')
    with open(f'models/feature_names_{user_id}_v2.json', 'w') as f:
        json.dump(feature_names, f)
        
    # Grid Search - Optimized down to the winning combinations from User 01 
    # to avoid a 15-minute wait time across 3 users.
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [15, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1],
        'max_features': ["sqrt"],
        'class_weight': ["balanced"]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_f1 = -1
    best_params = None
    
    for params in combinations:
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_val_trans)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()
            
    print(f"Best Validation F1: {best_f1:.4f}")
    
    # Final Retrain
    if sp.issparse(X_train_trans):
        X_train_val = sp.vstack([X_train_trans, X_val_trans])
    else:
        X_train_val = np.vstack([X_train_trans, X_val_trans])
    y_train_val = np.concatenate([y_train, y_val])
    
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)
    joblib.dump(final_model, f'models/{user_id}_v2_random_forest.joblib')
    
    y_pred_test = final_model.predict(X_test_trans)
    y_prob_test = final_model.predict_proba(X_test_trans)[:, 1]
    
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    roc_test = roc_auc_score(y_test, y_prob_test)
    pr_test = average_precision_score(y_test, y_prob_test)
    
    print(f"Test Precision: {prec_test:.4f}")
    print(f"Test Recall:    {rec_test:.4f}")
    print(f"Test F1:        {f1_test:.4f}")
    print(f"Test ROC-AUC:   {roc_test:.4f}")
    print(f"Test PR-AUC:    {pr_test:.4f}")
    
    return {
        'f1': f1_test,
        'precision': prec_test,
        'recall': rec_test,
        'roc_auc': roc_test,
        'pr_auc': pr_test
    }

def main():
    users = ['user_05', 'user_06', 'user_07', 'user_08', 'user_09', 'user_10']
    results = {}
    for u in users:
        df = generate_data(u)
        metrics = preprocess_and_train(df, u)
        results[u] = metrics
        
    print("\n===========================================")
    print("      FINAL TEST SET RESULTS SUMMARY      ")
    print("===========================================")
    
    # Create a nice dataframe summary
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    
    avg_f1 = summary_df['f1'].mean()
    print(f"\nAverage Test F1 across all users: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
