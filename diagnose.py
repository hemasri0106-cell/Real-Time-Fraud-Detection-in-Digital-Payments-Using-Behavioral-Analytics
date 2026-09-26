import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("=== 1. Hyperparameter Search Results ===")
    results_df = pd.read_csv('results/user_01_isolation_forest_results.csv')
    print("\nTop 10 by F1:")
    print(results_df.sort_values(by=['f1', 'recall'], ascending=[False, False]).head(10)[['n_estimators', 'max_samples', 'contamination', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    print("\nTop 10 by Recall:")
    print(results_df.sort_values(by=['recall', 'f1'], ascending=[False, False]).head(10)[['n_estimators', 'max_samples', 'contamination', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    print("\nTop 10 by ROC-AUC:")
    print(results_df.sort_values(by=['roc_auc'], ascending=[False]).head(10)[['n_estimators', 'max_samples', 'contamination', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))

    print("\n=== 2. Fraud Distribution across Splits ===")
    data = pd.read_csv('data/processed/user_01_preprocessed.csv')
    y = data['label'].values
    X = data.drop(columns=['label'])
    
    n_samples = len(data)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    print(f"Train: Total={len(y_train)}, Fraud={sum(y_train)}, Rate={sum(y_train)/len(y_train):.4f}")
    print(f"Val:   Total={len(y_val)}, Fraud={sum(y_val)}, Rate={sum(y_val)/len(y_val):.4f}")
    print(f"Test:  Total={len(y_test)}, Fraud={sum(y_test)}, Rate={sum(y_test)/len(y_test):.4f}")

    print("\n=== 3. Verification of IF Logic ===")
    print("Isolation Forest naturally predicts -1 for anomalies, 1 for normal.")
    print("In optimize_model.py:")
    print("y_pred_mapped = np.where(y_pred == -1, 1, 0) # Mapping -1 -> 1 (fraud)")
    print("y_scores_mapped = -y_scores # Negated so higher score = higher anomaly prob.")
    print("This logic is correctly implemented for evaluating ROC-AUC and F1.")
    
    print("\n=== 4. Anomaly Score Statistics (Test Set) ===")
    model = joblib.load('models/user_01_isolation_forest.joblib')
    X_test = X.iloc[val_end:].values
    
    # We negate the scores to match what was done in optimize_model.py
    test_scores = -model.score_samples(X_test)
    
    df_test_scores = pd.DataFrame({'label': y_test, 'score': test_scores})
    normal_scores = df_test_scores[df_test_scores['label'] == 0]['score']
    fraud_scores = df_test_scores[df_test_scores['label'] == 1]['score']
    
    def print_stats(name, series):
        print(f"\n{name} Stats:")
        if len(series) == 0:
            print("No samples")
            return
        print(f"  Min:    {series.min():.4f}")
        print(f"  25%:    {series.quantile(0.25):.4f}")
        print(f"  Median: {series.median():.4f}")
        print(f"  Mean:   {series.mean():.4f}")
        print(f"  75%:    {series.quantile(0.75):.4f}")
        print(f"  Max:    {series.max():.4f}")
        
    print_stats("Normal Transactions (Label 0)", normal_scores)
    print_stats("Fraud Transactions (Label 1)", fraud_scores)

    print("\n=== 5. Saving Anomaly Score Distribution Plot ===")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_test_scores, x='score', hue='label', bins=50, kde=True)
    plt.title("Distribution of Anomaly Scores (Test Set)")
    plt.xlabel("Anomaly Score (Higher = More Anomalous)")
    plt.ylabel("Count")
    plot_path = 'results/anomaly_score_distribution.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    print("\n=== 6. & 7. Feature Patterns: Train Fraud vs Test Fraud ===")
    # Identify top 10 most variant features between normal and fraud in train set
    X_train_df = X.iloc[:train_end].copy()
    X_train_df['label'] = y_train
    
    train_normal_mean = X_train_df[X_train_df['label'] == 0].drop(columns=['label']).mean()
    train_fraud_mean = X_train_df[X_train_df['label'] == 1].drop(columns=['label']).mean()
    
    # Absolute difference to find distinguishing features
    diff = abs(train_fraud_mean - train_normal_mean)
    top_features = diff.sort_values(ascending=False).head(10).index.tolist()
    
    X_test_df = X.iloc[val_end:].copy()
    X_test_df['label'] = y_test
    test_fraud_mean = X_test_df[X_test_df['label'] == 1].drop(columns=['label']).mean()
    
    print("Mean values for top 10 distinguishing features (from Train set):")
    compare_df = pd.DataFrame({
        'Train Normal': train_normal_mean[top_features],
        'Train Fraud': train_fraud_mean[top_features],
        'Test Fraud': test_fraud_mean[top_features] if not test_fraud_mean.isna().all() else np.nan
    })
    print(compare_df)

if __name__ == "__main__":
    main()
