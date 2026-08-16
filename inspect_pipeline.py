import pandas as pd
import joblib
import json

def inspect():
    # 1. Original features
    df = pd.read_csv('data/user_datasets/user_01_transactions_v2.csv', nrows=5)
    orig_cols = df.columns.tolist()
    print("1. Original Features:")
    print(orig_cols)
    
    # Preprocessor
    prep = joblib.load('models/preprocessor_user_01_v2.joblib')
    
    # Extract numerical and categorical columns from the ColumnTransformer
    num_cols = prep.transformers_[0][2]
    cat_cols = prep.transformers_[1][2]
    print("\n2. Numerical Columns used:")
    print(num_cols)
    
    print("\n3. Categorical Columns used:")
    print(cat_cols)
    
    # 4. Excluded columns
    used_cols = set(num_cols + cat_cols)
    excluded = [c for c in orig_cols if c not in used_cols]
    print("\n4. Excluded Columns:")
    print(excluded)
    
    # 5. Number before
    print("\n5. Number of features before preprocessing (excluding label/ids):")
    print(len(used_cols))
    
    # 6 & 7. After OHE / transformed features
    with open('models/feature_names_user_01_v2.json', 'r') as f:
        final_features = json.load(f)
    print("\n6. Number of features after preprocessing:")
    print(len(final_features))
    
    print("\n7. Final transformed feature names:")
    print(final_features)
    
    # 8. Data types
    df_trans = pd.read_csv('data/processed/user_01_v2_preprocessed.csv', nrows=5)
    print("\n8. Final feature data types (ignoring label):")
    df_no_label = df_trans.drop(columns=['label'])
    print(df_no_label.dtypes.value_counts())
    
    # 9 & 10. StandardScaler and OHE checks
    print("\n9. & 10. Preprocessing Steps:")
    for name, transformer, cols in prep.transformers_:
        if name != 'remainder':
            steps = [s[0] for s in transformer.steps]
            print(f"  {name}: {steps}")

if __name__ == '__main__':
    inspect()
