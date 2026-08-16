import os
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_and_save_pipeline(data_path, preprocessor_path, feature_names_path, processed_data_path):
    # 1. Load dataset
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if 'label' not in df.columns:
        raise ValueError("Label column 'label' not found in dataset.")
    
    y = df['label']
    X = df.drop(columns=['label'])

    # 2 & 3. Exclude identifier columns and raw timestamp
    cols_to_exclude = ['transaction_id', 'user_id', 'timestamp']
    X = X.drop(columns=[col for col in cols_to_exclude if col in X.columns])

    # Split to fit only on training data (Requirement 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape before preprocessing: {X_train.shape}")

    # Identify numeric and categorical columns
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Numerical transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 5. Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit on training data
    preprocessor.fit(X_train)

    # 4. Transform training data
    X_train_transformed = preprocessor.transform(X_train)

    # Get feature names
    feature_names_list = []
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names_list = feature_names.tolist()
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names_list, f, indent=4)
    except Exception as e:
        print(f"Could not get/save feature names: {e}")

    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)

    # 5. Save transformed training data with label
    # Convert transformed numpy array to DataFrame
    if feature_names_list:
        df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names_list, index=X_train.index)
    else:
        df_transformed = pd.DataFrame(X_train_transformed, index=X_train.index)
    
    # Add label column at the end
    df_transformed['label'] = y_train
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save to CSV
    df_transformed.to_csv(processed_data_path, index=False)

    # 10. Print statistics
    print(f"Training data shape after preprocessing: {df_transformed.shape}") # including label column
    print(f"Number of final features (excluding label): {X_train_transformed.shape[1]}")
    print(f"Preprocessed CSV saved to: {processed_data_path}")

if __name__ == "__main__":
    datasets = ["user_01", "user_02"]
    for user in datasets:
        DATA_PATH = f"data/user_datasets/{user}_transactions.csv"
        PREPROCESSOR_PATH = f"preprocessor_{user}.joblib"
        FEATURE_NAMES_PATH = f"feature_names_{user}.json"
        PROCESSED_DATA_PATH = f"data/processed/{user}_preprocessed.csv"
        
        print(f"\n--- Processing {user} ---")
        build_and_save_pipeline(DATA_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, PROCESSED_DATA_PATH)
