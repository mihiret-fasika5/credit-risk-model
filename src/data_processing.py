import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. CUSTOM TRANSFORMERS FOR COMPLEX STEPS
# ==========================================

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts datetime components from TransactionStartTime."""
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        # Ensure datetime type
        dt_series = pd.to_datetime(X_out[self.datetime_col])
        
        X_out['TransactionHour'] = dt_series.dt.hour
        X_out['TransactionDay'] = dt_series.dt.day
        X_out['TransactionMonth'] = dt_series.dt.month
        X_out['TransactionYear'] = dt_series.dt.year
        
        # Drop original datetime column to avoid issues in downstream scaling/encoding
        X_out = X_out.drop(columns=[self.datetime_col])
        return X_out


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Calculates historical aggregation metrics per CustomerId."""
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_rules_ = {}

    def fit(self, X, y=None):
        # Learn mapping from the training set to prevent data leakage during inference
        gp = X.groupby(self.customer_id_col)[self.amount_col]
        
        stats = gp.agg(['sum', 'mean', 'count', 'std']).fillna(0)
        
        # Store aggregations in a dictionary for fast mapping during transform
        self.agg_rules_ = stats.to_dict(orient='index')
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Map training stats, fallback to 0 if a new customer appears in test set
        X_out['Total_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('sum', 0))
        X_out['Avg_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('mean', 0))
        X_out['Transaction_Count'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('count', 0))
        X_out['Std_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('std', 0))
        
        return X_out


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) mapping to high-cardinality categorical columns.
    Uses basic target encoding math as a robust fallback to avoid external library version conflicts.
    """
    def __init__(self, columns, target_col='FraudResult', epsilon=1e-5):
        self.columns = columns
        self.target_col = target_col
        self.epsilon = epsilon
        self.woe_maps_ = {}

    def fit(self, X, y=None):
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        elif y is None:
            raise ValueError("Target variable y or target_col must be provided for WoE calculation.")
            
        df = X.copy()
        df['target'] = y
        
        global_pos = df['target'].sum()
        global_neg = len(df) - global_pos
        
        for col in self.columns:
            stats = df.groupby(col)['target'].agg(['sum', 'count'])
            stats['pos'] = stats['sum']
            stats['neg'] = stats['count'] - stats['sum']
            
            # Calculate WoE with epsilon smoothing to prevent division by zero
            pos_dist = (stats['pos'] + self.epsilon) / (global_pos + self.epsilon)
            neg_dist = (stats['neg'] + self.epsilon) / (global_neg + self.epsilon)
            
            stats['woe'] = np.log(pos_dist / neg_dist)
            self.woe_maps_[col] = stats['woe'].to_dict()
            
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns:
                # Map WoE values, default to 0 (neutral) for unknown categories
                X_out[col + '_WoE'] = X_out[col].map(self.woe_maps_.get(col, {})).fillna(0)
                X_out = X_out.drop(columns=[col])
        return X_out


# ==========================================
# 2. PIPELINE BUILDING FUNCTION
# ==========================================

def build_feature_engineering_pipeline(high_cardinality_cols, low_cardinality_cols, numerical_cols):
    """
    Constructs a complete scikit-learn Pipeline for data transformation.
    """
    
    # Pre-processing pipeline for initial transformations and extractions
    initial_transformations = Pipeline([
        ('datetime_extractor', DateTimeFeatureExtractor(datetime_col='TransactionStartTime')),
        ('customer_aggregator', CustomerAggregator(customer_id_col='CustomerId', amount_col='Amount')),
        ('woe_encoder', WoETransformer(columns=high_cardinality_cols))
    ])
    
    # Define updated feature pools post initial transformations
    # New aggregate features + engineered time features need scaling
    updated_numerical_cols = numerical_cols + [
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount'
    ]
    
    # Final column transformer for scaling and OH encoding
    col_transformer = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), updated_numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), low_cardinality_cols)
    ], remainder='drop')
    
    # Complete master pipeline
    full_pipeline = Pipeline([
        ('initial_features', initial_transformations),
        ('scaling_and_encoding', col_transformer)
    ])
    
    return full_pipeline

# ==========================================
# 3. EXECUTION BLOCK (EXAMPLE USAGE)
# ==========================================
if __name__ == "__main__":
    # Create mock dataframe using your exact columns
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'BatchId': ['B1', 'B1', 'B2', 'B2'],
        'AccountId': ['A1', 'A2', 'A1', 'A3'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': ['256', '256', '256', '256'],
        'ProviderId': ['P1', 'P2', 'P1', 'P3'],
        'ProductId': ['Prod1', 'Prod2', 'Prod1', 'Prod3'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility'],
        'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch1'],
        'Amount': [1000.0, 5000.0, 2000.0, 15000.0],
        'Value': [1000, 5000, 2000, 15000],
        'TransactionStartTime': ['2025-11-01 09:15:00', '2025-11-01 10:20:00', '2025-11-02 14:05:00', '2025-11-03 18:45:00'],
        'PricingStrategy': ['2', '4', '2', '2'],
        'FraudResult': [0, 1, 0, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Separate Features and Target
    X = df.drop(columns=['FraudResult'])
    y = df['FraudResult']
    
    # Segregate features based on your dataset properties
    # High cardinality IDs are perfect for WoE; categories are perfect for OHE
    high_card_features = ['AccountId', 'ProviderId', 'ProductId', 'BatchId', 'SubscriptionId']
    low_card_features = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'CurrencyCode', 'CountryCode']
    numerical_features = ['Amount', 'Value']
    
    print("Initializing and fitting Feature Engineering Pipeline...")
    pipeline = build_feature_engineering_pipeline(high_card_features, low_card_features, numerical_features)
    
    # Fit and transform data
    X_processed = pipeline.fit_transform(X, y)
    
    # Retrieve feature names from ColumnTransformer for clear visualization
    encoded_cat_features = pipeline.named_steps['scaling_and_encoding']\
                                   .named_transformers_['cat']\
                                   .named_steps['ohe']\
                                   .get_feature_names_out(low_card_features).tolist()
                                   
    all_feature_names = (numerical_features + 
                         ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                          'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount'] + 
                         encoded_cat_features)
                         
    df_ready = pd.DataFrame(X_processed, columns=all_feature_names)
    
    print("\n--- Model-Ready DataFrame Shape ---")
    print(df_ready.shape)
    print("\n--- Sample Processed Data Features ---")
    print(df_ready.iloc[:, :7].head())