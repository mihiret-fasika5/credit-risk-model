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



    import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# ==========================================
# 1. TASK 4: RFM & PROXY TARGET GENERATOR
# ==========================================

class ProxyTargetEngineer:
    """
    Engineers a proxy credit risk target variable using RFM metrics and K-Means clustering.
    Handles small datasets dynamically to prevent ValueError constraints.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.high_risk_cluster_id_ = None

    def fit_predict_labels(self, df):
        """Calculates RFM profiles, clusters customers, and tags the highest risk group."""
        X_rfm = df.copy()
        X_rfm['TransactionStartTime'] = pd.to_datetime(X_rfm['TransactionStartTime'])
        
        # 1. Establish snapshot date for Recency calculation
        snapshot_date = X_rfm['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # 2. Aggregate RFM values per customer
        rfm_df = X_rfm.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
            'TransactionId': 'count',                                         # Frequency
            'Amount': 'sum'                                                   # Monetary
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })
        
        n_samples = rfm_df.shape[0]
        
        # --- DYNAMIC SAFEGUARD ---
        # Adjust clusters down if sample size is too small for the default cluster count
        effective_clusters = min(self.n_clusters, n_samples)
        
        if effective_clusters < 2:
            # If there's only 1 customer overall, we can't isolate risk via variance.
            # Default to marking them as low risk (0) until more data populates.
            rfm_df['is_high_risk'] = 0
            return rfm_df['is_high_risk'].to_dict()
            
        # Initialize KMeans with the safely adjusted cluster sizes
        kmeans = KMeans(n_clusters=effective_clusters, random_state=self.random_state, n_init=10)
        # -------------------------

        # 3. Pre-process / Scale RFM features
        scaled_rfm = self.scaler.fit_transform(rfm_df)
        
        # 4. Cluster Customers
        rfm_df['Cluster'] = kmeans.fit_predict(scaled_rfm)
        
        # 5. Programmatically identify the highest-risk (least engaged) cluster
        cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(kmeans.cluster_centers_),
            columns=['Recency', 'Frequency', 'Monetary']
        )
        
        # Calculate structural risk score (higher score = lower transaction profiles)
        cluster_centers['Risk_Score'] = cluster_centers['Recency'] / (cluster_centers['Frequency'] * cluster_centers['Monetary'] + 1e-5)
        self.high_risk_cluster_id_ = cluster_centers['Risk_Score'].idxmax()
        
        # Create binary label mapping
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == self.high_risk_cluster_id_).astype(int)
        
        return rfm_df['is_high_risk'].to_dict()

# ==========================================
# 2. TASK 3: REUSABLE CUSTOM TRANSFORMERS
# ==========================================

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_out = X.copy()
        dt_series = pd.to_datetime(X_out[self.datetime_col])
        X_out['TransactionHour'] = dt_series.dt.hour
        X_out['TransactionDay'] = dt_series.dt.day
        X_out['TransactionMonth'] = dt_series.dt.month
        X_out['TransactionYear'] = dt_series.dt.year
        return X_out.drop(columns=[self.datetime_col])

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_rules_ = {}
    def fit(self, X, y=None):
        gp = X.groupby(self.customer_id_col)[self.amount_col]
        self.agg_rules_ = gp.agg(['sum', 'mean', 'count', 'std']).fillna(0).to_dict(orient='index')
        return self
    def transform(self, X):
        X_out = X.copy()
        X_out['Total_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('sum', 0))
        X_out['Avg_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('mean', 0))
        X_out['Transaction_Count'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('count', 0))
        X_out['Std_Transaction_Amount'] = X_out[self.customer_id_col].map(lambda x: self.agg_rules_.get(x, {}).get('std', 0))
        return X_out

# ==========================================
# 3. PIPELINE ORCHESTRATION FUNCTION
# ==========================================

def execute_full_processing_pipeline(raw_df):
    """
    Executes proxy engineering, merges the engineered label, and runs the feature scaling pipeline.
    """
    df_working = raw_df.copy()
    
    # Run Task 4 Execution First to create the Target Label
    print("Engineering Proxy Target Variable (is_high_risk) via Unsupervised RFM Clustering...")
    target_engineer = ProxyTargetEngineer(n_clusters=3, random_state=42)
    risk_mapping = target_engineer.fit_predict_labels(df_working)
    
    # Map the proxy variable directly back into the dataset using CustomerId
    df_working['is_high_risk'] = df_working['CustomerId'].map(risk_mapping)
    
    # Isolate targets and processing arrays
    X = df_working.drop(columns=['FraudResult', 'is_high_risk'])
    y_fraud = df_working['FraudResult']
    y_risk = df_working['is_high_risk']
    
    # Split features by logical type
    high_card_features = ['AccountId', 'ProviderId', 'ProductId', 'BatchId', 'SubscriptionId', 'TransactionId', 'CustomerId']
    low_card_features = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'CurrencyCode', 'CountryCode']
    numerical_features = ['Amount', 'Value']
    
    # Build scikit-learn sub-pipeline steps
    initial_transformations = Pipeline([
        ('datetime_extractor', DateTimeFeatureExtractor(datetime_col='TransactionStartTime')),
        ('customer_aggregator', CustomerAggregator(customer_id_col='CustomerId', amount_col='Amount'))
    ])
    
    # Track features that survive feature extraction
    updated_numerical_cols = numerical_features + [
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount'
    ]
    
    col_transformer = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), updated_numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), low_card_features)
    ], remainder='drop')
    
    # Fit & Transform feature space
    X_interim = initial_transformations.fit_transform(X)
    X_final_array = col_transformer.fit_transform(X_interim)
    
    # Reassemble back into an analytical-ready Pandas DataFrame
    encoded_cat_features = col_transformer.named_transformers_['cat']\
                                          .named_steps['ohe']\
                                          .get_feature_names_out(low_card_features).tolist()
    
    all_feature_names = updated_numerical_cols + encoded_cat_features
    processed_df = pd.DataFrame(X_final_array, columns=all_feature_names)
    
    # Append the engineered proxy label back onto the final output
    processed_df['is_high_risk'] = y_risk.values
    processed_df['FraudResult'] = y_fraud.values
    
    return processed_df

# ==========================================
# 4. EXECUTION SIMULATION
# ==========================================
if __name__ == "__main__":
    # Standard dummy data matching your real operational columns
    mock_raw_data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'BatchId': ['B1', 'B1', 'B2', 'B2', 'B3'],
        'AccountId': ['A1', 'A2', 'A1', 'A3', 'A4'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3'],
        'CustomerId': ['Cust_Active', 'Cust_Active', 'Cust_Risk', 'Cust_Risk', 'Cust_Active'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': ['256', '256', '256', '256', '256'],
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1'],
        'ProductId': ['Prod1', 'Prod2', 'Prod1', 'Prod3', 'Prod1'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility', 'airtime'],
        'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch1', 'Ch1'],
        'Amount': [50000.0, 65000.0, 50.0, 100.0, 45000.0], # Distinct volume differences
        'Value': [50000, 65000, 50, 100, 45000],
        'TransactionStartTime': ['2026-01-01 10:00:00', '2026-01-15 11:30:00', '2026-05-01 12:00:00', '2026-05-02 09:00:00', '2026-06-01 15:45:00'],
        'PricingStrategy': ['2', '4', '2', '2', '2'],
        'FraudResult': [0, 0, 0, 0, 0]
    }
    
    raw_df = pd.DataFrame(mock_raw_data)
    
    # Process
    final_dataset = execute_full_processing_pipeline(raw_df)
    
    print("\n--- Pipeline Complete Execution Results ---")
    print(f"Final Data Shape: {final_dataset.shape}")
    print("\nValue distributions for your engineered target column:")
    print(final_dataset['is_high_risk'].value_counts())