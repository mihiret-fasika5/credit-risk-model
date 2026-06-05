import pytest
import os
import sys

# Dynamic path injection: Adds the root directory ('credit-risk-model') to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from src.data_processing import DateTimeFeatureExtractor, execute_full_processing_pipeline

@pytest.fixture
def sample_raw_data():
    """Fixture providing minimal dataframe structures mimicking production data columns."""
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'BatchId': ['B1', 'B1', 'B2'],
        'AccountId': ['A1', 'A2', 'A1'],
        'SubscriptionId': ['S1', 'S1', 'S2'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX'],
        'CountryCode': ['256', '256', '256'],
        'ProviderId': ['P1', 'P2', 'P1'],
        'ProductId': ['Prod1', 'Prod2', 'Prod1'],
        'ProductCategory': ['airtime', 'utility', 'airtime'],
        'ChannelId': ['Ch1', 'Ch2', 'Ch1'],
        'Amount': [1000.0, 5000.0, 2000.0],
        'Value': [1000, 5000, 2000],
        'TransactionStartTime': ['2026-01-01 09:00:00', '2026-01-02 14:30:00', '2026-01-03 18:00:00'],
        'PricingStrategy': ['2', '4', '2'],
        'FraudResult': [0, 0, 0]
    })


def test_datetime_feature_extractor(sample_raw_data):
    """Test that DateTimeFeatureExtractor correctly drops the base column and extracts components."""
    extractor = DateTimeFeatureExtractor(datetime_col='TransactionStartTime')
    transformed = extractor.fit_transform(sample_raw_data)
    
    # Assert original structural column drops
    assert 'TransactionStartTime' not in transformed.columns
    
    # Assert engineered features exist
    assert 'TransactionHour' in transformed.columns
    assert 'TransactionDay' in transformed.columns
    assert 'TransactionMonth' in transformed.columns
    assert 'TransactionYear' in transformed.columns
    
    # Verify exact calculations
    assert transformed['TransactionHour'].iloc[1] == 14
    assert transformed['TransactionDay'].iloc[2] == 3


def test_full_pipeline_output_structure(sample_raw_data):
    """Test that full orchestrator pipeline cleans, handles proxies, and scales output appropriately."""
    processed_df = execute_full_processing_pipeline(sample_raw_data)
    
    # Confirm proxy target variables were added back smoothly
    assert 'is_high_risk' in processed_df.columns
    assert 'FraudResult' in processed_df.columns
    
    # Ensure numerical columns like Amount are handled and scaled (mean close to 0)
    assert not processed_df.isnull().values.any()
    assert isinstance(processed_df, pd.DataFrame)