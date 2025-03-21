from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler



def index_categories(df: pd.DataFrame, col, categories: List[str] | None = None, drop=False) -> pd.DataFrame:
    # Create a mapping for col if not provided
    if categories is None:
        categories = list(np.unique(df[col]))
    elif drop:
        # Filter out records without appropriate targets
        df = df[df[col].isin(categories)]

    # Map values to index
    mapping = {label: idx for idx, label in enumerate(categories)}
    df[col] = df[col].map(mapping)
    return df


def onehot_encoding(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    if not cols:
        return df

    # One-Hot Encoding for nominal categorical variables
    df = pd.get_dummies(df, columns=cols)

    return df

def frequency_encoding(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    if not cols:
        return df

    for col in cols:
        # Calculate frequency of each category
        freq = df[col].value_counts() / len(df)

        # Map frequency to each record
        df[col] = df[col].map(freq)

    return df

def drop_corr_pairs(df, corr_matrix):
    # Find features with correlation > 0.8
    high_corr_var = np.where(corr_matrix > 0.8)
    high_corr_var = [
        (corr_matrix.columns[x], corr_matrix.columns[y])
        for x, y in zip(*high_corr_var)
        if x != y and x < y
    ]

    # Drop one feature from each pair with correlation > 0.8
    for feature1, feature2 in high_corr_var:
        if feature1 in df.columns:
            df = df.drop([feature1], axis=1)

    return df

def new_features(df: pd.DataFrame) -> pd.DataFrame:
    # First digits of zip code
    df['zip_code'] = df['zip_code'].str.extract(r'(\d+)').astype(float)

    return df
