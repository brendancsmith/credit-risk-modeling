from pathlib import Path
from typing import List
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder

import pandas as pd

def load_data() -> pd.DataFrame:
    # Load the dataset
    dataset = Path(__file__).parent / '../data/raw/accepted_2007_to_2018Q4.csv.gz'
    df = pd.read_csv(dataset, compression='gzip', low_memory=False)

    return df

def drop_sparse_cols(df: pd.DataFrame, missing_rate: float = 0.5) -> pd.DataFrame:
    # Remove columns with more than 50% missing values
    threshold = int(len(df) * missing_rate)
    df = df.dropna(thresh=threshold, axis=1)

    return df

def drop_cols(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    if not cols:
        return df

    # Drop a list of columns from the dataframe
    df = df.drop(columns=cols)

    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill numerical columns with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df

def convert_dates(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    if not cols:
        return df

    # Convert date columns to datetime
    for col in cols:
        dates = pd.to_datetime(df[col], format='%b-%Y')
        unix_times = dates.astype(int) / 10**9
        df[col] = unix_times

    return df

def extract_digits(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    if not cols:
        return df

    for col in cols:
        # Extract numerical values from strings
        df[col] = df[col].str.extract(r'(\d+)').astype(float)

    return df

def scale_features(df: pd.DataFrame, target=None) -> pd.DataFrame:
    # Identify numerical variables
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target and target in numerical_cols:
        numerical_cols.remove("loan_status")  # Exclude target variable

    # Standardization
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
