"""
Preprocessing module for Adult Census Income Prediction.
Provides data loading, cleaning, metadata helpers, and pipeline feature transformation.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Ordered lists of features
NUMERICAL_COLS: List[str] = [
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week'
]

CATEGORICAL_COLS: List[str] = [
    'workclass',
    'education',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native.country'
]

EDUCATION_TO_NUM: Dict[str, int] = {
    'Preschool': 1,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad': 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Prof-school': 15,
    'Doctorate': 16
}

CATEGORICAL_OPTIONS: Dict[str, List[str]] = {
    'workclass': [
        'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
        'Self-emp-inc', 'Federal-gov', 'Without-pay'
    ],
    'education': [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
    ],
    'marital.status': [
        'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated',
        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ],
    'occupation': [
        'Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical',
        'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving',
        'Handlers-cleaners', 'Farming-fishing', 'Tech-support',
        'Protective-serv', 'Priv-house-serv', 'Armed-Forces'
    ],
    'relationship': [
        'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'
    ],
    'race': [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ],
    'sex': [
        'Male', 'Female'
    ],
    'native.country': [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
        'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England',
        'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic',
        'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia',
        'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru',
        'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Cambodia',
        'Trinadad&Tobago', 'Laos', 'Thailand', 'Yugoslavia',
        'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras',
        'Scotland', 'Holand-Netherlands'
    ]
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    'age': 'Age of the individual in continuous years (17–90).',
    'workclass': 'Employment sector/type (e.g., Private, Federal-gov, Self-emp).',
    'fnlwgt': 'Final sampling weight; estimated population represented by the census record.',
    'education': 'Highest completed level of formal education.',
    'education.num': 'Numerical encoding of the highest education completed (1–16).',
    'marital.status': 'Marital status of the individual.',
    'occupation': 'Primary job occupation category.',
    'relationship': 'Role/relationship within the household unit.',
    'race': 'Reported racial demographic group.',
    'sex': 'Biological sex (Male or Female).',
    'capital.gain': 'Annual financial capital gains in USD from investments/assets.',
    'capital.loss': 'Annual financial capital losses in USD from investments/assets.',
    'hours.per.week': 'Typical weekly work hours reported (1–99).',
    'native.country': 'Country of origin / birth country.',
    'income': 'Target variable: binary indicator whether annual income exceeds $50K.'
}


def get_feature_metadata() -> Dict[str, Any]:
    """
    Returns metadata dictionaries for UI components and validation.
    """
    return {
        'numerical_cols': NUMERICAL_COLS,
        'categorical_cols': CATEGORICAL_COLS,
        'education_to_num': EDUCATION_TO_NUM,
        'categorical_options': CATEGORICAL_OPTIONS,
        'feature_descriptions': FEATURE_DESCRIPTIONS
    }


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Adult Census DataFrame following the notebook methodology:
    1. Replaces '?' with NA and drops rows with missing values.
    2. Drops duplicate records.
    3. Trims whitespace from string/categorical columns.
    """
    df_clean = df.replace('?', pd.NA).dropna().reset_index(drop=True)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    # Strip whitespace
    str_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()
        
    return df_clean


def load_dataset(csv_path: str = 'Data/adult.csv') -> pd.DataFrame:
    """
    Loads raw CSV and returns the cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    return clean_raw_data(df)


def transform_raw_to_features(
    raw_df: pd.DataFrame,
    expected_feature_names: List[str]
) -> pd.DataFrame:
    """
    Transforms human-readable input dataframe into the exact 96 features expected
    by the trained model (standardizing dummy column alignment and ordering).
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame with columns corresponding to the original features.
    expected_feature_names : List[str]
        The 96 feature names stored in model.feature_names_in_
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_samples, 96) and float64 dtype.
    """
    df_in = raw_df.copy()
    
    # Trim strings in input
    for col in CATEGORICAL_COLS:
        if col in df_in.columns:
            df_in[col] = df_in[col].astype(str).str.strip()

    # One-hot encode using pandas get_dummies
    df_encoded = pd.get_dummies(df_in, columns=CATEGORICAL_COLS, drop_first=False)

    # Initialize zero-filled matrix matching expected features
    df_result = pd.DataFrame(
        0.0,
        index=df_in.index,
        columns=expected_feature_names,
        dtype=float
    )

    # Populate numerical features
    for col in NUMERICAL_COLS:
        if col in df_in.columns:
            df_result[col] = pd.to_numeric(df_in[col], errors='coerce').fillna(0.0).astype(float)

    # Populate dummy features that match expected column names
    for col in expected_feature_names:
        if col in df_encoded.columns:
            df_result[col] = df_encoded[col].astype(float)

    return df_result
