from datetime import datetime
import pandas as pd
from typing import Tuple

def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    train_X = train_data.drop(columns=[target_column_name])
    train_y = train_data[target_column_name]
    
    test_X = test_data.drop(columns=[target_column_name])
    test_y = test_data[target_column_name]

    return train_X, train_y, test_X, test_y
