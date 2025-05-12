import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def feature_engineering(data, manual_row_removal=False):
    y = data['CRUDE PROTEIN'] 
    X = data.drop(columns=['CRUDE PROTEIN', 'WING TAG', 'EMPTY MUSCULAR STOMACH'])
    
    if isinstance(manual_row_removal, list):
        for row in manual_row_removal:
            X = X.drop(row)
            y = y.drop(row)

    return X, y


def preprocess_data(X, y):
    
    # RE THY ALREADY TENSORS
    if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
        return X, y
    
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    else:
        try:
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
        except Exception as e:
            raise TypeError(f"Unsupported data type: {type(X)}. Expected pd.DataFrame, np.ndarray, or torch.Tensor.")
    
    return X, y


def random_split(X, y, val_size=0.2):
    """
    Splits the data into training and testing sets, preprocesses the data, and returns the final training and testing sets.

    Args:
        X (pd.DataFrame): training data
        y (pd.Series): target variable
        val_size (float, optional): proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple: preprocessed training and testing data
    
    """
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
    
    X_final_train, y_final_train = preprocess_data(X_train, y_train)
    X_final_val, y_final_val = preprocess_data(X_val, y_val)
    
    return X_final_train, X_final_val, y_final_train, y_final_val