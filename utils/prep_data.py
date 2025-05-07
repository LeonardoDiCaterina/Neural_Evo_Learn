import torch

def feature_engineering(data, manual_row_removal=False):
    y = data['CRUDE PROTEIN'] 
    X = data.drop(columns=['CRUDE PROTEIN', 'WING TAG', 'EMPTY MUSCULAR STOMACH'])
    
    if isinstance(manual_row_removal, list):
        for row in manual_row_removal:
            X = X.drop(row)
            y = y.drop(row)

    return X, y


def preprocess_data(X, y):
    # convert it to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    return X, y