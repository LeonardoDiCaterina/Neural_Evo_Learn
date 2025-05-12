import pandas as pd
import numpy as np
import itertools
import json
import datetime
from pathlib import Path
from typing import Union, Optional


from typing import Union, Callable, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


from utils.prep_data import preprocess_data

MODEL_REGISTRY = {
    "linear regression": LinearRegression,
    "logistic regression": LogisticRegression,
    "random forest": RandomForestRegressor,
    "svm": SVC
}


def montecarlo_cv(
    model_class: Union[str, Callable],
    X,
    y,
    params: dict,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 1,
    scorer: Optional[Callable] = None
) -> dict:
    """
    Perform Monte Carlo Cross-Validation.

    Returns a dictionary with the mean score and individual fold scores.
    """
    if isinstance(model_class, str):
        if model_class not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_class}'. Choose from: {list(MODEL_REGISTRY.keys())}")
        model_class = MODEL_REGISTRY[model_class]

    if not callable(model_class):
        raise TypeError("model_class must be a callable or a valid model name string.")

    scores = []
    for i in range(n_splits):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed + i
        )
        X_train_prep, y_train_prep = preprocess_data(X_train, y_train)
        X_val_prep, y_val_prep = preprocess_data(X_val, y_val)
        model = model_class(**params)
        model.fit(X_train, y_train)

        score = scorer(model, X_val, y_val) if scorer else model.score(X_val, y_val)
        scores.append(score)

    return {
        "mean_score": float(np.mean(scores)),
        "fold_scores": list(scores)
    }


def grid_search(
    X,
    y,
    model_name: str = "linear regression",
    param_grid: Optional[dict] = None,
    cv: int = 5,
    seed: int = 1,
    log_path: Union[str, Path] = "gp_grid_log.csv",
    minimize: bool = True,
    scorer: Optional[Callable] = None
) -> tuple[dict, float, pd.DataFrame]:
    """
    Grid search with Monte Carlo CV over hyperparameter combinations.

    Returns best_params, best_score, and a DataFrame with the search history.
    """
    if param_grid is None or not param_grid:
        raise ValueError("param_grid must be a non-empty dictionary.")

    param_combinations = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    best_score = float("inf") if minimize else float("-inf")
    best_params = None
    history = []
    import tqdm
    
    for combo in tqdm.tqdm(param_combinations, desc="Grid Search Progress", unit="combination"):
        params = dict(zip(param_keys, combo))
        result = montecarlo_cv(model_name, X, y, params, n_splits=cv, test_size=0.2, seed=seed, scorer=scorer)

        record = {
            "model": model_name,
            **params,
            "mean_score": result["mean_score"],
            "fold_scores": json.dumps(result["fold_scores"]),
            "timestamp": datetime.datetime.now().isoformat()
        }
        history.append(record)

        is_better = (result["mean_score"] < best_score) if minimize else (result["mean_score"] > best_score)
        if is_better:
            best_score = result["mean_score"]
            best_params = params

    history_df = pd.DataFrame(history)
    history_df.sort_values("mean_score", ascending=minimize, inplace=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(log_path, index=False)

    return best_params, best_score, history_df