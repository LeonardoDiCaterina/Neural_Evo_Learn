import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from utils.grid_search_v2_leo import montecarlo_cv, grid_search
import os
os

@pytest.fixture # this decorator is used to create a fixture (a reusable piece of code, in this case, a dataset)
# if you notice, whenever I call regression_data in the test functions, it is passed as an argument like a variable
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


def test_montecarlo_cv_basic(regression_data):
    X, y = regression_data
    params = {}
    result = montecarlo_cv("linear regression", X, y, params, n_splits=3, seed=42)

    assert isinstance(result, dict)
    assert "mean_score" in result
    assert "fold_scores" in result
    assert isinstance(result["fold_scores"], list)
    assert len(result["fold_scores"]) == 3


def test_montecarlo_cv_invalid_model(regression_data):
    X, y = regression_data
    # pytest.raises is used to check if the code raises an exception withot stopping the test (as I expect it to brake)
    # if it doesn't, the test will fail
    with pytest.raises(ValueError):
        montecarlo_cv("invalid model", X, y, {})


def test_montecarlo_cv_custom_scorer(regression_data):
    X, y = regression_data
    def neg_rmse(model, X_val, y_val):
        return -np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    result = montecarlo_cv("linear regression", X, y, {}, n_splits=3, seed=42, scorer=neg_rmse)
    assert result["mean_score"] < 0  # Because it's negative RMSE


def test_grid_search_basic(regression_data, tmp_path):
    X, y = regression_data
    param_grid = {
        "fit_intercept": [True, False]
    }

    log_path = tmp_path / "grid_log.csv"
    best_params, best_score, history_df = grid_search(
        X, y,
        model_name="linear regression",
        param_grid=param_grid,
        cv=3,
        seed=42,
        log_path=log_path,
        minimize=True
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert isinstance(history_df, pd.DataFrame)
    assert log_path.exists()
    assert len(history_df) == 2  # 2 combinations tested


def test_grid_search_empty_param_grid(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError):
        grid_search(X, y, model_name="linear regression", param_grid={})