# Standard library imports
from itertools import product, chain
import itertools
import json
import os
import statistics

# Third-party imports

from sklearn.model_selection import KFold


def fit_model_GridSearch(gp_model, fixed_params, param_grid, seed):
    models = []
    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        dynamic_params = dict(zip(keys, combo))
        full_params = {**fixed_params, **dynamic_params}
        model = gp_model(**full_params, seed=seed)
        res = {"model": model}
        res.update({"rmse_train": model.fitness.item()})
        res.update({"rmse_test": model.test_fitness.item()})
        res.update({"dynamic_params": dynamic_params})
        models.append(res)
    return models


def group_and_median_rmse(results_data):
    """
    Groups results by 'dynamic_params' and calculates the median 'rmse_test' for each group.

    Args:
        results_data (list): A list of lists, where each inner list contains dictionaries
                             with 'dynamic_params' and 'rmse_test' keys.

    Returns:
        list: A list of dictionaries, each containing:
              {'dynamic_params': {...}, 'rmse_test_median': float}
    """

    # Flatten the list of lists into a single list of dictionaries
    flattened_results = list(itertools.chain.from_iterable(results_data))

    grouped_scores_data = {}

    for item in flattened_results:
        dynamic_params_dict = item["dynamic_params"]
        rmse_test = item["rmse_test"]

        # Sort params to ensure consistency
        # Convert to tuple to make it hashable, and so able to be used as a dictionary key
        hashable_dynamic_params = tuple(sorted(dynamic_params_dict.items()))

        # Check if combination does not exist in the dictionary
        if hashable_dynamic_params not in grouped_scores_data:

            # Create entry if not
            grouped_scores_data[hashable_dynamic_params] = {
                "dynamic_params": dynamic_params_dict,
                "rmse_test": [],
            }
        grouped_scores_data[hashable_dynamic_params]["rmse_test"].append(rmse_test)

    # Calculate median for each group and format output
    final_output = []
    for group_info in grouped_scores_data.values():
        combination = group_info["dynamic_params"]
        rmse_scores = group_info["rmse_test"]

        # Calculate median RMSE
        median_rmse = statistics.median(rmse_scores)
        final_output.append(
            {"dynamic_params": combination, "rmse_test_median": median_rmse}
        )

    return final_output


def gp_nested_cross_validation(
    X,
    y,
    gp_model: callable,
    k_outer: int,
    k_inner: int,
    fixed_params: dict,
    param_grid: dict,
    seed: int,
    LOG_DIR: str,
    DATASET_NAME: str,
):
    """
    Perform nested cross-validation for a given model and dataset.

    Args:
        X (torch.Tensor): Feature matrix.
        y (torch.Tensor): Target vector.
        gp_model (callable): The gp model to be evaluated.
        k_outer (int): Number of outer folds.
        k_inner (int): Number of inner folds.
        fixed_params (dict): Fixed parameters for the model.
        param_grid (dict): Parameter grid for hyperparameter tuning.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of dictionaries containing model results.
    """

    cv_outer = KFold(n_splits=k_outer, random_state=seed, shuffle=True)
    cv_inner = KFold(n_splits=k_inner, random_state=seed, shuffle=True)

    data_cv_outer = [
        [learning_ix, test_ix] for learning_ix, test_ix in cv_outer.split(X, y)
    ]

    models = []

    for i, (train_ix, test_ix) in enumerate(data_cv_outer):
        print(f"Outer fold {i+1}/{k_outer}")
        X_learning, y_learning = X[train_ix], y[train_ix]
        X_test, y_test = X[test_ix], y[test_ix]

        # Inner cross-validation
        results = []

        data_cv_inner = [
            [learning_ix, val_ix]
            for learning_ix, val_ix in cv_inner.split(X_learning, y_learning)
        ]
        for j, (train_ix, val_ix) in enumerate(data_cv_inner):

            # Split the data into training and validation sets K times
            print(f"-----\n Inner fold {j+1}/{k_inner}")
            X_inner_train, y_inner_train = X_learning[train_ix], y_learning[train_ix]
            X_inner_val, y_inner_val = X_learning[val_ix], y_learning[val_ix]

            print(
                f"Training shape: {X_inner_train.shape}\nValidation shape: {X_inner_val.shape}\n"
            )

            # Update the X and y values in the fixed_params dictionary
            fixed_params.update(
                {
                    "X_train": X_inner_train,
                    "y_train": y_inner_train,
                    "X_test": X_inner_val,
                    "y_test": y_inner_val,
                }
            )

            # Update LOG_PATH in the fixed_params dictionary
            LOG_PATH = (
                LOG_DIR
                + DATASET_NAME
                + "_"
                + "outer"
                + "_"
                + str(i)
                + "_"
                + "inner"
                + "_"
                + str(j)
                + ".csv"
            )
            if os.path.exists(LOG_PATH):
                os.remove(LOG_PATH)
            fixed_params.update({"log_path": LOG_PATH})

            res = fit_model_GridSearch(
                gp_model=gp_model,
                fixed_params=fixed_params,
                param_grid=param_grid,
                seed=(seed + k_inner),
            )

            # Log
            results.append(res)

        medians = group_and_median_rmse(results)

        # Find minimum median rmse
        best_dynamic_combo_median = min(medians, key=lambda x: x["rmse_test_median"])

        print(
            f'Best inner combination: {best_dynamic_combo_median["dynamic_params"]} with median RMSE: {best_dynamic_combo_median["rmse_test_median"]}'
        )

        # Train the best model on the entire training set
        print("Training best combination on entire learning set")

        best_hyper_combo = best_dynamic_combo_median["dynamic_params"]

        fixed_params.update(
            {
                "X_train": X_learning,
                "y_train": y_learning,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

        LOG_PATH = LOG_DIR + DATASET_NAME + "_" + "outer" + "_" + str(i) + ".csv"
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        fixed_params.update({"log_path": LOG_PATH})

        full_params = {**fixed_params, **best_hyper_combo}

        outer_model = gp_model(**full_params, seed=(seed + k_outer))

        # Add the best hyperparameters to the log .csv
        df = pd.read_csv(LOG_PATH)
        df["params"] = best_hyper_combo
        df.to_csv(LOG_PATH, index=False, header=None)

        res = {"model": outer_model}
        res.update({"rmse_train": outer_model.fitness.item()})
        res.update({"rmse_test": outer_model.test_fitness.item()})
        res.update({"dynamic_params": best_hyper_combo})

        models.append(res)

    return models
