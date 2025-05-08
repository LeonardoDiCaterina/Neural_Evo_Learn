import pandas as pd
import itertools
import json
import datetime
import pathlib


def grid_search(
    X,
    y,
    param_grid: dict | None = None,
    cv: int = 5,
    seed: int = 1,
    log_path: str | pathlib.Path = "gp_grid_log.csv",
    minimize: bool = True,
):
    """
    Grid search over param_grid for slim-gsgp models.

    Returns
    -------
    best_params : dict
    best_score  : float
    history_df  : pandas.DataFrame   (one row per configuration)
    """
    grid = list(itertools.product(*param_grid.values()))

    records, best_cfg = [], None

    if minimize:
        best_score = float("inf")
    else:
        best_score = float("-inf")

    for combo in grid:
        hparams = dict(zip(param_grid.keys(), combo))
        cv_res = CROSS_VALIDATION_FUNCTION_HERE(
            X, y, hparams, n_folds=cv, seed=seed
        )  # It would be nice if this function could recieve the model as a parameter like: CROSS_VALIDATION_FUNCTION_HERE(gp, X, y, hparams, n_folds=cv, seed=seed)

        records.append(
            {
                **hparams,
                "mean_score": cv_res["mean_score"],
                "fold_scores": json.dumps(cv_res["fold_scores"]),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        if cv_res["mean_score"] < best_score and minimize:
            best_cfg, best_score = hparams, cv_res["mean_score"]

        elif cv_res["mean_score"] > best_score and not minimize:
            best_cfg, best_score = hparams, cv_res["mean_score"]

    hist_df = pd.DataFrame(records)
    hist_df.to_csv(log_path, index=False)

    return best_cfg, best_score, hist_df
