import json
from pathlib import Path
from itertools import product
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

from joblib import delayed, Parallel, dump
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from scipy.stats import loguniform, randint

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import load_seer
from hazardous._gb_multi_incidence import GBMultiIncidence

# from hazardous.survtrace._encoder import SurvFeatureEncoder
# from hazardous._deep_hit import _DeepHit
from hazardous._fine_and_gray import FineGrayEstimator
from hazardous._aalen_johasen import AalenJohansenEstimator
from hazardous.survtrace._model import SurvTRACE

# from hazardous.utils import CumulativeIncidencePipeline

SEED = 0
# Enable oracle scoring for GridSearchCV
# GBMI.set_score_request(scale=True, shape=True)

gbmi_competing_loss = GBMultiIncidence(loss="competing_risks", show_progressbar=True)
gbmi_log_loss = GBMultiIncidence(loss="inll", show_progressbar=True)

# deephit = _DeepHit(
#   num_nodes_shared=[64, 64],
#    num_nodes_indiv=[32],
#    verbose=True,
#    num_durations=10,
#    batch_norm=True,
#    dropout=None,
# )

fine_and_gray = FineGrayEstimator()
aalen_johansen = AalenJohansenEstimator(calculate_variance=False, seed=SEED)
survtrace = SurvTRACE(device="cpu", max_epochs=30)

gbmi_param_grid = {
    "learning_rate": loguniform(0.01, 0.5),
    "max_depth": randint(2, 10),
    "n_iter": randint(20, 200),
    "n_times": randint(1, 5),
    "n_iter_before_feedback": randint(20, 50),
}

survtrace_grid = {
    "lr": loguniform(1e-5, 1e-3),
    "batch_size": [256, 512, 1024],
    "hidden_size": [8, 16, 32, 64],
    "num_hidden_layers": [2, 3, 4],
    "intermediate_size": [64],
}

ESTIMATOR_GRID = {
    # "gbmi_competing_loss": {
    #     "estimator": gbmi_competing_loss,
    #     "param_grid": gbmi_param_grid,
    # },
    # "gbmi_log_loss": {
    #     "estimator": gbmi_log_loss,
    #     "param_grid": gbmi_param_grid,
    # },
    # "survtrace":{
    #     "estimator": survtrace,
    #     "param_grid": survtrace_grid
    #     }
    # }
    "fine_and_gray": {
        "estimator": fine_and_gray,
        "param_grid": {},
    },
    "aalen_johansen": {
        "estimator": aalen_johansen,
        "param_grid": {},
    },
}


# Parameters of the make_synthetic_competing_weibull function.
DATASET_GRID = {
    "n_events": [3, 5],
    "n_samples": [1_000, 10_000, 20_000, 50_000, 200_000],
    "censoring_relative_scale": [0.65, 0.8, 1.5, 2.5],
    "complex_features": [True, False],
    "independent_censoring": [False, True],
    "n_features": [10, 20, 40],
}

PATH_DAILY_SESSION = Path(datetime.now().strftime("%Y-%m-%d"))

SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
CHURN_PATH = "../hazardous/data/churn.csv"
SEED = 0
N_JOBS_CV = 15
N_ITER_CV = 10


def run_all_synthetic_datasets(estimator_name):
    grid_dataset_params = list(product(*DATASET_GRID.values()))

    parallel = Parallel(n_jobs=None)
    parallel(
        delayed(run_synthetic_dataset)(estimator_name, dataset_params)
        for dataset_params in grid_dataset_params
    )


def run_synthetic_dataset(estimator_name, dataset_params):
    dataset_params = dict(zip(DATASET_GRID.keys(), dataset_params))
    data_bunch = make_synthetic_competing_weibull(
        **dataset_params, feature_rounding=None, target_rounding=None
    )

    print(f"{estimator_name}\n{dataset_params}")

    if estimator_name == "fine_and_gray" and dataset_params["n_samples"] > 10_000:
        print("SKIP")
        return

    run_estimator(
        estimator_name,
        data_bunch,
        dataset_name="weibull",
        dataset_params=dataset_params,
    )


def run_real_dataset(estimator_name, dataset_name="seer"):
    if dataset_name == "seer":
        data_bunch = load_seer(
            SEER_PATH,
            survtrace_preprocessing=True,
            return_X_y=False,
        )
        X, y = data_bunch.X, data_bunch.y

    elif dataset_name == "churn":
        churn_data = pd.read_csv(CHURN_PATH)
        X, y = churn_data[["months_active"]], churn_data["churned"]
        data_bunch = Bunch(X=X, y=y)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    X_train_, _, y_train_, _ = train_test_split(X, y, test_size=0.3, random_state=SEED)
    SAMPLES_RATIO = [0.1, 0.3, 0.5, 0.7, 1.0]
    for samples_ratio in SAMPLES_RATIO:
        idx_train = np.random.choice(
            len(X_train_), int(len(X_train_) * samples_ratio), replace=False
        )
        X_train, y_train = X_train_.iloc[idx_train], y_train_.iloc[idx_train]
        data_bunch.X, data_bunch.y = X_train, y_train

        run_estimator(
            estimator_name,
            data_bunch,
            dataset_name=dataset_name,
            dataset_params={"samples_ratio": samples_ratio},
        )


def run_estimator(estimator_name, data_bunch, dataset_name, dataset_params):
    """Find the best hyper-parameters for a given model and a given dataset."""

    X, y = data_bunch.X, data_bunch.y
    # scale_censoring = data_bunch.scale_censoring
    # shape_censoring = data_bunch.shape_censoring
    estimator = ESTIMATOR_GRID[estimator_name]["estimator"]
    param_grid = ESTIMATOR_GRID[estimator_name]["param_grid"]

    hp_search = RandomizedSearchCV(
        estimator,
        param_grid,
        cv=3,
        return_train_score=True,
        n_iter=N_ITER_CV,
        n_jobs=N_JOBS_CV,
    )
    hp_search.fit(
        X,
        y,
    )

    best_params = hp_search.best_params_
    best_estimator = hp_search.best_estimator_

    cols = [
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    best_results = pd.DataFrame(hp_search.cv_results_)[cols]
    best_results = best_results.iloc[hp_search.best_index_].to_dict()
    best_results["estimator_name"] = estimator_name

    best_estimator.fit(X, y)

    # hack for benchmarks
    best_estimator.y_train = y

    str_params = [str(v) for v in dataset_params.values()]
    str_params = "_".join([estimator_name, *str_params])
    path_profile = PATH_DAILY_SESSION / dataset_name / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_params, open(path_profile / "best_params.json", "w"))
    dump(best_estimator, path_profile / "best_estimator.joblib")
    json.dump(best_results, open(path_profile / "cv_results.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))


if __name__ == "__main__":
    run_all_synthetic_datasets("fine_and_gray")
    run_all_synthetic_datasets("aalen_johansen")
