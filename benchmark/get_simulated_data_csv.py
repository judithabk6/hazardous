
from itertools import product
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
import pandas as pd


DATASET_GRID = {
    "weibull": {
        "n_events": [3],
        "n_samples": [1_000, 5_000, 10_000, 20_000],
        "censoring_relative_scale": [1.5],
        "complex_features": [False],
        "independent_censoring": [False],
    },
}

dataset_name = "weibull"

dataset_grid = DATASET_GRID[dataset_name]
grid_dataset_params = list(product(*dataset_grid.values()))


for dataset_params in grid_dataset_params:
    dataset_params = dict(zip(dataset_grid.keys(), dataset_params))
    nb_samples = dataset_params['n_samples']
    for seed in range(5):
        print(dataset_params)

        
        data_bunch = make_synthetic_competing_weibull(
            random_state=seed, **dataset_params)
        full_df = pd.concat((data_bunch['X'], data_bunch['y']), axis=1)
        full_df.to_csv(
            'simulated_data/weibull_train_nsamples_{}_seed_{}.csv'.format(
                nb_samples, seed),
            index=False, sep=',')


        # dataset_params["n_samples"] = 10_000


for seed in range(5):
    dataset_params["n_samples"] = 10_000
    test_bunch = make_synthetic_competing_weibull(
        random_state=seed + 100, **dataset_params)
    test_df = pd.concat((test_bunch['X'], test_bunch['y']), axis=1)
    test_df.to_csv(
        'simulated_data/weibull_test_seed_{}.csv'.format(seed),
        index=False, sep=',')


DATASET_GRID = {
    "weibull": {
        "n_events": [3],
        "n_samples": [20_000],
        "censoring_relative_scale": [0.8, 2.5],
        "complex_features": [False],
        "independent_censoring": [False],
    },
}

dataset_name = "weibull"

dataset_grid = DATASET_GRID[dataset_name]
grid_dataset_params = list(product(*dataset_grid.values()))


for dataset_params in grid_dataset_params:
    dataset_params = dict(zip(dataset_grid.keys(), dataset_params))
    nb_samples = dataset_params['n_samples']
    censoring = dataset_params['censoring_relative_scale']
    for seed in range(5):
        print(dataset_params)

        
        data_bunch = make_synthetic_competing_weibull(
            random_state=seed, **dataset_params)
        full_df = pd.concat((data_bunch['X'], data_bunch['y']), axis=1)
        full_df.to_csv(
            'simulated_data/weibull_train_nsamples_{}_censoring_{}_seed_{}.csv'.format(
                nb_samples, str(censoring).replace('.', '-'), seed),
            index=False, sep=',')


        # dataset_params["n_samples"] = 10_000


for seed in range(5):
    dataset_params["n_samples"] = 10_000
    test_bunch = make_synthetic_competing_weibull(
        random_state=seed + 100, **dataset_params)
    test_df = pd.concat((test_bunch['X'], test_bunch['y']), axis=1)
    test_df.to_csv(
        'simulated_data/weibull_test_censoring_{}_seed_{}.csv'.format(str(censoring).replace('.', '-'), seed),
        index=False, sep=',')

