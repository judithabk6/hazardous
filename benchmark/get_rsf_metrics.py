
import pandas as pd
import numpy as np
from hazardous.metrics._negative_loss_weighted import integrated_brier_score_incidence, integrated_log_incidence
from display_utils import load_dataset
from hazardous.utils import get_n_events
from hazardous.metrics._concordance import concordance_index_ipcw
from scipy.interpolate import interp1d

from hazardous.metrics._brier_score import (
    brier_score_incidence,
)


def get_test_data(DATASET_NAME, seed=0):
    bunch = load_dataset(DATASET_NAME, data_params={}, random_state=seed)
    X_test, y_test = bunch.X, bunch.y
    return X_test, y_test

def compute_metrics(
    y_train,
    y_test,
    all_y_pred,
    time_grid,
    truncation_quantiles=[0.25, 0.5, 0.75],
    n_events=3
):
    y_multiclass_train = y_train.copy(deep=True)
    y_multiclass_test = y_test.copy(deep=True)
    y_train_single = y_train.copy(deep=True)
    y_test_single = y_test.copy(deep=True)

    results = []
    for estimator_name, y_pred in all_y_pred.items():
        times_event = np.hstack([[0], time_grid, [np.inf]])
        y_pred_long = np.concatenate([np.moveaxis(np.array([np.zeros(y_pred[:, :, 0].shape)]), 0, 2),
                            y_pred,
                            np.moveaxis(np.array([y_pred[:, :, -1]]), 0, 2)],
                            axis=2)

        times = np.quantile(times_event, truncation_quantiles)
        for event in range(1, n_events+1):
            y_train_single  = y_train_single.assign(event=(y_multiclass_train["event"] == (event)).astype(bool))
            y_test_single  = y_test_single.assign(event=(y_multiclass_test["event"] == (event)).astype(bool))

            ibs = integrated_brier_score_incidence(
                y_train=y_multiclass_train,
                y_test=y_multiclass_test,
                y_pred=y_pred[event-1],
                times=time_grid,
                event_of_interest=event,
            )
            log_loss = integrated_log_incidence(
                y_train=y_multiclass_train,
                y_test=y_multiclass_test,
                y_pred=y_pred[event-1],
                times=time_grid,
                event_of_interest=event,
            )

            def get_target(df):
                return (df["duration"].values, df["event"].values)

            durations_train, events_train = get_target(y_train_single)
            et_train = np.array(
                [(events_train[i], durations_train[i]) for i in range(len(events_train))],
                dtype=[("e", bool), ("t", float)],
            )

            durations_test, events_test = get_target(y_test_single)
            et_test = np.array(
                [(events_test[i], durations_test[i]) for i in range(len(events_test))],
                dtype=[("e", bool), ("t", float)],
            )

            summary_y_pred = np.zeros((n_events, y_pred.shape[1], len(times)))
            for idx_event in range(n_events):
                for idx_patient in range(y_pred.shape[1]):
                    summary_y_pred[idx_event, idx_patient, :] = interp1d(
                        x=times_event,
                        y=y_pred_long[idx_event, idx_patient, :],
                        kind="linear",
                        axis=0
                        )(times)

            surv_curve = (1 - np.sum(summary_y_pred, axis=0))[None, :]

            summary_y_pred = np.concatenate([surv_curve, summary_y_pred], axis=0)

            for time_idx, (time, quantile) in enumerate(zip(times, truncation_quantiles)):
                y_pred_at_t = summary_y_pred[event, :, time_idx]
                try:
                    # ct_index, _, _, _, _ = metrics.concordance_index_ipcw(
                    #     et_train, #y_train_single,
                    #     et_test, #y_test_single,
                    #     y_pred_at_t,
                    #     tau=time,
                    #     tied_tol=1e-08
                    # )
                    # print(time, event, ct_index)
                    ct_index, _, _, _, _ = concordance_index_ipcw(
                        y_train_single,
                        y_test_single,
                        y_pred_at_t,
                        tau=time,
                        tied_tol=1e-08
                    )
                except ValueError:
                    ct_index = np.nan
                    print('no event before time horizon')

                y_pred_time = summary_y_pred[:, :, time_idx]
                mask = (y_multiclass_test["event"] == 0) & (y_multiclass_test["duration"] < times[time_idx])
                y_pred_time = y_pred_time[:, ~mask]
                
                y_pred_class = y_pred_time.argmax(axis=0)
                y_test_class = y_multiclass_test["event"] * (y_multiclass_test["duration"] < times[time_idx])
                y_test_class = y_test_class.loc[~mask]

                acc_in_time = (y_test_class.values == y_pred_class).mean()
                results.append(
                    dict(
                        estimator_name=estimator_name,
                        ibs=ibs,
                        log_loss=log_loss,
                        event=event,
                        truncation_q=quantile,
                        ct_index=ct_index,
                        acc_in_time=acc_in_time
                    )
                )

    results = pd.DataFrame(results)
    return results


# for seer
truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
results_all_seeds = []


for seed in range(5):
    test_df = pd.read_csv('seer_test_{}.csv'.format(seed))
    y_test = test_df[['event', 'duration']]
    #X_test, y_test = get_test_data("seer", seed)
    n_events = get_n_events(y_test["event"])
    #time_grid = make_time_grid(y_test["duration"], n_steps=100)
    cif = pd.read_csv('test_cif_seer_srf_50000_{}.csv'.format(seed))
    time_grid = pd.read_csv('test_timegrid_seer_srf_50000_{}.csv'.format(seed)).values
    y_pred = cif.values.reshape(len(cif), len(time_grid), n_events, order='F')
    y_pred = np.moveaxis(y_pred, 2, 0)
    
    
    times = np.quantile(time_grid, truncation_quantiles)
    y_train = pd.read_csv('seer_srf_50000_{}.csv'.format(seed))[['event', 'duration']]
    all_y_pred = {'srf_cr_50000': y_pred}
    results = compute_metrics(
        y_train, y_test, all_y_pred, time_grid.ravel(), truncation_quantiles, n_events
    )

    results["seed"] = seed
    results_all_seeds.append(results)
    n_events = get_n_events(y_test["event"])
    #time_grid = make_time_grid(y_test["duration"], n_steps=100)
    cif = pd.read_csv('test_cif_seer_srf_100000_{}.csv'.format(seed))
    time_grid = pd.read_csv('test_timegrid_seer_srf_100000_{}.csv'.format(seed)).values
    y_pred = cif.values.reshape(len(cif), len(time_grid), n_events, order='F')
    y_pred = np.moveaxis(y_pred, 2, 0)
    

    times = np.quantile(time_grid, truncation_quantiles)
    y_train = pd.read_csv('seer_srf_100000_{}.csv'.format(seed))[['event', 'duration']]
    all_y_pred = {'srf_cr_100000': y_pred}
    results = compute_metrics(
        y_train, y_test, all_y_pred, time_grid.ravel(), truncation_quantiles, n_events
    )


    results["seed"] = seed
    results_all_seeds.append(results)
# %%
results_all_seeds = pd.concat(results_all_seeds)
# %%
results_all_seeds
results_all_seeds.to_csv('new_all_results_srf_cr_log_ibs.csv', index=False)

# for simulated data
truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
results_simu_all_seeds = []

for n_samples in [1_000, 5_000, 10_000, 20_000]:
    for seed in range(5):
        test_df = pd.read_csv(
            'simulated_data/weibull_test_seed_{}.csv'.format(seed))
        y_test = test_df[['event', 'duration']]
        #X_test, y_test = get_test_data("seer", seed)
        n_events = get_n_events(y_test["event"])
        #time_grid = make_time_grid(y_test["duration"], n_steps=100)
        cif = pd.read_csv(
            'simulated_data/weibull_test_cif_nsamples_{}_seed_{}.csv'.format(
                n_samples, seed))
        time_grid = pd.read_csv(
            'simulated_data/weibull_test_timegrid_nsamples_{}_seed_{}.csv'.format(
                n_samples, seed)).values.ravel()
        y_pred = cif.values.reshape(len(cif),
                                    len(time_grid),
                                    n_events,
                                    order='F')
        y_pred = np.moveaxis(y_pred, 2, 0)


        y_train = pd.read_csv(
            'simulated_data/weibull_train_nsamples_{}_seed_{}.csv'.format(
                n_samples, seed))[['event', 'duration']]
        all_y_pred = {'simulated_data_srf_cr': y_pred}
        results = compute_metrics(
            y_train,
            y_test,
            all_y_pred,
            time_grid,
            truncation_quantiles,
            n_events,
        )



        results['n_samples'] = n_samples
        results["seed"] = seed
        results_simu_all_seeds.append(results)
# %%
results_simu_all_seeds = pd.concat(results_simu_all_seeds)
# %%
results_simu_all_seeds
results_simu_all_seeds.to_csv('new_results_rsf_cr_simulated_data.csv', index=False, sep=',')


truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
results_simu_all_censoring = []

for seed in range(5):
    for censoring in ('0-8', '2-5'):

        test_df = pd.read_csv(
            'simulated_data/weibull_test_seed_{}.csv'.format(seed))
        y_test = test_df[['event', 'duration']]
        #X_test, y_test = get_test_data("seer", seed)
        n_events = get_n_events(y_test["event"])
        #time_grid = make_time_grid(y_test["duration"], n_steps=100)
        cif = pd.read_csv(
            'simulated_data/weibull_test_cif__20000_censoring_{}_seed_{}.csv'.format(
                censoring, seed))

        time_grid = pd.read_csv(
            'simulated_data/weibull_test_timegrid__20000_censoring_{}_seed_{}.csv'.format(
                censoring, seed)).values.ravel()
        y_pred = cif.values.reshape(len(cif),
                                    len(time_grid),
                                    n_events,
                                    order='F')
        y_pred = np.moveaxis(y_pred, 2, 0)


        y_train = pd.read_csv(
            'simulated_data/weibull_train_nsamples_20000_censoring_{}_seed_{}.csv'.format(
                censoring, seed))[['event', 'duration']]
        all_y_pred = {'simulated_data_srf_cr': y_pred}
        results = compute_metrics(
            y_train,
            y_test,
            all_y_pred,
            time_grid,
            truncation_quantiles,
            n_events,
        )



        results['censoring'] = censoring
        results["seed"] = seed
        results_simu_all_censoring.append(results)
# %%
results_simu_all_censoring = pd.concat(results_simu_all_censoring)
# %%
results_simu_all_censoring
results_simu_all_censoring.to_csv('new_results_rsf_cr_simulated_data_censoring.csv', index=False, sep=',')



truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
results_simu_all_censoring = []

for seed in range(5):
    for censoring in ('0-8', '2-5'):

        test_df = pd.read_csv(
            'simulated_data/weibull_test_seed_{}.csv'.format(seed))
        y_test = test_df[['event', 'duration']]
        #X_test, y_test = get_test_data("seer", seed)
        n_events = get_n_events(y_test["event"])
        #time_grid = make_time_grid(y_test["duration"], n_steps=100)
        cif = pd.read_csv(
            'simulated_data/weibull_test_cif__10000_censoring_{}_seed_{}.csv'.format(
                censoring, seed))

        time_grid = pd.read_csv(
            'simulated_data/weibull_test_timegrid__10000_censoring_{}_seed_{}.csv'.format(
                censoring, seed)).values.ravel()
        y_pred = cif.values.reshape(len(cif),
                                    len(time_grid),
                                    n_events,
                                    order='F')
        y_pred = np.moveaxis(y_pred, 2, 0)


        y_train = pd.read_csv(
            'simulated_data/weibull_train_nsamples_10000_censoring_{}_seed_{}.csv'.format(
                censoring, seed))[['event', 'duration']]
        all_y_pred = {'simulated_data_srf_cr': y_pred}
        results = compute_metrics(
            y_train,
            y_test,
            all_y_pred,
            time_grid,
            truncation_quantiles,
            n_events,
        )



        results['censoring'] = censoring
        results["seed"] = seed
        results_simu_all_censoring.append(results)
# %%
results_simu_all_censoring = pd.concat(results_simu_all_censoring)
# %%
results_simu_all_censoring
results_simu_all_censoring.to_csv('new_results_rsf_cr_simulated_data_censoring_10000.csv', index=False, sep=',')




    for censoring in ('0-8', '2-5'):
seed = 0
censoring = '0-8'
test_df = pd.read_csv(
    'simulated_data/weibull_test_seed_{}.csv'.format(seed))
y_test = test_df[['event', 'duration']]
#X_test, y_test = get_test_data("seer", seed)
n_events = get_n_events(y_test["event"])
#time_grid = make_time_grid(y_test["duration"], n_steps=100)
cif = pd.read_csv(
    'simulated_data/weibull_test_cif__20000_censoring_{}_seed_{}.csv'.format(
        censoring, seed))

time_grid = pd.read_csv(
    'simulated_data/weibull_test_timegrid__20000_censoring_{}_seed_{}.csv'.format(
        censoring, seed)).values.ravel()
y_pred = cif.values.reshape(len(cif),
                            len(time_grid),
                            n_events,
                            order='F')
y_pred = np.moveaxis(y_pred, 2, 0)
y_train = pd.read_csv(
    'simulated_data/weibull_train_nsamples_20000_censoring_{}_seed_{}.csv'.format(
        censoring, seed))[['event', 'duration']]

all_brier = np.empty((n_events, len(time_grid)))
for idx in range(n_events):
    all_brier[idx, :] = brier_score_incidence(
        y_train,
        y_test,
        y_pred[idx],
        times=time_grid,
        event_of_interest=idx + 1,
    )

        ax.plot(time_grid, brier_scores, label=estimator_name)
# # event_of_interest = 1

# y_test = y_test.assign(event_of_interest=(y_test.event==event_of_interest))
# _, y_test_sksurv = get_x_y(y_test,
#                            attr_labels=['event_of_interest', 'duration'],
#                            pos_label=True,
#                            survival=True)
# _, y_train_sksurv = get_x_y(y_train,
#                            attr_labels=['event', 'duration'],
#                            pos_label=event_of_interest,
#                            survival=True)
# concordance_index_ipcw(y_train_sksurv,
#                        y_test_sksurv,
#                        y_pred[event_of_interest-1, :, :],
#                        tau=times[0], tied_tol=1e-08)


_, y_test_sksurv = get_x_y(y_test_single,
                           attr_labels=['event', 'duration'],
                           pos_label=1,
                           survival=True)
_, y_train_sksurv = get_x_y(y_train_single,
                           attr_labels=['event', 'duration'],
                           pos_label=1,
                           survival=True)
# metrics.concordance_index_ipcw(y_train_sksurv,
#                        y_test_sksurv,
#                        y_pred_at_t,
#                        tau=time, tied_tol=1e-08)



