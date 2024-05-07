
def compute_metrics(
    y_train,
    y_test,
    all_y_pred,
    time_grid,
    times,
    truncation_quantiles=[0.25, 0.5, 0.75],
    n_events=3,
):
    results = []
    for event in (1, 2, 3):
        for estimator_name, y_pred in all_y_pred.items():
            ibs = integrated_brier_score_incidence(
                y_train=y_train,
                y_test=y_test,
                y_pred=y_pred[event-1],
                times=time_grid,
                event_of_interest=event,
            )

            def get_target(df):
                return (df["duration"].values, df["event"].values)

            durations_train, events_train = get_target(y_train)
            et_train = np.array(
                [(events_train[i], durations_train[i]) for i in range(len(events_train))],
                dtype=[("e", bool), ("t", float)],
            )

            durations_test, events_test = get_target(y_test)
            et_test = np.array(
                [(events_test[i], durations_test[i]) for i in range(len(events_test))],
                dtype=[("e", bool), ("t", float)],
            )
            for time_idx, (time, quantile) in enumerate(zip(times, truncation_quantiles)):
                y_pred_at_t = y_pred[event-1, :, time_idx]
                ct_index, _, _, _, _ = concordance_index_ipcw(
                    et_train,
                    et_test,
                    y_pred_at_t,
                    tau=time,
                )
                results.append(
                    dict(
                        estimator_name=estimator_name,
                        ibs=ibs,
                        event=event,
                        truncation_q=quantile,
                        ct_index=ct_index,
                    )
                )

    results = pd.DataFrame(results)
    return results


truncation_quantiles = [0.25, 0.5, 0.75]
results_all_seeds = []
for seed in range(5):
    X_test, y_test = get_test_data("seer", seed)
    #time_grid = make_time_grid(y_test["duration"], n_steps=100)
    cif = pd.read_csv('test_cif_seer_srf_50000_{}.csv'.format(seed))
    y_pred = cif.values.reshape(len(cif), len(time_grid), n_events, order='F')
    y_pred = np.moveaxis(y_pred, 2, 0)
    time_grid = pd.read_csv('test_timegrid_seer_srf_50000_{}.csv'.format(seed)).values
    n_events = get_n_events(y_test["event"])
    times = np.quantile(time_grid, truncation_quantiles)
    y_train = pd.read_csv('seer_srf_50000_{}.csv'.format(seed))[['event', 'duration']]
    all_y_pred = {'srf_cr_50000': y_pred}
    results = compute_metrics(
        y_train, y_test, all_y_pred, time_grid.ravel(), times, truncation_quantiles, n_events
    )


    results["seed"] = seed
    results_all_seeds.append(results)

for seed in range(5):
    X_test, y_test = get_test_data("seer", seed)
    #time_grid = make_time_grid(y_test["duration"], n_steps=100)
    cif = pd.read_csv('test_cif_seer_srf_100000_{}.csv'.format(seed))
    y_pred = cif.values.reshape(len(cif), len(time_grid), n_events, order='F')
    y_pred = np.moveaxis(y_pred, 2, 0)
    time_grid = pd.read_csv('test_timegrid_seer_srf_100000_{}.csv'.format(seed)).values
    n_events = get_n_events(y_test["event"])
    times = np.quantile(time_grid, truncation_quantiles)
    y_train = pd.read_csv('seer_srf_100000_{}.csv'.format(seed))[['event', 'duration']]
    all_y_pred = {'srf_cr_100000': y_pred}
    results = compute_metrics(
        y_train, y_test, all_y_pred, time_grid.ravel(), times, truncation_quantiles, n_events
    )


    results["seed"] = seed
    results_all_seeds.append(results)
# %%
results_all_seeds = pd.concat(results_all_seeds)
# %%
results_all_seeds
