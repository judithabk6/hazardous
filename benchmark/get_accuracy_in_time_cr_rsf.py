
seed = 0
#y_pred = pd.read_csv("test_cif_seer_srf_100000_0.csv")
y_test = pd.read_csv("seer_test_0.csv")[['event', 'duration']]
X_test, y_test = get_test_data("seer", seed)
#time_grid = make_time_grid(y_test["duration"], n_steps=100)
cif = pd.read_csv('test_cif_seer_srf_100000_{}.csv'.format(seed))
y_pred = cif.values.reshape(len(cif), len(time_grid), n_events, order='F')
y_pred = np.moveaxis(y_pred, 2, 0)
time_grid = pd.read_csv('test_timegrid_seer_srf_100000_{}.csv'.format(seed)).values

truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
times = np.quantile(time_grid, truncation_quantiles)

# Create the censoring first column
y_pred_censore = (
    1 - y_pred.sum(axis=0)
)[None, :, :]
y_pred = np.concatenate([y_pred_censore, y_pred], axis=0)

results = []
for time_idx in range(len(times)):
    y_pred_time = y_pred[:, :, time_idx]
    mask = (y_test["event"] == 0) & (y_test["duration"] < times[time_idx])
    y_pred_time = y_pred_time[:, ~mask]
    
    y_pred_class = y_pred_time.argmax(axis=0)
    y_test_class = y_test["event"] * (y_test["duration"] < times[time_idx])
    y_test_class = y_test_class.loc[~mask]

    score = (y_test_class.values == y_pred_class).mean()

    results.append(
        dict(
            time=times[time_idx],
            quantile=truncation_quantiles[time_idx],
            score=score,
        )
    )

results = pd.DataFrame(results)
results.to_csv("rsf_acc_in_time.csv", index=False)