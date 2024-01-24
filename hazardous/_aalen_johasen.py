import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
)
from hazardous.utils import check_y_survival


class AalenJohansenEstimator(BaseEstimator):
    """Aalen Johasen competing risk estimator.

    Parameters
    ----------
    random_state : default=None
        Used to subsample X during fit when X has more samples
        than max_fit_samples.
    """

    def __init__(
        self,
        max_fit_samples=10_000,
        random_state=None,
        calculate_variance=False,
        seed=0,
    ):
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.calculate_variance = calculate_variance
        self.seed = seed

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input covariates

        y : pandas.DataFrame of shape (n_samples, 2)
            The target, with columns 'event' and 'duration'.

        Returns
        -------
        self : fitted instance of FineGrayEstimator
        """
        X = self._check_input(X, y)
        event, duration = check_y_survival(y)

        self.times_ = np.unique(duration[event > 0])
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

        self.aj_fitter_events_ = []
        for event_id in self.event_ids_[1:]:
            aj_fitter_event = AalenJohansenFitter(
                calculate_variance=self.calculate_variance, seed=self.seed
            )
            aj_fitter_event.fit(duration, event, event_of_interest=event_id)

            self.aj_fitter_events_.append(aj_fitter_event)

        self.y_train = y

        self.times_ = np.unique(duration[event > 0])
        return self

    def predict_cumulative_incidence(self, X, times=None):
        """Predict the conditional cumulative incidence.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)

        times : ndarray of shape (n_times,), default=None
            The time steps to estimate the cumulative incidence at.
            * If set to None, the duration of the event of interest
              seen during fit 'times_' is used.
            * If not None, this performs a linear interpolation for each sample.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_times)
            The conditional cumulative cumulative incidence at times.
        """
        check_is_fitted(self, "aj_fitter_events_")

        all_event_y_pred = []
        if times is None:
            times = self.times_
        all_event_y_pred_x = np.ones((X.shape[0], len(times)))
        for event_id in self.event_ids_[1:]:
            # Interpolate each sample
            cif = self.aj_fitter_events_[event_id - 1].cumulative_density_
            times_event = cif.index
            y_pred = cif.iloc[:, 0].values
            times_event = np.hstack([[0], times_event, [np.inf]])
            y_pred = np.hstack([[0], y_pred, [y_pred[-1]]])

            y_pred_ = interp1d(
                x=times_event,
                y=y_pred,
                kind="linear",
            )(times)

            all_event_y_pred.append(y_pred_)

        surv_curve = 1 - np.sum(all_event_y_pred, axis=0)
        all_event_y_pred = [surv_curve] + all_event_y_pred
        all_event_y_pred = np.asarray(all_event_y_pred)
        all_event_y_pred_x = all_event_y_pred[:, :, None] * np.ones(X.shape[0])
        return all_event_y_pred_x.swapaxes(1, 2)

    def _check_input(self, X, y):
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)

        if not hasattr(y, "__dataframe__"):
            raise TypeError(f"'y' must be a Pandas dataframe, got {type(y)}.")

        # Check no categories
        numeric_columns = X.select_dtypes("number").columns
        if numeric_columns.shape[0] != X.shape[1]:
            categorical_columns = set(X.columns).difference(list(numeric_columns))
            raise ValueError(
                f"Categorical columns {categorical_columns} need to be encoded."
            )

        return X

    def score(self, X, y, shape_censoring=None, scale_censoring=None):
        """Return

        #TODO: implement time integrated NLL.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []

        for idx, event_id in enumerate(self.event_ids_[1:]):
            predicted_curves_for_event = predicted_curves[idx]
            if scale_censoring is not None and shape_censoring is not None:
                ibs_event = integrated_brier_score_incidence_oracle(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event_id,
                )

            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    event_of_interest=event_id,
                )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)
