"""Traditional Algorithm Baselines for Benchmarking (T4.3).

Implements classical time-series and anomaly detection methods for comparison:
- ARIMA / AR linear models
- Isolation Forest
- Kalman Filter
- Exponential Smoothing (ETS)
- Simple moving average / exponential moving average

All baselines share a common interface for fair comparison.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Detection metrics (shared by every baseline and by MORPHOS)
# ----------------------------------------------------------------------
def calibrate_threshold(scores: np.ndarray, quantile: float = 0.9) -> float:
    """Pick an anomaly threshold from *clean* calibration scores.

    Using a quantile of the clean-score distribution gives every detector
    the same false-positive budget, which is what makes the precision /
    recall comparison between them meaningful.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("inf")
    return float(np.quantile(s, float(np.clip(quantile, 0.0, 1.0))))


def threshold_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary anomaly labels: 1 when score exceeds the threshold."""
    s = np.asarray(scores, dtype=float).ravel()
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return (s > threshold).astype(int)


def precision_recall(labels_true: np.ndarray,
                     labels_pred: np.ndarray) -> dict:
    """Precision / recall / F1 for binary anomaly labels."""
    y = np.asarray(labels_true).astype(int).ravel()
    p = np.asarray(labels_pred).astype(int).ravel()
    n = min(len(y), len(p))
    y, p = y[:n], p[:n]
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": int(np.sum(y == 1))}


def best_f1_over_thresholds(labels_true: np.ndarray,
                            scores: np.ndarray) -> dict:
    """Supervised upper bound: best F1 over all candidate thresholds.

    Reported for completeness only - the headline comparison uses each
    detector's clean-calibrated operating point instead.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if s.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "threshold": float("inf")}
    candidates = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 64)))
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
            "threshold": float("inf")}
    for thr in candidates:
        m = precision_recall(labels_true, threshold_labels(s, thr))
        if m["f1"] > best["f1"]:
            best = {**m, "threshold": float(thr)}
    return best

# Try to import statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.ar_model import AutoReg
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Try to import sklearn for Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class BaselineResult:
    """Result from a baseline model."""
    predictions: np.ndarray       # Shape: (n_steps, n_dims)
    residuals: np.ndarray         # Shape: (n_steps, n_dims)
    mse: np.ndarray              # Per-dimension MSE
    mae: np.ndarray              # Per-dimension MAE
    rmse: np.ndarray             # Per-dimension RMSE
    model_name: str
    fit_time: float
    predict_time: float
    # Anomaly detection (if applicable)
    anomaly_scores: Optional[np.ndarray] = None  # Shape: (n_steps,)
    anomaly_labels: Optional[np.ndarray] = None  # Shape: (n_steps,)


class TimeSeriesBaseline(ABC):
    """Abstract base class for time series baselines."""
    
    @abstractmethod
    def fit(self, train_data: np.ndarray) -> 'TimeSeriesBaseline':
        """Fit the model on training data.
        
        Args:
            train_data: Shape (n_train, n_dims)
        """
        pass
    
    @abstractmethod
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        """Predict on test data.
        
        Args:
            test_data: Shape (n_test, n_dims) - ground truth for computing residuals
            n_steps: Number of steps to predict ahead (default: len(test_data))
        """
        pass


class ARBaseline(TimeSeriesBaseline):
    """Autoregressive (AR) baseline - simple linear predictor."""
    
    def __init__(self, order: int = 5, name: str = "AR"):
        self.order = order
        self.name = name
        self.coeffs_ = None
        self.intercept_ = None
        self.n_dims_ = None
        self.train_mean_ = None
        self.train_std_ = None
    
    def fit(self, train_data: np.ndarray) -> 'ARBaseline':
        """Fit AR model using least squares (Yule-Walker)."""
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        n_samples, self.n_dims_ = train_data.shape
        
        # Normalize
        self.train_mean_ = train_data.mean(axis=0)
        self.train_std_ = train_data.std(axis=0) + 1e-8
        normalized = (train_data - self.train_mean_) / self.train_std_

        # Keep the tail of the training series so one-step prediction on
        # the test stream starts from real history instead of zeros.
        if len(normalized) >= self.order:
            self.tail_ = normalized[-self.order:].copy()
        else:
            pad = np.zeros((self.order - len(normalized), self.n_dims_))
            self.tail_ = np.vstack([pad, normalized])
        
        self.coeffs_ = np.zeros((self.n_dims_, self.order))
        self.intercept_ = np.zeros(self.n_dims_)
        
        for d in range(self.n_dims_):
            y = normalized[:, d]
            if len(y) <= self.order:
                # Not enough data, use simple mean
                self.intercept_[d] = y.mean()
                continue
            
            # Build lagged matrix: column i is y[t-i], so coeff[0] applies
            # to the most recent observation (matching the prediction path,
            # which feeds hist[-order:][::-1]).
            X = np.column_stack([
                y[self.order - 1 - i:-(i + 1)] for i in range(self.order)
            ])
            y_target = y[self.order:]
            
            # Add intercept column
            X = np.column_stack([np.ones(len(X)), X])
            
            # Least squares
            try:
                beta, *_ = np.linalg.lstsq(X, y_target, rcond=None)
                self.intercept_[d] = beta[0]
                self.coeffs_[d] = beta[1:]
            except np.linalg.LinAlgError:
                self.intercept_[d] = y.mean()
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        """Predict using fitted AR model."""
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        # Normalize test data using train stats
        normalized = (test_data - self.train_mean_) / self.train_std_
        
        predictions = np.zeros((n_test, self.n_dims_))

        # Teacher-forced one-step prediction: prediction for test[t] uses
        # test[t-1] .. test[t-order], seeded with the tail of the train
        # series.  (An earlier version used test[t] itself, which shifted
        # every prediction by one step and made the baseline look far
        # worse than it is.)
        for d in range(self.n_dims_):
            hist = [float(v) for v in self.tail_[:, d]]
            if len(hist) < self.order:
                hist = [0.0] * (self.order - len(hist)) + hist

            for t in range(n_test):
                pred_norm = self.intercept_[d] + np.dot(
                    self.coeffs_[d], np.asarray(hist[-self.order:][::-1]))

                # Denormalize
                predictions[t, d] = pred_norm * self.train_std_[d] + self.train_mean_[d]

                # Reveal the ground truth (teacher forcing)
                hist.append(float(normalized[t, d]))
        
        residuals = test_data[:n_test] - predictions
        mse = np.mean(residuals ** 2, axis=0)
        mae = np.mean(np.abs(residuals), axis=0)
        rmse = np.sqrt(mse)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
        )


class ARIMABaseline(TimeSeriesBaseline):
    """ARIMA baseline using statsmodels (if available)."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 0, 0), name: str = "ARIMA",
                 one_step: bool = True):
        self.order = order
        self.name = name
        # one_step=True re-applies the fitted parameters to the test stream
        # through the state-space filter, giving genuine one-step-ahead
        # predictions.  one_step=False falls back to a multi-step forecast.
        self.one_step = one_step
        self.models_ = []
        self.train_mean_ = None
        self.train_std_ = None
        self.n_dims_ = None
    
    def fit(self, train_data: np.ndarray) -> 'ARIMABaseline':
        if not HAS_STATSMODELS:
            raise RuntimeError("statsmodels not available for ARIMA")
        
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        n_samples, self.n_dims_ = train_data.shape
        
        # Normalize
        self.train_mean_ = train_data.mean(axis=0)
        self.train_std_ = train_data.std(axis=0) + 1e-8
        normalized = (train_data - self.train_mean_) / self.train_std_
        
        self.models_ = []
        for d in range(self.n_dims_):
            try:
                model = ARIMA(normalized[:, d], order=self.order)
                fitted = model.fit()
                self.models_.append(fitted)
            except Exception:
                # Fallback to simple mean
                self.models_.append(None)
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        if not HAS_STATSMODELS:
            raise RuntimeError("statsmodels not available for ARIMA")
        
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        predictions = np.zeros((n_test, self.n_dims_))
        
        for d in range(self.n_dims_):
            model = self.models_[d]
            if model is None:
                predictions[:, d] = self.train_mean_[d]
            else:
                try:
                    if self.one_step:
                        # Fixed parameters, filtered through the test stream
                        applied = model.apply(
                            (test_data[:n_test, d] - self.train_mean_[d]) /
                            self.train_std_[d])
                        one = np.asarray(applied.fittedvalues, dtype=float)
                        one = np.where(np.isfinite(one), one,
                                       self.train_mean_[d] / self.train_std_[d])
                        predictions[:, d] = (one * self.train_std_[d] +
                                             self.train_mean_[d])
                    else:
                        forecast = model.forecast(steps=n_test)
                        predictions[:, d] = (forecast * self.train_std_[d] +
                                             self.train_mean_[d])
                except Exception:
                    predictions[:, d] = self.train_mean_[d]
        
        residuals = test_data[:n_test] - predictions
        mse = np.mean(residuals ** 2, axis=0)
        mae = np.mean(np.abs(residuals), axis=0)
        rmse = np.sqrt(mse)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
        )


class IsolationForestBaseline(TimeSeriesBaseline):
    """Isolation Forest for anomaly detection (not forecasting).
    
    Note: This is primarily an anomaly detector, not a forecaster.
    We use it by training on normal data and scoring test points.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 name: str = "IsolationForest"):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.name = name
        self.model_ = None
        self.train_mean_ = None
        self.train_std_ = None
        self.n_dims_ = None
    
    def fit(self, train_data: np.ndarray) -> 'IsolationForestBaseline':
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not available for IsolationForest")
        
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        self.n_dims_ = train_data.shape[1]
        
        # Normalize
        self.train_mean_ = train_data.mean(axis=0)
        self.train_std_ = train_data.std(axis=0) + 1e-8
        normalized = (train_data - self.train_mean_) / self.train_std_
        
        # Fit Isolation Forest
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model_.fit(normalized)
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not available for IsolationForest")
        
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        normalized = (test_data[:n_test] - self.train_mean_) / self.train_std_
        
        # Get anomaly scores (negative = more anomalous)
        anomaly_scores = -self.model_.score_samples(normalized)
        # Predictions: 1 = normal, -1 = anomaly
        anomaly_labels = self.model_.predict(normalized)
        # Convert to binary: 1 = anomaly, 0 = normal
        anomaly_binary = (anomaly_labels == -1).astype(int)
        
        # For "predictions", we just return the input (Isolation Forest doesn't forecast)
        # The meaningful outputs are anomaly scores and labels
        predictions = test_data[:n_test].copy()  # Placeholder
        residuals = np.zeros_like(predictions)   # No meaningful residuals
        
        # Dummy metrics for compatibility
        mse = np.zeros(self.n_dims_)
        mae = np.zeros(self.n_dims_)
        rmse = np.zeros(self.n_dims_)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_binary,
        )


class KalmanFilterBaseline(TimeSeriesBaseline):
    """Kalman Filter baseline for tracking and prediction.
    
    Implements a simple linear Kalman filter with constant velocity model.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1,
                 name: str = "KalmanFilter"):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.name = name
        self.n_dims_ = None
        self.x_ = None  # State: [position, velocity] for each dim
        self.P_ = None  # Covariance
        self.F_ = None  # State transition
        self.H_ = None  # Observation
        self.Q_ = None  # Process noise covariance
        self.R_ = None  # Measurement noise covariance
    
    def _initialize(self, n_dims: int):
        """Initialize Kalman filter matrices."""
        self.n_dims_ = n_dims
        # State: [pos_1, vel_1, pos_2, vel_2, ...]
        state_dim = 2 * n_dims
        
        self.x_ = np.zeros(state_dim)
        self.P_ = np.eye(state_dim) * 100  # High initial uncertainty
        
        # State transition: pos += vel, vel unchanged
        self.F_ = np.eye(state_dim)
        for i in range(n_dims):
            self.F_[2*i, 2*i+1] = 1.0  # dt = 1 (per step)
        
        # Observation: we only measure position
        self.H_ = np.zeros((n_dims, state_dim))
        for i in range(n_dims):
            self.H_[i, 2*i] = 1.0
        
        # Process noise
        self.Q_ = np.eye(state_dim) * self.process_noise
        for i in range(n_dims):
            # Correlated pos-vel noise
            self.Q_[2*i, 2*i] = self.process_noise * 0.25
            self.Q_[2*i, 2*i+1] = self.process_noise * 0.5
            self.Q_[2*i+1, 2*i] = self.process_noise * 0.5
            self.Q_[2*i+1, 2*i+1] = self.process_noise
        
        # Measurement noise
        self.R_ = np.eye(n_dims) * self.measurement_noise
    
    def fit(self, train_data: np.ndarray) -> 'KalmanFilterBaseline':
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        n_samples, n_dims = train_data.shape
        
        self._initialize(n_dims)
        
        # Initialize state with first observation
        self.x_[::2] = train_data[0]  # positions
        # Estimate initial velocity from first few samples
        if n_samples > 1:
            self.x_[1::2] = train_data[1] - train_data[0]
        
        # Run filter on training data to converge
        for t in range(n_samples):
            self._predict()
            self._update(train_data[t])
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def _predict(self):
        """Predict step."""
        self.x_ = self.F_ @ self.x_
        self.P_ = self.F_ @ self.P_ @ self.F_.T + self.Q_
    
    def _update(self, z: np.ndarray):
        """Update step with measurement z."""
        y = z - self.H_ @ self.x_  # Innovation
        S = self.H_ @ self.P_ @ self.H_.T + self.R_  # Innovation covariance
        K = self.P_ @ self.H_.T @ np.linalg.inv(S)  # Kalman gain
        self.x_ = self.x_ + K @ y
        self.P_ = (np.eye(len(self.x_)) - K @ self.H_) @ self.P_
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        predictions = np.zeros((n_test, self.n_dims_))
        
        # Make a copy of state for rolling prediction
        x = self.x_.copy()
        P = self.P_.copy()
        
        for t in range(n_test):
            # Predict
            x = self.F_ @ x
            P = self.F_ @ P @ self.F_.T + self.Q_
            
            # Predicted position
            pred = x[::2].copy()
            predictions[t] = pred
            
            # Update with actual measurement (teacher forcing for evaluation)
            z = test_data[t]
            y = z - self.H_ @ x
            S = self.H_ @ P @ self.H_.T + self.R_
            K = P @ self.H_.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(len(x)) - K @ self.H_) @ P
        
        residuals = test_data[:n_test] - predictions
        mse = np.mean(residuals ** 2, axis=0)
        mae = np.mean(np.abs(residuals), axis=0)
        rmse = np.sqrt(mse)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
        )


class EMABaseline(TimeSeriesBaseline):
    """Exponential Moving Average baseline."""
    
    def __init__(self, alpha: float = 0.3, name: str = "EMA"):
        self.alpha = alpha
        self.name = name
        self.ema_ = None
        self.n_dims_ = None
    
    def fit(self, train_data: np.ndarray) -> 'EMABaseline':
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        self.n_dims_ = train_data.shape[1]
        
        # Initialize EMA with first value
        self.ema_ = train_data[0].copy()
        
        # Update through training data
        for t in range(1, len(train_data)):
            self.ema_ = self.alpha * train_data[t] + (1 - self.alpha) * self.ema_
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        predictions = np.zeros((n_test, self.n_dims_))
        ema = self.ema_.copy()
        
        for t in range(n_test):
            predictions[t] = ema
            # Update EMA with actual (teacher forcing)
            ema = self.alpha * test_data[t] + (1 - self.alpha) * ema
        
        residuals = test_data[:n_test] - predictions
        mse = np.mean(residuals ** 2, axis=0)
        mae = np.mean(np.abs(residuals), axis=0)
        rmse = np.sqrt(mse)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
        )


class NaiveBaseline(TimeSeriesBaseline):
    """Naive baseline: predict last observed value (random walk)."""
    
    def __init__(self, name: str = "Naive"):
        self.name = name
        self.last_value_ = None
        self.n_dims_ = None
    
    def fit(self, train_data: np.ndarray) -> 'NaiveBaseline':
        import time
        start = time.perf_counter()
        
        train_data = np.asarray(train_data)
        self.n_dims_ = train_data.shape[1]
        self.last_value_ = train_data[-1].copy()
        
        self.fit_time_ = time.perf_counter() - start
        return self
    
    def predict(self, test_data: np.ndarray, n_steps: Optional[int] = None) -> BaselineResult:
        import time
        start = time.perf_counter()
        
        test_data = np.asarray(test_data)
        n_test = len(test_data) if n_steps is None else min(n_steps, len(test_data))
        
        predictions = np.tile(self.last_value_, (n_test, 1))
        residuals = test_data[:n_test] - predictions
        mse = np.mean(residuals ** 2, axis=0)
        mae = np.mean(np.abs(residuals), axis=0)
        rmse = np.sqrt(mse)
        
        predict_time = time.perf_counter() - start
        
        return BaselineResult(
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            mae=mae,
            rmse=rmse,
            model_name=self.name,
            fit_time=self.fit_time_,
            predict_time=predict_time,
        )


def get_all_baselines() -> List[TimeSeriesBaseline]:
    """Get list of all available baseline models."""
    baselines = [
        NaiveBaseline(),
        EMABaseline(alpha=0.3),
        EMABaseline(alpha=0.1, name="EMA_slow"),
        ARBaseline(order=3, name="AR3"),
        ARBaseline(order=5, name="AR5"),
        ARBaseline(order=10, name="AR10"),
        KalmanFilterBaseline(process_noise=0.01, measurement_noise=0.1),
        KalmanFilterBaseline(process_noise=0.1, measurement_noise=1.0, name="Kalman_noisy"),
    ]
    
    if HAS_STATSMODELS:
        baselines.extend([
            ARIMABaseline(order=(3, 0, 0), name="ARIMA_3_0_0"),
            ARIMABaseline(order=(5, 0, 0), name="ARIMA_5_0_0"),
            ARIMABaseline(order=(5, 1, 0), name="ARIMA_5_1_0"),
        ])
    
    if HAS_SKLEARN:
        baselines.extend([
            IsolationForestBaseline(contamination=0.05, name="IForest_05"),
            IsolationForestBaseline(contamination=0.1, name="IForest_10"),
            IsolationForestBaseline(contamination=0.2, name="IForest_20"),
        ])
    
    return baselines


def residual_scores(result: BaselineResult) -> np.ndarray:
    """Anomaly score from a forecaster: squared one-step prediction error."""
    r = np.asarray(result.residuals, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    return np.sum(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) ** 2,
                  axis=1)


def run_baseline_comparison(train_data: np.ndarray, test_data: np.ndarray,
                             baselines: Optional[List[TimeSeriesBaseline]] = None) -> dict:
    """Run all baselines and return comparison results."""
    if baselines is None:
        baselines = get_all_baselines()
    
    results = {}
    
    for baseline in baselines:
        name = baseline.name
        print(f"[baselines] Running {name}...")
        try:
            baseline.fit(train_data)
            result = baseline.predict(test_data)
            results[name] = {
                "mse": result.mse.tolist(),
                "mae": result.mae.tolist(),
                "rmse": result.rmse.tolist(),
                "mean_mse": float(np.mean(result.mse)),
                "mean_mae": float(np.mean(result.mae)),
                "mean_rmse": float(np.mean(result.rmse)),
                "fit_time": result.fit_time,
                "predict_time": result.predict_time,
                "model_name": result.model_name,
            }
            if result.anomaly_scores is not None:
                results[name]["anomaly_scores"] = result.anomaly_scores.tolist()
                results[name]["anomaly_labels"] = result.anomaly_labels.tolist()
            
            print(f"  {name}: MAE={np.mean(result.mae):.4f}, RMSE={np.mean(result.rmse):.4f}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            results[name] = {"error": str(e)}
    
    return results