"""Online linear predictor with NLMS weight updates.

Learns  z_{t+1} ~= W . [features_t, 1]  causally: predict first, then
update with the revealed target.  Updates are confidence-weighted and
clipped so a burst of garbage cannot blow the weights up.
"""

from __future__ import annotations

import numpy as np


class NLMSPredictor:
    def __init__(self, n_in: int, n_out: int = 3, mu: float = 0.05,
                 eps: float = 1e-2, pred_clip: float = 1.5,
                 max_update_norm: float = 0.5):
        self.n_in = n_in
        self.n_out = n_out
        self.mu = mu
        self.eps = eps
        self.pred_clip = pred_clip
        self.max_update_norm = max_update_norm
        self.W = np.zeros((n_out, n_in + 1), dtype=float)
        # Readout input standardisation (set by fit_batch).
        self._x_mu: np.ndarray | None = None
        self._x_sd: np.ndarray | None = None

    def _design(self, feat: np.ndarray) -> np.ndarray:
        x = np.asarray(feat, dtype=float).ravel()
        if self._x_mu is not None and self._x_mu.shape == x.shape:
            x = (x - self._x_mu) / self._x_sd
        return np.concatenate([x, [1.0]])

    def predict(self, feat: np.ndarray) -> np.ndarray:
        x = self._design(feat)
        y = self.W @ x
        if not np.all(np.isfinite(y)):
            return np.zeros(self.n_out)
        return np.clip(y, -self.pred_clip, self.pred_clip)

    def update(self, feat: np.ndarray, target: np.ndarray,
               weight: float = 1.0) -> float:
        """Returns the squared prediction error (pre-update)."""
        x = self._design(feat)
        t = np.asarray(target, dtype=float).ravel()
        if t.shape[0] != self.n_out or not np.all(np.isfinite(t)):
            return float("nan")
        y = self.W @ x
        if not np.all(np.isfinite(y)):
            self.W[:] = 0.0
            y = np.zeros(self.n_out)
        e = t - y
        err2 = float(e @ e)

        w = float(np.clip(weight, 0.0, 1.0))
        if w <= 0.0:
            return err2
        denom = self.eps + float(x @ x)
        dW = (self.mu * w / denom) * np.outer(e, x)
        nrm = float(np.linalg.norm(dW))
        if nrm > self.max_update_norm:
            dW *= self.max_update_norm / nrm
        self.W += dW
        if not np.all(np.isfinite(self.W)):
            self.W[:] = 0.0
        return err2

    def state_dict(self) -> dict:
        return {"W": self.W.copy(), "mu": self.mu, "eps": self.eps,
                "pred_clip": self.pred_clip,
                "max_update_norm": self.max_update_norm,
                "x_mu": None if self._x_mu is None else self._x_mu.copy(),
                "x_sd": None if self._x_sd is None else self._x_sd.copy()}

    def fit_batch(self, X, Y, ridge: float = 1e-2) -> "NLMSPredictor":
        """Solve the readout offline with ridge regression.

        Standard reservoir-computing training: collect the feature/target
        pairs from a training stream and solve
            W = argmin ||A W - Y||^2 + lam ||W||^2
        where A = [X, 1].  Online NLMS stays available for streaming use;
        benchmark comparisons use this fit so the readout is trained the
        same way the offline baselines are (on the whole train segment).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or len(X) == 0:
            raise ValueError("fit_batch expects non-empty (n, d) / (n, k) arrays")
        # Standardise columns: decoded neural channels have a much larger
        # scale than the encoded state, and a single ridge penalty would
        # otherwise suppress the informative z channels.
        # No column standardisation: neural decoded channels carry the
        # useful signal at their natural scale, and rescaling the nearly
        # constant slow-scale channels amplified noise (measured worse).
        self._x_mu = None
        self._x_sd = None
        A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        AtA = A.T @ A
        scale = float(np.trace(AtA)) / max(AtA.shape[0], 1)
        lam = max(ridge * scale, 1e-12)
        # Solve the ridge problem through SVD-augmented least squares rather
        # than the normal equations: the decoded channels are strongly
        # collinear, and forming A^T A squared the condition number enough
        # to lose the linear (AR-equivalent) part of the fit.
        d = A.shape[1]
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(d)])
        Y_aug = np.vstack([Y, np.zeros((d, Y.shape[1]))])
        W = np.linalg.lstsq(A_aug, Y_aug, rcond=None)[0].T
        if not np.all(np.isfinite(W)):
            raise FloatingPointError("non-finite readout weights")
        self.W = W
        return self

    def load_state_dict(self, d: dict) -> None:
        self.W = np.asarray(d["W"], dtype=float).copy()
        self._x_mu = (None if d.get("x_mu") is None
                      else np.asarray(d["x_mu"], dtype=float))
        self._x_sd = (None if d.get("x_sd") is None
                      else np.asarray(d["x_sd"], dtype=float))
        self.mu = float(d.get("mu", self.mu))
        self.eps = float(d.get("eps", self.eps))
        self.pred_clip = float(d.get("pred_clip", self.pred_clip))
        self.max_update_norm = float(d.get("max_update_norm",
                                          self.max_update_norm))
