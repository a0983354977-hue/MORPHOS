"""Orthogonal subspace projection for heterogeneous sources (T2.2).

Each source's z-scored state is projected into a shared proj_dim space
through a matrix whose column space is mutually orthogonal across
sources.  The summed projection v = sum_s P_s z_s is therefore
losslessly separable: P_s^T v = z_s (up to the per-source z-score
normalisation).  Neural features are [v, tanh(v)].

mode='interference' (ablation): dense random matrices that are NOT
orthogonal across sources -> cross-source semantic interference.
"""

from __future__ import annotations

import numpy as np


class SubspaceProjector:
    def __init__(self, sources: dict[str, int], proj_dim: int = 16,
                 mode: str = "orthogonal", seed: int = 0):
        assert mode in ("orthogonal", "interference")
        self.sources = dict(sources)          # name -> state_dim
        self.proj_dim = proj_dim
        self.mode = mode
        self.order = list(sources.keys())
        total = sum(sources.values())
        if mode == "orthogonal":
            assert total <= proj_dim, \
                "orthogonal mode needs proj_dim >= sum(state_dims)"
        rng = np.random.default_rng(seed)

        if mode == "orthogonal":
            # Q: proj_dim x total_dim with orthonormal columns.
            # Disjoint column blocks => mutually orthogonal subspaces.
            q, _ = np.linalg.qr(rng.standard_normal((proj_dim, total)))
            mats = {}
            off = 0
            for name in self.order:
                d = sources[name]
                mats[name] = q[:, off:off + d]
                off += d
            self.matrices = mats
        else:
            # dense random projections scaled to unit-ish gain; column
            # spaces overlap -> sources interfere inside v
            scale = 1.0 / np.sqrt(proj_dim)
            self.matrices = {
                name: rng.standard_normal((proj_dim, d)) * scale
                for name, d in sources.items()
            }

        self.feature_dim = 2 * proj_dim

    def project(self, zs: dict[str, np.ndarray]) -> np.ndarray:
        v = np.zeros(self.proj_dim)
        for name, z in zs.items():
            v = v + self.matrices[name] @ np.asarray(z, dtype=float)
        return v

    def features(self, zs: dict[str, np.ndarray]) -> np.ndarray:
        v = self.project(zs)
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([v, np.tanh(v)])

    def demux(self, v: np.ndarray) -> dict[str, np.ndarray]:
        """Recover per-source z estimates (exact in orthogonal mode)."""
        out = {}
        for name in self.order:
            p = self.matrices[name]
            # orthogonal mode: P^T P = I -> exact recovery
            out[name] = p.T @ np.asarray(v, dtype=float)
        return out

    def state_dict(self) -> dict:
        return {"sources": self.sources, "proj_dim": self.proj_dim,
                "mode": self.mode,
                "matrices": {k: m.copy() for k, m in
                             self.matrices.items()}}

    def load_state_dict(self, d: dict) -> None:
        self.sources = dict(d["sources"])
        self.proj_dim = int(d["proj_dim"])
        self.mode = d["mode"]
        self.order = list(self.sources.keys())
        self.matrices = {k: np.asarray(v, dtype=float)
                         for k, v in d["matrices"].items()}
        self.feature_dim = 2 * self.proj_dim
