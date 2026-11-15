"""Phase II unit tests: system adapter, orthogonal projector, weave smoke.

Run:  python tests/test_phase2.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.system import SystemSource
from core.encoding.projector import SubspaceProjector


def test_system_source_schema():
    src = SystemSource()
    src.advance_to(0.005)
    pending = src.read_pending()
    assert len(pending) == 1
    t, vec = pending[0]
    assert vec.shape == (4,)
    assert np.all(np.isfinite(vec))
    assert 0.0 <= vec[0] <= 1.0   # cpu util
    assert 0.0 <= vec[1] <= 1.0   # mem percent
    src.advance_to(0.010)
    assert len(src.read_pending()) == 1


def test_projector_orthogonality_and_demux():
    proj = SubspaceProjector({"lorenz": 3, "system": 4}, proj_dim=16,
                             mode="orthogonal", seed=0)
    Pl, Ps = proj.matrices["lorenz"], proj.matrices["system"]
    # column spaces mutually orthogonal
    cross = np.abs(Pl.T @ Ps).max()
    assert cross < 1e-10, f"subspaces not orthogonal: {cross}"
    # within-source columns orthonormal
    assert np.allclose(Pl.T @ Pl, np.eye(3), atol=1e-8)
    assert np.allclose(Ps.T @ Ps, np.eye(4), atol=1e-8)

    rng = np.random.default_rng(1)
    z_l, z_s = rng.standard_normal(3), rng.standard_normal(4)
    v = proj.project({"lorenz": z_l, "system": z_s})
    rec = proj.demux(v)
    assert np.allclose(rec["lorenz"], z_l, atol=1e-8)
    assert np.allclose(rec["system"], z_s, atol=1e-8)

    feats = proj.features({"lorenz": z_l, "system": z_s})
    assert feats.shape == (proj.feature_dim,)


def test_projector_interference_mode_is_not_orthogonal():
    proj = SubspaceProjector({"lorenz": 3, "system": 4}, proj_dim=16,
                             mode="interference", seed=0)
    cross = np.abs(proj.matrices["lorenz"].T
                   @ proj.matrices["system"]).max()
    assert cross > 1e-3, "interference mode unexpectedly orthogonal"


def test_weave_smoke():
    """Small multi-source weave run must stay finite."""
    from core.dynamics.weave import WeaveEngine
    weave = WeaveEngine(feature_dim=32, n_neurons=120, dt=0.005,
                        recurrent_gain=0.3, seed=0)
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((40, 32)) * 0.5
    res = weave.run_chunk(feats)
    assert res["decoded"].shape == (40, 32)
    assert np.all(np.isfinite(res["decoded"]))
    assert np.all(np.isfinite(res["rate_mean"]))
    assert float(np.mean(res["rate_mean"])) < 1000.0


if __name__ == "__main__":
    test_system_source_schema()
    print("ok: system source")
    test_projector_orthogonality_and_demux()
    print("ok: orthogonal projector")
    test_projector_interference_mode_is_not_orthogonal()
    print("ok: interference mode")
    test_weave_smoke()
    print("ok: weave smoke")
    print("ALL PHASE II UNIT TESTS PASSED")
