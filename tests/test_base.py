"""
Tests for attopy.propagators.base
"""
import numpy as np
import pytest
from attopy.propagators.base import PropagatorBase, PropagatorResult


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing the base class
# ---------------------------------------------------------------------------

class DummyPropagator(PropagatorBase):
    """Identity propagator — returns psi unchanged at every step.
    Used only for testing the base class interface."""

    def _advance_step(self, psi, t_start, t_end, V, pulse):
        return psi.copy()


# ---------------------------------------------------------------------------
# Grid constants and fixtures
# ---------------------------------------------------------------------------

N = 64
DX = 0.1
DT = 0.05   # tgrid spacing must match this


@pytest.fixture
def xgrid():
    return np.linspace(-N // 2 * DX, N // 2 * DX, N, endpoint=False)


@pytest.fixture
def simple_psi():
    psi = np.zeros(N, dtype=complex)
    psi[N // 2] = 1.0 / np.sqrt(DX)   # normalized: dx * (1/dx) = 1.0
    return psi


@pytest.fixture
def tgrid():
    """tgrid with spacing exactly DT, as required by the base class loop."""
    n_steps = 20
    return np.linspace(0.0, n_steps * DT, n_steps + 1)


@pytest.fixture
def dummy(xgrid):
    return DummyPropagator(dx=DX, dt=DT, xgrid=xgrid)


# ---------------------------------------------------------------------------
# PropagatorBase instantiation
# ---------------------------------------------------------------------------

class TestPropagatorBaseInstantiation:
    def test_cannot_instantiate_directly(self, xgrid):
        with pytest.raises(TypeError):
            PropagatorBase(dx=0.1, dt=DT, xgrid=xgrid)

    def test_dummy_instantiates(self, dummy):
        assert dummy.dx == DX
        assert dummy.dt == DT

    def test_xgrid_stored(self, dummy, xgrid):
        assert np.array_equal(dummy.xgrid, xgrid)

    def test_store_psi_default_true(self, dummy):
        assert dummy.store_psi is True

    def test_store_psi_false(self, xgrid):
        prop = DummyPropagator(dx=DX, dt=DT, xgrid=xgrid, store_psi=False)
        assert prop.store_psi is False

    def test_repr(self, dummy):
        assert "DummyPropagator" in repr(dummy)
        assert "dx=" in repr(dummy)

    def test_negative_dx_raises(self, xgrid):
        with pytest.raises(ValueError, match="dx must be positive"):
            DummyPropagator(dx=-0.1, dt=DT, xgrid=xgrid)

    def test_negative_dt_raises(self, xgrid):
        with pytest.raises(ValueError, match="dt must be positive"):
            DummyPropagator(dx=DX, dt=-0.05, xgrid=xgrid)

    def test_zero_dx_raises(self, xgrid):
        with pytest.raises(ValueError):
            DummyPropagator(dx=0.0, dt=DT, xgrid=xgrid)

    def test_zero_dt_raises(self, xgrid):
        with pytest.raises(ValueError):
            DummyPropagator(dx=DX, dt=0.0, xgrid=xgrid)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_2d_psi_raises(self, dummy, tgrid):
        psi_2d = np.ones((4, 4), dtype=complex)
        with pytest.raises(ValueError, match="1D"):
            dummy.propagate(psi_2d, tgrid, None, lambda t: 0.0)

    def test_non_monotonic_tgrid_raises(self, dummy, simple_psi):
        bad_tgrid = np.array([0.0, DT, DT * 0.5, DT * 2])
        with pytest.raises(ValueError, match="monotonically increasing"):
            dummy.propagate(simple_psi, bad_tgrid, None, lambda t: 0.0)

    def test_short_tgrid_raises(self, dummy, simple_psi):
        with pytest.raises(ValueError):
            dummy.propagate(simple_psi, np.array([0.0]), None, lambda t: 0.0)

    def test_unnormalized_psi_emits_warning(self, dummy, tgrid):
        psi_unnorm = np.ones(N, dtype=complex) * 5.0
        with pytest.warns(UserWarning, match="norm"):
            result = dummy.propagate(psi_unnorm, tgrid, None, lambda t: 0.0)
        assert np.isclose(result.norm[0], 1.0, atol=1e-6)

    def test_real_psi_converted_to_complex(self, dummy, tgrid):
        psi_real = np.zeros(N, dtype=float)
        psi_real[N // 2] = 1.0 / np.sqrt(DX)   # normalized
        result = dummy.propagate(psi_real, tgrid, None, lambda t: 0.0)
        assert result.psi.dtype == np.complex128


# ---------------------------------------------------------------------------
# PropagatorResult — construction and shape checks
# ---------------------------------------------------------------------------

class TestPropagatorResult:
    def test_result_shapes(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        Nt = len(tgrid)
        assert result.t.shape == (Nt,)
        assert result.psi.shape == (Nt, N)
        assert result.norm.shape == (Nt,)
        assert result.dipole.shape == (Nt,)

    def test_psi_none_when_store_psi_false(self, xgrid, simple_psi, tgrid):
        prop = DummyPropagator(dx=DX, dt=DT, xgrid=xgrid, store_psi=False)
        result = prop.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.psi is None

    def test_result_info_keys(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        for key in ("backend", "dx", "dt", "n_steps", "wall_time_s", "final_psi"):
            assert key in result.info

    def test_result_info_backend_name(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.info["backend"] == "DummyPropagator"

    def test_mismatched_psi_shape_raises(self):
        t = np.linspace(0, 1, 10)
        psi = np.ones((5, N), dtype=complex)   # wrong: shape[0] should be 10
        norm = np.ones(10)
        dipole = np.zeros(10)
        with pytest.raises(ValueError, match="psi.shape"):
            PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)

    def test_psi_none_skips_shape_check(self):
        """psi=None should not raise even though shape can't be checked."""
        t = np.linspace(0, 1, 10)
        norm = np.ones(10)
        dipole = np.zeros(10)
        result = PropagatorResult(t=t, psi=None, norm=norm, dipole=dipole)
        assert result.psi is None

    def test_mismatched_norm_shape_raises(self):
        t = np.linspace(0, 1, 10)
        psi = np.ones((10, N), dtype=complex)
        norm = np.ones(5)    # wrong length
        dipole = np.zeros(10)
        with pytest.raises(ValueError, match="norm.shape"):
            PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)

    def test_dipole_acceleration_none_by_default(self, dummy, simple_psi, tgrid):
        """dipole_acceleration is None until Phase 4 populates it via Ehrenfest."""
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.dipole_acceleration is None

    def test_dipole_acceleration_shape_check_when_provided(self):
        """If dipole_acceleration is provided, its shape must match t."""
        t = np.linspace(0, 1, 10)
        norm = np.ones(10)
        dipole = np.zeros(10)
        bad_accel = np.zeros(5)   # wrong length
        with pytest.raises(ValueError, match="dipole_acceleration.shape"):
            PropagatorResult(
                t=t, psi=None, norm=norm, dipole=dipole,
                dipole_acceleration=bad_accel,
            )


# ---------------------------------------------------------------------------
# PropagatorResult — properties
# ---------------------------------------------------------------------------

class TestPropagatorResultProperties:
    def test_ionization_yield_zero_for_unit_norm(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert np.allclose(result.ionization_yield, 0.0)

    def test_ionization_yield_formula(self):
        t = np.linspace(0, 1, 10)
        norm = np.linspace(1.0, 0.8, 10)
        dipole = np.zeros(10)
        result = PropagatorResult(t=t, psi=None, norm=norm, dipole=dipole)
        assert np.allclose(result.ionization_yield, 1.0 - norm)

    def test_final_psi_from_psi_history(self, dummy, simple_psi, tgrid):
        """When store_psi=True, final_psi comes from psi[-1]."""
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.psi is not None
        assert np.array_equal(result.final_psi, result.psi[-1])

    def test_final_psi_from_info_when_store_psi_false(self, xgrid, simple_psi, tgrid):
        """When store_psi=False, final_psi comes from info['final_psi']."""
        prop = DummyPropagator(dx=DX, dt=DT, xgrid=xgrid, store_psi=False)
        result = prop.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.psi is None
        assert result.final_psi is not None
        assert result.final_psi.shape == (N,)

    def test_final_psi_shape(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.final_psi.shape == simple_psi.shape

    def test_final_psi_no_data_raises(self):
        """final_psi with psi=None and no info['final_psi'] should raise RuntimeError."""
        t = np.linspace(0, 1, 10)
        norm = np.ones(10)
        dipole = np.zeros(10)
        result = PropagatorResult(t=t, psi=None, norm=norm, dipole=dipole)
        with pytest.raises(RuntimeError):
            _ = result.final_psi


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TestCallback:
    def test_callback_called_at_every_step(self, dummy, simple_psi, tgrid):
        call_count = []
        dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0,
                        callback=lambda t, psi: call_count.append(t))
        assert len(call_count) == len(tgrid)

    def test_callback_receives_correct_times(self, dummy, simple_psi, tgrid):
        recorded = []
        dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0,
                        callback=lambda t, psi: recorded.append(t))
        assert np.allclose(recorded, tgrid, atol=1e-12)