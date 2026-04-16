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
    """Minimal concrete propagator that returns the initial state unchanged.
    Used only for testing the base class interface."""

    def propagate(self, psi0, tgrid, H0, pulse, callback=None):
        import time
        t0 = time.time()
        psi0 = self._validate_inputs(psi0, tgrid)
        Nt = len(tgrid)

        psi_history = [psi0.copy() for _ in range(Nt)]
        norm_history = [1.0] * Nt
        dipole_history = [0.0] * Nt

        return self._make_result(
            tgrid, psi_history, norm_history, dipole_history,
            wall_time=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 64
DX = 0.1

@pytest.fixture
def xgrid():
    return np.linspace(-N // 2 * DX, N // 2 * DX, N)

@pytest.fixture
def simple_psi():
    psi = np.zeros(N, dtype=complex)
    psi[N // 2] = 1.0
    return psi

@pytest.fixture
def tgrid():
    return np.linspace(0, 10.0, 100)

@pytest.fixture
def dummy(xgrid):
    return DummyPropagator(dx=DX, dt=0.05, xgrid=xgrid)


# ---------------------------------------------------------------------------
# PropagatorBase instantiation
# ---------------------------------------------------------------------------

class TestPropagatorBaseInstantiation:
    def test_cannot_instantiate_directly(self, xgrid):
        with pytest.raises(TypeError):
            PropagatorBase(dx=0.1, dt=0.05, xgrid=xgrid)

    def test_dummy_instantiates(self, dummy):
        assert dummy.dx == DX
        assert dummy.dt == 0.05

    def test_xgrid_stored(self, dummy, xgrid):
        assert np.array_equal(dummy.xgrid, xgrid)

    def test_store_psi_default_true(self, dummy):
        assert dummy.store_psi is True

    def test_store_psi_false(self, xgrid):
        prop = DummyPropagator(dx=DX, dt=0.05, xgrid=xgrid, store_psi=False)
        assert prop.store_psi is False

    def test_repr(self, dummy):
        assert "DummyPropagator" in repr(dummy)
        assert "dx=" in repr(dummy)

    def test_negative_dx_raises(self, xgrid):
        with pytest.raises(ValueError, match="dx must be positive"):
            DummyPropagator(dx=-0.1, dt=0.05, xgrid=xgrid)

    def test_negative_dt_raises(self, xgrid):
        with pytest.raises(ValueError, match="dt must be positive"):
            DummyPropagator(dx=0.1, dt=-0.05, xgrid=xgrid)

    def test_zero_dx_raises(self, xgrid):
        with pytest.raises(ValueError):
            DummyPropagator(dx=0.0, dt=0.05, xgrid=xgrid)

    def test_zero_dt_raises(self, xgrid):
        with pytest.raises(ValueError):
            DummyPropagator(dx=0.1, dt=0.0, xgrid=xgrid)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_2d_psi_raises(self, dummy, tgrid):
        psi_2d = np.ones((4, 4), dtype=complex)
        with pytest.raises(ValueError, match="1D"):
            dummy.propagate(psi_2d, tgrid, None, lambda t: 0.0)

    def test_non_monotonic_tgrid_raises(self, dummy, simple_psi):
        bad_tgrid = np.array([0.0, 1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="monotonically increasing"):
            dummy.propagate(simple_psi, bad_tgrid, None, lambda t: 0.0)

    def test_short_tgrid_raises(self, dummy, simple_psi):
        with pytest.raises(ValueError):
            dummy.propagate(simple_psi, np.array([0.0]), None, lambda t: 0.0)

    def test_unnormalised_psi_is_normalised(self, dummy, tgrid):
        psi_unnorm = np.ones(N, dtype=complex) * 5.0
        result = dummy.propagate(psi_unnorm, tgrid, None, lambda t: 0.0)
        assert np.isclose(result.norm[0], 1.0, atol=1e-6)

    def test_real_psi_converted_to_complex(self, dummy, tgrid):
        psi_real = np.zeros(N, dtype=float)
        psi_real[N // 2] = 1.0
        result = dummy.propagate(psi_real, tgrid, None, lambda t: 0.0)
        assert result.psi.dtype == np.complex128


# ---------------------------------------------------------------------------
# PropagatorResult
# ---------------------------------------------------------------------------

class TestPropagatorResult:
    def test_result_shapes(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        Nt = len(tgrid)
        assert result.t.shape == (Nt,)
        assert result.psi.shape == (Nt, N)
        assert result.norm.shape == (Nt,)
        assert result.dipole.shape == (Nt,)

    def test_result_info_keys(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        for key in ("backend", "dx", "dt", "n_steps", "wall_time_s"):
            assert key in result.info

    def test_result_info_backend_name(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.info["backend"] == "DummyPropagator"

    def test_mismatched_psi_shape_raises(self):
        t = np.linspace(0, 1, 10)
        psi = np.ones((5, N), dtype=complex)   # wrong: should be (10, N)
        norm = np.ones(10)
        dipole = np.zeros(10)
        with pytest.raises(ValueError, match="psi.shape"):
            PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)

    def test_mismatched_norm_shape_raises(self):
        t = np.linspace(0, 1, 10)
        psi = np.ones((10, N), dtype=complex)
        norm = np.ones(5)    # wrong length
        dipole = np.zeros(10)
        with pytest.raises(ValueError, match="norm.shape"):
            PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)

    def test_ionization_yield_zero_for_unit_norm(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert np.allclose(result.ionization_yield, 0.0)

    def test_ionization_yield_formula(self):
        """ionization_yield should equal 1 - norm at every step."""
        t = np.linspace(0, 1, 10)
        psi = np.ones((10, N), dtype=complex)
        norm = np.linspace(1.0, 0.8, 10)   # decaying norm simulating ionization
        dipole = np.zeros(10)
        result = PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)
        assert np.allclose(result.ionization_yield, 1.0 - norm)

    def test_final_psi_shape(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.final_psi.shape == simple_psi.shape

    def test_final_psi_is_last_step(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert np.array_equal(result.final_psi, result.psi[-1])

    def test_dipole_acceleration_shape(self, dummy, simple_psi, tgrid):
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert result.dipole_acceleration.shape == (len(tgrid),)

    def test_dipole_acceleration_zero_for_zero_dipole(self, dummy, simple_psi, tgrid):
        """Zero dipole should give zero acceleration."""
        result = dummy.propagate(simple_psi, tgrid, None, lambda t: 0.0)
        assert np.allclose(result.dipole_acceleration, 0.0, atol=1e-10)

    def test_dipole_acceleration_uses_tgrid(self):
        """dipole_acceleration should use self.t, not assume uniform spacing."""
        t = np.linspace(0, 2 * np.pi, 500)
        psi = np.ones((500, N), dtype=complex)
        norm = np.ones(500)
        # dipole = sin(t) → acceleration = -sin(t)
        dipole = np.sin(t)
        result = PropagatorResult(t=t, psi=psi, norm=norm, dipole=dipole)
        accel = result.dipole_acceleration
        # Check interior points only (edges have lower-order accuracy)
        assert np.allclose(accel[10:-10], -np.sin(t[10:-10]), atol=1e-4)