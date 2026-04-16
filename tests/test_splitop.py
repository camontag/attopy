"""
Tests for attopy.propagators.splitop

Grid parameters
---------------
Each physics test class uses its own grid sized for the problem.
Critically, tgrid is always constructed so that tgrid[1]-tgrid[0] == dt,
since the propagator takes exactly one step of size dt per tgrid interval.
"""

from __future__ import annotations

import numpy as np
import pytest
from attopy.propagators.splitop import (
    SplitOperatorPropagator,
    _apply_T,
    _apply_V_full,
    _apply_V_half,
    _check_tgrid_dt_consistency,
    _check_uniform_grid,
    _compute_dipole,
    _compute_norm,
)


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

# Generic grid for unit/constructor tests
N_UNIT = 128
DX_UNIT = 0.2
DT_UNIT = 0.05
XGRID_UNIT = np.linspace(
    -N_UNIT // 2 * DX_UNIT,
     N_UNIT // 2 * DX_UNIT,
     N_UNIT, endpoint=False
)

# Harmonic oscillator grid — fine, small box matched to HO length scale
# Box: 12.8 au, dx=0.05 au. Ground state negligible beyond |x|~5 au.
N_HO = 256
DX_HO = 0.05
DT_HO = 0.01
XGRID_HO = np.linspace(
    -N_HO // 2 * DX_HO,
     N_HO // 2 * DX_HO,
     N_HO, endpoint=False
)
V_HO = 0.5 * XGRID_HO ** 2

# Free particle grid — large box so wavepacket doesn't reach boundary
# Box: 102.4 au, wavepacket starts at x=0, travels ~3 au
N_FP = 512
DX_FP = 0.2
DT_FP = 0.05
XGRID_FP = np.linspace(
    -N_FP // 2 * DX_FP,
     N_FP // 2 * DX_FP,
     N_FP, endpoint=False
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ground_state(xgrid: np.ndarray, dx: float, omega: float = 1.0) -> np.ndarray:
    psi = (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * xgrid ** 2)
    psi = psi.astype(complex)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def _make_coherent_state(
    xgrid: np.ndarray, dx: float, x0: float, omega: float = 1.0
) -> np.ndarray:
    psi = (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * (xgrid - x0) ** 2)
    psi = psi.astype(complex)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def _make_tgrid(T: float, dt: float) -> np.ndarray:
    """Build a tgrid with spacing exactly dt covering [0, T]."""
    n_steps = int(round(T / dt))
    return np.linspace(0.0, n_steps * dt, n_steps + 1)


# ---------------------------------------------------------------------------
# 1. Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestCheckUniformGrid:
    def test_uniform_grid_passes(self):
        _check_uniform_grid(XGRID_UNIT, DX_UNIT)

    def test_non_uniform_grid_raises(self):
        bad_grid = XGRID_UNIT.copy()
        bad_grid[N_UNIT // 2] += 0.05
        with pytest.raises(ValueError, match="uniformly spaced"):
            _check_uniform_grid(bad_grid, DX_UNIT)

    def test_wrong_dx_raises(self):
        with pytest.raises(ValueError, match="uniformly spaced"):
            _check_uniform_grid(XGRID_UNIT, DX_UNIT * 2)


class TestCheckTgridDtConsistency:
    def test_consistent_passes(self):
        tgrid = _make_tgrid(1.0, DT_UNIT)
        _check_tgrid_dt_consistency(tgrid, DT_UNIT)  # should not raise

    def test_inconsistent_raises(self):
        tgrid = np.linspace(0, 1.0, 200)   # spacing ≠ DT_UNIT
        with pytest.raises(ValueError, match="tgrid spacing"):
            _check_tgrid_dt_consistency(tgrid, DT_UNIT)

    def test_single_point_tgrid_passes(self):
        """Edge case: single-point tgrid has no spacing to check."""
        _check_tgrid_dt_consistency(np.array([0.0]), DT_UNIT)


class TestComputeNorm:
    def test_normalised_state_has_unit_norm(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        assert np.isclose(_compute_norm(psi, DX_HO), 1.0, atol=1e-8)

    def test_norm_scales_quadratically_with_amplitude(self):
        psi = np.ones(N_UNIT, dtype=complex)
        norm_1 = _compute_norm(psi, DX_UNIT)
        norm_2 = _compute_norm(2.0 * psi, DX_UNIT)
        assert np.isclose(norm_2 / norm_1, 4.0, rtol=1e-10)

    def test_norm_returns_float(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        assert isinstance(_compute_norm(psi, DX_HO), float)


class TestComputeDipole:
    def test_symmetric_state_has_zero_dipole(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        assert np.isclose(_compute_dipole(psi, XGRID_HO, DX_HO), 0.0, atol=1e-10)

    def test_displaced_state_dipole(self):
        x0 = 2.0
        psi = _make_coherent_state(XGRID_HO, DX_HO, x0=x0)
        assert np.isclose(_compute_dipole(psi, XGRID_HO, DX_HO), x0, atol=1e-4)

    def test_dipole_returns_float(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        assert isinstance(_compute_dipole(psi, XGRID_HO, DX_HO), float)


class TestApplyVHalf:
    def test_norm_preserved_real_potential(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        psi_out = _apply_V_half(psi, V_HO.astype(complex), 0.0, XGRID_HO, DT_HO)
        assert np.isclose(_compute_norm(psi_out, DX_HO), 1.0, atol=1e-12)

    def test_norm_decreases_imaginary_potential(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        V_cap = -0.1j * np.ones(N_HO, dtype=complex)
        psi_out = _apply_V_half(psi, V_cap, 0.0, XGRID_HO, DT_HO)
        assert _compute_norm(psi_out, DX_HO) < 1.0

    def test_zero_potential_zero_dt(self):
        psi = np.ones(N_UNIT, dtype=complex) / np.sqrt(N_UNIT * DX_UNIT)
        psi_out = _apply_V_half(psi, np.zeros(N_UNIT, dtype=complex),
                                0.0, XGRID_UNIT, 0.0)
        assert np.allclose(psi_out, psi)

    def test_applies_correct_phase(self):
        psi = np.ones(N_UNIT, dtype=complex)
        V = np.ones(N_UNIT, dtype=complex) * 2.0
        dt = 0.1
        psi_out = _apply_V_half(psi, V, 0.0, XGRID_UNIT, dt)
        expected_phase = np.exp(-0.5j * 2.0 * dt)
        assert np.allclose(psi_out / psi, expected_phase)


class TestApplyVFull:
    def test_norm_preserved_real_potential(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        psi_out = _apply_V_full(psi, V_HO.astype(complex), 0.0, XGRID_HO, DT_HO)
        assert np.isclose(_compute_norm(psi_out, DX_HO), 1.0, atol=1e-12)

    def test_full_step_equals_two_half_steps(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        V = V_HO.astype(complex)
        E = 0.5
        psi_half_twice = _apply_V_half(
            _apply_V_half(psi, V, E, XGRID_HO, DT_HO),
            V, E, XGRID_HO, DT_HO,
        )
        psi_full = _apply_V_full(psi, V, E, XGRID_HO, DT_HO)
        assert np.allclose(psi_full, psi_half_twice, atol=1e-12)


class TestApplyT:
    def test_norm_preserved(self):
        psi = _make_ground_state(XGRID_HO, DX_HO)
        k = 2.0 * np.pi * np.fft.fftfreq(N_HO, d=DX_HO)
        kinetic_phase = np.exp(-0.5j * k**2 * DT_HO)
        psi_out = _apply_T(psi, kinetic_phase)
        assert np.isclose(_compute_norm(psi_out, DX_HO), 1.0, atol=1e-10)

    def test_plane_wave_acquires_correct_phase(self):
        """Plane wave at an exact FFT frequency acquires phase exp(-ik₀²dt/2)."""
        k_grid = 2.0 * np.pi * np.fft.fftfreq(N_HO, d=DX_HO)
        k0 = k_grid[1]   # second bin — exactly on grid, no spectral leakage

        psi_k0 = np.exp(1j * k0 * XGRID_HO)
        psi_k0 /= np.sqrt(np.sum(np.abs(psi_k0) ** 2) * DX_HO)

        kinetic_phase = np.exp(-0.5j * k_grid**2 * DT_HO)
        psi_out = _apply_T(psi_k0, kinetic_phase)

        expected_phase = np.exp(-0.5j * k0**2 * DT_HO)
        ratio = psi_out / psi_k0
        assert np.allclose(ratio, expected_phase, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Constructor and input validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_instantiates(self):
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        assert prop.dx == DX_HO
        assert prop.dt == DT_HO

    def test_kinetic_phase_shape(self):
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        assert prop._kinetic_phase.shape == (N_HO,)

    def test_kinetic_phase_is_unitary(self):
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        assert np.allclose(np.abs(prop._kinetic_phase), 1.0)

    def test_non_uniform_xgrid_raises(self):
        bad_xgrid = XGRID_HO.copy()
        bad_xgrid[N_HO // 2] += 0.05
        with pytest.raises(ValueError, match="uniformly spaced"):
            SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=bad_xgrid)

    def test_V_shape_mismatch_raises(self):
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        bad_V = np.zeros(N_HO + 10)
        tgrid = _make_tgrid(1.0, DT_HO)
        with pytest.raises(ValueError, match="V.shape"):
            prop.propagate(psi0, tgrid, bad_V, lambda t: 0.0)

    def test_inconsistent_tgrid_dt_raises(self):
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        bad_tgrid = np.linspace(0, 1.0, 200)   # spacing ≠ DT_HO
        with pytest.raises(ValueError, match="tgrid spacing"):
            prop.propagate(psi0, bad_tgrid, V_HO.astype(complex), lambda t: 0.0)


# ---------------------------------------------------------------------------
# 3. Physics validation tests
# ---------------------------------------------------------------------------

class TestFreeParticlePropagation:
    """Benchmark 1: Gaussian wavepacket in zero potential."""

    def _make_wavepacket(self, x0=0.0, k0=1.0, sigma=2.0):
        psi = np.exp(-((XGRID_FP - x0) ** 2) / (4 * sigma**2) + 1j * k0 * XGRID_FP)
        psi = psi.astype(complex)
        return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * DX_FP)

    def test_norm_conservation(self):
        psi0 = self._make_wavepacket()
        prop = SplitOperatorPropagator(dx=DX_FP, dt=DT_FP, xgrid=XGRID_FP)
        tgrid = _make_tgrid(5.0, DT_FP)
        result = prop.propagate(psi0, tgrid, np.zeros(N_FP, dtype=complex),
                                pulse=lambda t: 0.0)
        assert np.allclose(result.norm, 1.0, atol=1e-6)

    def test_group_velocity(self):
        """Centre of wavepacket moves at velocity k0."""
        x0, k0, sigma = 0.0, 1.0, 2.0
        psi0 = self._make_wavepacket(x0=x0, k0=k0, sigma=sigma)
        prop = SplitOperatorPropagator(dx=DX_FP, dt=DT_FP, xgrid=XGRID_FP)
        T_final = 3.0
        tgrid = _make_tgrid(T_final, DT_FP)
        result = prop.propagate(psi0, tgrid, np.zeros(N_FP, dtype=complex),
                                pulse=lambda t: 0.0)
        x_expected = x0 + k0 * T_final
        assert np.isclose(result.dipole[-1], x_expected, atol=0.1)


class TestHarmonicOscillatorGroundState:
    """Benchmark 2: Ground state of harmonic oscillator, no field."""

    def test_norm_conservation(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(10.0, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        assert np.allclose(result.norm, 1.0, atol=1e-8)

    def test_dipole_is_zero(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(10.0, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        assert np.allclose(result.dipole, 0.0, atol=1e-8)

    def test_density_is_stationary(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(10.0, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        density_0 = np.abs(result.psi[0]) ** 2
        density_f = np.abs(result.psi[-1]) ** 2
        assert np.allclose(density_0, density_f, atol=1e-6)

    def test_ground_state_energy(self):
        """After one period T=2π/E₀=4π, wavefunction returns to itself."""
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        E0 = 0.5
        T_period = 2.0 * np.pi / E0
        tgrid = _make_tgrid(T_period, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        overlap = np.abs(np.sum(np.conj(psi0) * result.psi[-1]) * DX_HO)
        assert np.isclose(overlap, 1.0, atol=1e-3)


class TestCoherentState:
    """Benchmark 3: Coherent state — <x>(t) = x0·cos(ωt) exactly."""

    def test_dipole_oscillation(self):
        omega = 1.0
        x0 = 2.0
        psi0 = _make_coherent_state(XGRID_HO, DX_HO, x0=x0, omega=omega)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        T = 2.0 * np.pi / omega
        tgrid = _make_tgrid(T, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        x_analytic = x0 * np.cos(omega * tgrid)
        assert np.allclose(result.dipole, x_analytic, atol=5e-4)

    def test_norm_conserved_during_oscillation(self):
        psi0 = _make_coherent_state(XGRID_HO, DX_HO, x0=2.0)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(2.0 * np.pi, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex),
                                pulse=lambda t: 0.0)
        assert np.allclose(result.norm, 1.0, atol=1e-8)


class TestTimeReversal:
    """Benchmark 4: Forward + backward propagation recovers initial state."""

    def test_time_reversal_ground_state(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        tgrid = _make_tgrid(5.0, DT_HO)

        prop_fwd = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        result_fwd = prop_fwd.propagate(psi0, tgrid, V_HO.astype(complex),
                                        pulse=lambda t: 0.0)
        psi_forward = result_fwd.final_psi

        prop_rev = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        prop_rev._kinetic_phase = np.conj(prop_fwd._kinetic_phase)
        result_rev = prop_rev.propagate(psi_forward, tgrid, -V_HO.astype(complex),
                                        pulse=lambda t: 0.0)

        overlap = np.abs(np.sum(np.conj(psi0) * result_rev.final_psi) * DX_HO)
        assert np.isclose(overlap, 1.0, atol=1e-3)


class TestConvergence:
    """Benchmark 5: Global error scales as O(dt²)."""

    def test_second_order_convergence(self):
        omega = 1.0
        x0 = 2.0
        psi0 = _make_coherent_state(XGRID_HO, DX_HO, x0=x0, omega=omega)

        T = 2.0
        dt_coarse = 0.04
        dt_fine = 0.02

        tgrid_coarse = _make_tgrid(T, dt_coarse)
        tgrid_fine = _make_tgrid(T, dt_fine)

        prop_coarse = SplitOperatorPropagator(dx=DX_HO, dt=dt_coarse, xgrid=XGRID_HO)
        prop_fine = SplitOperatorPropagator(dx=DX_HO, dt=dt_fine, xgrid=XGRID_HO)

        result_coarse = prop_coarse.propagate(
            psi0, tgrid_coarse, V_HO.astype(complex), pulse=lambda t: 0.0
        )
        result_fine = prop_fine.propagate(
            psi0, tgrid_fine, V_HO.astype(complex), pulse=lambda t: 0.0
        )

        # Evaluate error at the shared endpoint T
        x_analytic_T = x0 * np.cos(omega * T)
        err_coarse = abs(result_coarse.dipole[-1] - x_analytic_T)
        err_fine = abs(result_fine.dipole[-1] - x_analytic_T)

        ratio = err_coarse / err_fine
        assert ratio > 3.0, f"Expected convergence ratio > 3, got {ratio:.2f}"


class TestStorePsi:
    def test_store_psi_true_saves_all_steps(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO,
                                       store_psi=True)
        tgrid = _make_tgrid(5.0, DT_HO)
        result = prop.propagate(psi0, tgrid, V_HO.astype(complex), lambda t: 0.0)
        assert result.psi.shape == (len(tgrid), N_HO)
        assert not np.allclose(result.psi[0], 0.0)

    def test_store_psi_false_final_psi_matches(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        tgrid = _make_tgrid(5.0, DT_HO)

        prop_full = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO,
                                            store_psi=True)
        prop_lean = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO,
                                            store_psi=False)

        result_full = prop_full.propagate(psi0, tgrid, V_HO.astype(complex),
                                          lambda t: 0.0)
        result_lean = prop_lean.propagate(psi0, tgrid, V_HO.astype(complex),
                                          lambda t: 0.0)

        assert np.allclose(result_full.final_psi, result_lean.final_psi, atol=1e-12)

    def test_store_psi_false_observables_identical(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        tgrid = _make_tgrid(5.0, DT_HO)

        prop_full = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO,
                                            store_psi=True)
        prop_lean = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO,
                                            store_psi=False)

        result_full = prop_full.propagate(psi0, tgrid, V_HO.astype(complex),
                                          lambda t: 0.0)
        result_lean = prop_lean.propagate(psi0, tgrid, V_HO.astype(complex),
                                          lambda t: 0.0)

        assert np.allclose(result_full.norm, result_lean.norm, atol=1e-12)
        assert np.allclose(result_full.dipole, result_lean.dipole, atol=1e-12)


class TestCallback:
    def test_callback_called_at_every_step(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(5.0, DT_HO)
        call_count = []

        prop.propagate(psi0, tgrid, V_HO.astype(complex), lambda t: 0.0,
                       callback=lambda t, psi: call_count.append(t))
        assert len(call_count) == len(tgrid)

    def test_callback_receives_correct_times(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(5.0, DT_HO)
        recorded_times = []

        prop.propagate(psi0, tgrid, V_HO.astype(complex), lambda t: 0.0,
                       callback=lambda t, psi: recorded_times.append(t))
        assert np.allclose(recorded_times, tgrid, atol=1e-12)

    def test_callback_psi_has_unit_norm(self):
        psi0 = _make_ground_state(XGRID_HO, DX_HO)
        prop = SplitOperatorPropagator(dx=DX_HO, dt=DT_HO, xgrid=XGRID_HO)
        tgrid = _make_tgrid(5.0, DT_HO)
        norms = []

        prop.propagate(psi0, tgrid, V_HO.astype(complex), lambda t: 0.0,
                       callback=lambda t, psi: norms.append(_compute_norm(psi, DX_HO)))
        assert np.allclose(norms, 1.0, atol=1e-8)