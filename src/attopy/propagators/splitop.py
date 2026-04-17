"""
attopy.propagators.splitop
==========================
Split-operator FFT propagator for the 1D TDSE.

The Hamiltonian is split as H = T + V_eff(x, t), where:
    T        = -½ d²/dx²              (kinetic energy, diagonal in k-space)
    V_eff    = V(x) + E(t)·x          (potential + laser, diagonal in x-space)

Time evolution over one step Δt uses the symmetric Strang splitting:

    exp(-iH·Δt) ≈ exp(-iV·Δt/2) · exp(-iT·Δt) · exp(-iV·Δt/2) + O(Δt³)

This backend implements _advance_step(), which applies one full V/T/V
sequence per call. The base class PropagatorBase owns the time-stepping
loop and observable recording.

Note: the merged V-step optimization (combining consecutive half-V steps
into a single full-V step) is not implemented here because it requires
cross-step state that conflicts with the _advance_step() interface. The
performance difference is negligible for typical strong-field grid sizes
since potential evaluations are cheap relative to FFT pairs.

Requires a uniformly spaced position grid. Raises ValueError if xgrid is
non-uniform, rather than silently producing incorrect results.

Notes
-----
All quantities are in atomic units unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

from attopy.propagators.base import PropagatorBase, PropagatorResult


class SplitOperatorPropagator(PropagatorBase):
    """Split-operator FFT propagator for the 1D TDSE.

    Implements the symmetric Strang splitting: V/2 → T → V/2 per step.
    The time-stepping loop and observable recording are handled by the
    base class PropagatorBase.

    Parameters
    ----------
    dx : float
        Spatial grid spacing in atomic units. Must be uniform.
    dt : float
        Time step in atomic units.
    xgrid : np.ndarray, shape (N,)
        Position grid in atomic units. Must be uniformly spaced
        with spacing dx.
    store_psi : bool, optional
        If True (default), store the full wavefunction at every time step.
        If False, only the final wavefunction is retained, saving memory
        for long propagations. Observables (norm, dipole) are always stored.

    Raises
    ------
    ValueError
        If xgrid is not uniformly spaced with spacing dx.

    Examples
    --------
    >>> import numpy as np
    >>> from attopy.propagators.splitop import SplitOperatorPropagator
    >>> N, dx, dt = 256, 0.05, 0.01
    >>> xgrid = np.linspace(-N // 2 * dx, N // 2 * dx, N, endpoint=False)
    >>> prop = SplitOperatorPropagator(dx=dx, dt=dt, xgrid=xgrid)
    >>> V = 0.5 * xgrid**2                          # harmonic oscillator
    >>> psi0 = (1/np.pi)**0.25 * np.exp(-0.5*xgrid**2).astype(complex)
    >>> psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    >>> n = int(round(2*np.pi / dt))
    >>> tgrid = np.linspace(0, n * dt, n + 1)
    >>> result = prop.propagate(psi0, tgrid, V.astype(complex), pulse=lambda t: 0.0)
    """

    def __init__(
        self,
        dx: float,
        dt: float,
        xgrid: np.ndarray,
        store_psi: bool = True,
    ):
        super().__init__(dx=dx, dt=dt, xgrid=xgrid, store_psi=store_psi)
        _check_uniform_grid(xgrid, dx)

        N = len(xgrid)
        # Momentum grid: k = 2π · fftfreq(N, d=dx)
        self._k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        # Kinetic phase factor — precomputed once at construction time
        # T = k²/2  →  exp(-i·T·dt) = exp(-i·k²·dt/2)
        self._kinetic_phase = np.exp(-0.5j * self._k**2 * dt)

        # Store V between propagate() calls so _advance_step can access it.
        # Set in propagate() before the loop starts.
        self._V: np.ndarray | None = None
        self._pulse: callable | None = None

    def propagate(
        self,
        psi0: np.ndarray,
        tgrid: np.ndarray,
        V: np.ndarray,
        pulse: callable,
        callback: callable | None = None,
    ) -> PropagatorResult:
        """Propagate an initial wavefunction forward in time.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,), complex
            Initial wavefunction. Normalised internally if needed.
        tgrid : np.ndarray, shape (Nt,)
            Time points in atomic units. Must be monotonically increasing
            with spacing equal to dt.
        V : np.ndarray, shape (N,), complex
            Field-free potential in atomic units. Real part is the physical
            potential; imaginary part (if any) is a complex absorbing
            potential (CAP).
        pulse : callable
            Laser pulse with signature pulse(t) -> float, returning the
            electric field amplitude E(t) in atomic units. For zero field,
            pass lambda t: 0.0.
        callback : callable, optional
            Called at each step as callback(t, psi).

        Returns
        -------
        PropagatorResult

        Notes
        -----
        Laser interaction in the length gauge: V_laser(x, t) = E(t) · x.
        Total effective potential: V_eff(x, t) = V(x) + E(t) · x.
        """
        V = np.asarray(V, dtype=np.complex128)
        psi0_validated = self._validate_inputs(psi0, tgrid)

        if V.shape != psi0_validated.shape:
            raise ValueError(
                f"V.shape {V.shape} does not match psi0.shape {psi0_validated.shape}"
            )
        _check_tgrid_dt_consistency(tgrid, self.dt)

        # Store V and pulse so _advance_step can access them
        self._V = V
        self._pulse = pulse

        try:
            result = super().propagate(psi0, tgrid, V, pulse, callback)
        finally:
            # Always clear references to avoid accidental reuse
            self._V = None
            self._pulse = None

        return result

    def _advance_step(
        self,
        psi: np.ndarray,
        t_start: float,
        t_end: float,
        V: np.ndarray,
        pulse: callable,
    ) -> np.ndarray:
        """Advance wavefunction by one step using V/2 → T → V/2 splitting.

        Parameters
        ----------
        psi : np.ndarray, shape (N,), complex
            Wavefunction at t_start.
        t_start : float
            Start of the time step (au).
        t_end : float
            End of the time step (au).
        V : np.ndarray, shape (N,), complex
            Field-free potential.
        pulse : callable
            Laser pulse function.

        Returns
        -------
        np.ndarray, shape (N,), complex
            Wavefunction at t_end.
        """
        dt = self.dt
        xgrid = self.xgrid
        t_mid = 0.5 * (t_start + t_end)
        E_mid = pulse(t_mid)          # evaluate once at midpoint

        psi = _apply_V_half(psi, V, E_mid, xgrid, dt)   # half-V at t_mid
        psi = _apply_T(psi, self._kinetic_phase)          # full T
        psi = _apply_V_half(psi, V, E_mid, xgrid, dt)   # half-V at t_mid again

        return psi


# ---------------------------------------------------------------------------
# Private helper functions — pure NumPy, no Python loops
# ---------------------------------------------------------------------------

def _check_tgrid_dt_consistency(tgrid: np.ndarray, dt: float, rtol: float = 1e-6) -> None:
    """Raise ValueError if tgrid spacing does not match dt.

    The base class loop takes exactly one step of size dt per tgrid interval.
    A mismatch causes the simulation to integrate over the wrong time interval.
    """
    if len(tgrid) < 2:
        return
    tgrid_dt = tgrid[1] - tgrid[0]
    if not np.isclose(tgrid_dt, dt, rtol=rtol):
        raise ValueError(
            f"tgrid spacing ({tgrid_dt:.6f} au) does not match propagator "
            f"dt ({dt:.6f} au). Either adjust tgrid spacing to match dt, "
            f"or reconstruct the propagator with dt={tgrid_dt:.6f}."
        )


def _check_uniform_grid(xgrid: np.ndarray, dx: float, rtol: float = 1e-6) -> None:
    """Raise ValueError if xgrid is not uniformly spaced with spacing dx."""
    spacings = np.diff(xgrid)
    if not np.allclose(spacings, dx, rtol=rtol):
        raise ValueError(
            "SplitOperatorPropagator requires a uniformly spaced xgrid. "
            f"Expected spacing dx={dx:.6f} au, "
            f"got min={spacings.min():.6f}, max={spacings.max():.6f}."
        )


def _apply_V_half(
    psi: np.ndarray,
    V: np.ndarray,
    E: float,
    xgrid: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply half a potential step: psi *= exp(-i·V_eff·dt/2).

    V_eff(x, t) = V(x) + E(t)·x
    """
    return psi * np.exp(-0.5j * (V + E * xgrid) * dt)


def _apply_V_full(
    psi: np.ndarray,
    V: np.ndarray,
    E: float,
    xgrid: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply a full potential step: psi *= exp(-i·V_eff·dt).

    V_eff(x, t) = V(x) + E(t)·x
    """
    return psi * np.exp(-1.0j * (V + E * xgrid) * dt)


def _apply_T(psi: np.ndarray, kinetic_phase: np.ndarray) -> np.ndarray:
    """Apply the full kinetic step: FFT → kinetic phase → IFFT."""
    return np.fft.ifft(kinetic_phase * np.fft.fft(psi))


def _compute_norm(psi: np.ndarray, dx: float) -> float:
    """Compute ||ψ||² = dx · Σ |ψ(xⱼ)|²"""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def _compute_dipole(psi: np.ndarray, xgrid: np.ndarray, dx: float) -> float:
    """Compute <x> = dx · Σ xⱼ · |ψ(xⱼ)|²"""
    return float(np.sum(xgrid * np.abs(psi) ** 2) * dx)