"""
attopy.propagators.splitop
==========================
Split-operator FFT propagator for the 1D TDSE.

The Hamiltonian is split as H = T + V_eff(x, t), where:
    T        = -½ d²/dx²              (kinetic energy, diagonal in k-space)
    V_eff    = V(x) + E(t)·x          (potential + laser, diagonal in x-space)

Time evolution over one step Δt uses the symmetric Strang splitting:

    exp(-iH·Δt) ≈ exp(-iV·Δt/2) · exp(-iT·Δt) · exp(-iV·Δt/2) + O(Δt³)

Consecutive half-V steps are merged into full V-steps for efficiency, so
the loop body is: full-V → FFT → kinetic phase → IFFT, with half-V steps
only at the very start and very end.

Requires a uniformly spaced position grid. Raises ValueError if xgrid is
non-uniform, rather than silently producing incorrect results.

Notes
-----
All quantities are in atomic units unless stated otherwise.
"""

from __future__ import annotations

import time

import numpy as np

from attopy.propagators.base import PropagatorBase, PropagatorResult


class SplitOperatorPropagator(PropagatorBase):
    """Split-operator FFT propagator for the 1D TDSE.

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
        If True (default), store the full wavefunction at every time step
        in PropagatorResult.psi. If False, only the final wavefunction is
        stored, reducing memory usage for long propagations.

    Raises
    ------
    ValueError
        If xgrid is not uniformly spaced with spacing dx.

    Examples
    --------
    >>> import numpy as np
    >>> from attopy.propagators.splitop import SplitOperatorPropagator
    >>> N, dx = 512, 0.2
    >>> xgrid = np.arange(N) * dx - N * dx / 2
    >>> prop = SplitOperatorPropagator(dx=dx, dt=0.05, xgrid=xgrid)
    >>> psi0 = np.exp(-xgrid**2 / 2, dtype=complex)   # Gaussian
    >>> psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    >>> tgrid = np.linspace(0, 10.0, 200)
    >>> result = prop.propagate(psi0, tgrid, V=np.zeros(N), pulse=lambda t: 0.0)
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
        # Momentum grid (au): k = 2π · fftfreq(N, d=dx)
        self._k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        # Kinetic phase factor — precomputed once, never changes
        # T = k²/2  →  exp(-i·T·dt) = exp(-i·k²·dt/2)
        self._kinetic_phase = np.exp(-0.5j * self._k**2 * dt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            Time points in atomic units at which to record observables
            and (optionally) the wavefunction. Must be monotonically
            increasing and uniformly spaced.
        V : np.ndarray, shape (N,)
            Field-free potential on the position grid, in atomic units.
            Typically a soft-core Coulomb or harmonic oscillator potential.
            May include a complex absorbing potential (CAP) as an imaginary
            part: V_total = V_real + i·V_CAP.
        pulse : callable
            Laser pulse function with signature pulse(t) -> float.
            Returns the electric field amplitude E(t) in atomic units
            at time t. For zero field, pass lambda t: 0.0.
        callback : callable, optional
            Called at each time step as callback(t, psi). Use to compute
            custom observables mid-propagation without storing full psi
            history. Return value is ignored.

        Returns
        -------
        PropagatorResult

        Notes
        -----
        The laser interaction is treated in the length gauge:
            V_laser(x, t) = E(t) · x
        The total effective potential at each half-step is:
            V_eff(x, t) = V(x) + E(t) · x
        """
        t_start = time.time()
        psi0 = self._validate_inputs(psi0, tgrid)
        _check_uniform_grid(self.xgrid, self.dx)
        _check_tgrid_dt_consistency(tgrid, self.dt)

        V = np.asarray(V, dtype=np.complex128)
        if V.shape != psi0.shape:
            raise ValueError(
                f"V.shape {V.shape} does not match psi0.shape {psi0.shape}"
            )

        psi, psi_history, norm_history, dipole_history = self._run_loop(
            psi0, tgrid, V, pulse, callback
        )

        return self._make_result(
            tgrid=tgrid,
            psi_history=psi_history,
            norm_history=norm_history,
            dipole_history=dipole_history,
            wall_time=time.time() - t_start,
        )

    # ------------------------------------------------------------------
    # Private: propagation loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        psi: np.ndarray,
        tgrid: np.ndarray,
        V: np.ndarray,
        pulse: callable,
        callback: callable | None,
    ) -> tuple[np.ndarray, list, list, list]:
        """Core propagation loop.

        Uses the merged V-step optimisation: consecutive half-V steps
        from adjacent time steps are combined into a single full V-step,
        halving the number of potential evaluations.

        The sequence is:
            half-V(t0) → [T → full-V(t)] × (Nt-1) → T → half-V(t_final)

        Returns
        -------
        tuple of (final psi, psi_history, norm_history, dipole_history)
        """
        Nt = len(tgrid)
        dt = self.dt
        dx = self.dx
        xgrid = self.xgrid

        psi_history = []
        norm_history = []
        dipole_history = []

        def record(psi: np.ndarray, t: float) -> None:
            """Store observables and optionally wavefunction."""
            norm = _compute_norm(psi, dx)
            dipole = _compute_dipole(psi, xgrid, dx)
            norm_history.append(norm)
            dipole_history.append(dipole)
            if self.store_psi:
                psi_history.append(psi.copy())
            else:
                psi_history.append(None)   # placeholder; only final used
            if callback is not None:
                callback(t, psi)

        # --- Initial half-V step at t = tgrid[0] ---
        E0 = pulse(tgrid[0])
        psi = _apply_V_half(psi, V, E0, xgrid, dt)
        record(psi, tgrid[0])

        # --- Main loop: T-step then full-V step ---
        for n in range(1, Nt):
            t = tgrid[n]

            # Full kinetic step in momentum space
            psi = _apply_T(psi, self._kinetic_phase)

            if n < Nt - 1:
                # Merge: second half of step n + first half of step n+1
                # Both evaluated at the midpoint time t_mid
                t_mid = 0.5 * (tgrid[n] + tgrid[n + 1])
                E_mid = pulse(t_mid)
                psi = _apply_V_full(psi, V, E_mid, xgrid, dt)
            else:
                # Final half-V step at the last time point
                E_final = pulse(t)
                psi = _apply_V_half(psi, V, E_final, xgrid, dt)

            record(psi, t)

        # If store_psi=False, replace placeholder list with final psi only
        if not self.store_psi:
            final = psi.copy()
            psi_history = [np.zeros_like(psi)] * (Nt - 1) + [final]

        return psi, psi_history, norm_history, dipole_history


# ---------------------------------------------------------------------------
# Private helper functions — pure NumPy, no Python loops
# ---------------------------------------------------------------------------

def _check_tgrid_dt_consistency(tgrid: np.ndarray, dt: float, rtol: float = 1e-6) -> None:
    """Raise ValueError if tgrid spacing is inconsistent with dt.

    The propagator takes exactly one step of size dt per tgrid interval.
    If the tgrid spacing differs from dt, the simulation will silently
    integrate over the wrong time interval.

    Parameters
    ----------
    tgrid : np.ndarray
        Time grid passed to propagate().
    dt : float
        Time step the propagator was constructed with.
    rtol : float
        Relative tolerance for the check.

    Raises
    ------
    ValueError
        If tgrid spacing does not match dt within rtol.
    """
    if len(tgrid) < 2:
        return
    tgrid_dt = tgrid[1] - tgrid[0]
    if not np.isclose(tgrid_dt, dt, rtol=rtol):
        raise ValueError(
            f"tgrid spacing ({tgrid_dt:.6f} au) does not match propagator "
            f"dt ({dt:.6f} au). The propagator takes exactly one step of "
            f"size dt per tgrid interval. Either adjust tgrid spacing to "
            f"match dt, or reconstruct the propagator with "
            f"dt={tgrid_dt:.6f}."
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
    """Apply half a potential step in position space.

    psi *= exp(-i · V_eff(x, t) · dt/2)

    where V_eff(x, t) = V(x) + E(t)·x
    """
    phase = np.exp(-0.5j * (V + E * xgrid) * dt)
    return psi * phase


def _apply_V_full(
    psi: np.ndarray,
    V: np.ndarray,
    E: float,
    xgrid: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply a full potential step in position space.

    psi *= exp(-i · V_eff(x, t) · dt)

    where V_eff(x, t) = V(x) + E(t)·x
    """
    phase = np.exp(-1.0j * (V + E * xgrid) * dt)
    return psi * phase


def _apply_T(psi: np.ndarray, kinetic_phase: np.ndarray) -> np.ndarray:
    """Apply the full kinetic step in momentum space.

    FFT → multiply by exp(-i·k²·dt/2) → IFFT
    """
    return np.fft.ifft(kinetic_phase * np.fft.fft(psi))


def _compute_norm(psi: np.ndarray, dx: float) -> float:
    """Compute ||ψ||² = dx · Σ |ψ(xⱼ)|²"""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def _compute_dipole(psi: np.ndarray, xgrid: np.ndarray, dx: float) -> float:
    """Compute <x> = dx · Σ xⱼ · |ψ(xⱼ)|²"""
    return float(np.sum(xgrid * np.abs(psi) ** 2) * dx)