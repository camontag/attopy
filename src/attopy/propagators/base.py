"""
attopy.propagators.base
=======================
Abstract base class and result dataclass for all attopy propagator backends.

All backends subclass PropagatorBase. The base class owns the time-stepping
loop via propagate(), which calls the abstract _advance_step() method that
each backend must implement. This ensures that observable computation,
memory management, and result assembly are handled uniformly across all
backends.

The exception is backends with fundamentally different calling conventions
(e.g. ITVOLT, which operates on intervals rather than individual steps) —
these may override propagate() entirely.

Example
-------
    from attopy.propagators.splitop import SplitOperatorPropagator
    import numpy as np

    N, dx, dt = 256, 0.05, 0.01
    xgrid = np.linspace(-N // 2 * dx, N // 2 * dx, N, endpoint=False)

    prop = SplitOperatorPropagator(dx=dx, dt=dt, xgrid=xgrid)
    result = prop.propagate(psi0, tgrid, V, pulse)

    print(result.norm)    # norm at each time step
    print(result.dipole)  # <x>(t) at each time step
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import warnings



@dataclass
class PropagatorResult:
    """Container for the output of a propagation run.

    Attributes
    ----------
    t : np.ndarray, shape (Nt,)
        Time grid in atomic units.
    psi : np.ndarray, shape (Nt, N), or None
        Wavefunction at each time step. Complex array.
        None if the propagator was constructed with store_psi=False,
        in which case only final_psi is available.
    norm : np.ndarray, shape (Nt,)
        Norm of the wavefunction at each time step. Should be ~1.0
        for field-free propagation with a real potential, decreasing
        when a complex absorbing potential (CAP) is present.
    dipole : np.ndarray, shape (Nt,)
        Expectation value of position <x>(t) in atomic units.
    dipole_acceleration : np.ndarray, shape (Nt,), or None
        Expectation value of acceleration <a>(t) computed via
        Ehrenfest's theorem during the propagation loop. None until
        Phase 4 when force operators are passed to the propagator.
        Do not use the numerical double-gradient of dipole for HHG
        spectra — it amplifies high-frequency noise quadratically.
    info : dict
        Metadata: backend name, dx, dt, step count, wall time.
    """

    t: np.ndarray
    psi: np.ndarray | None
    norm: np.ndarray
    dipole: np.ndarray
    dipole_acceleration: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        Nt = len(self.t)

        # psi is optional — only check shape if provided
        if self.psi is not None:
            if self.psi.shape[0] != Nt:
                raise ValueError(
                    f"psi.shape[0] = {self.psi.shape[0]} does not match "
                    f"len(t) = {Nt}"
                )

        if self.norm.shape != (Nt,):
            raise ValueError(
                f"norm.shape = {self.norm.shape}, expected ({Nt},)"
            )
        if self.dipole.shape != (Nt,):
            raise ValueError(
                f"dipole.shape = {self.dipole.shape}, expected ({Nt},)"
            )
        if self.dipole_acceleration is not None:
            if self.dipole_acceleration.shape != (Nt,):
                raise ValueError(
                    f"dipole_acceleration.shape = {self.dipole_acceleration.shape}, "
                    f"expected ({Nt},)"
                )

    @property
    def ionization_yield(self) -> np.ndarray:
        """Ionization yield at each time step: 1 - norm(t).

        Only meaningful when a complex absorbing potential (CAP) or
        mask function is used to absorb outgoing flux.

        Returns
        -------
        np.ndarray, shape (Nt,)
        """
        return 1.0 - self.norm

    @property
    def final_psi(self) -> np.ndarray:
        """Wavefunction at the final time step.

        Works regardless of store_psi setting: if psi history was stored,
        returns the last row; if not, returns the separately stored final
        wavefunction from info['final_psi'].

        Returns
        -------
        np.ndarray, shape (N,)

        Raises
        ------
        RuntimeError
            If neither psi history nor final_psi is available.
        """
        if self.psi is not None:
            return self.psi[-1]
        if "final_psi" in self.info:
            return self.info["final_psi"]
        raise RuntimeError(
            "No wavefunction data available. This should not happen — "
            "please file a bug report."
        )


class PropagatorBase(ABC):
    """Abstract base class for all attopy propagator backends.

    The base class owns the time-stepping loop (propagate()) and calls the
    abstract _advance_step() method that each backend implements. This
    ensures observable computation and result assembly are handled uniformly.

    Backends with fundamentally different calling conventions (e.g. ITVOLT)
    may override propagate() entirely instead of implementing _advance_step().

    Parameters
    ----------
    dx : float
        Spatial grid spacing in atomic units. Must be positive.
    dt : float
        Time step in atomic units. Must be positive.
    xgrid : np.ndarray, shape (N,)
        Position grid in atomic units. Must be uniformly spaced with
        spacing dx.
    store_psi : bool, optional
        If True (default), store the full wavefunction at every time step.
        If False, only the final wavefunction is retained, reducing memory
        usage for long propagations. Observables (norm, dipole) are always
        stored regardless of this setting.
    """

    def __init__(
        self,
        dx: float,
        dt: float,
        xgrid: np.ndarray,
        store_psi: bool = True,
    ):
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dx = dx
        self.dt = dt
        self.xgrid = xgrid
        self.store_psi = store_psi
        self._backend_name = self.__class__.__name__

    def propagate(
        self,
        psi0: np.ndarray,
        tgrid: np.ndarray,
        V: np.ndarray,
        pulse: callable,
        callback: callable | None = None,
    ) -> PropagatorResult:
        """Propagate an initial wavefunction forward in time.

        Owns the time-stepping loop. Calls _advance_step() at each step,
        records observables, and assembles a PropagatorResult.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,), complex
            Initial wavefunction. Normalised internally if needed.
        tgrid : np.ndarray, shape (Nt,)
            Time points in atomic units. Must be monotonically increasing
            with uniform spacing equal to dt.
        V : np.ndarray, shape (N,), complex
            Field-free potential in atomic units. Real part is the physical
            potential; imaginary part (if any) is a complex absorbing
            potential (CAP).
        pulse : callable
            Laser pulse. Signature: pulse(t) -> float for linear
            polarization. Each backend documents its own convention.
        callback : callable, optional
            Called at each step as callback(t, psi). Use to compute custom
            observables without storing full psi history.

        Returns
        -------
        PropagatorResult
        """
        t_start = time.time()
        psi = self._validate_inputs(psi0, tgrid)

        norm_history = []
        dipole_history = []
        psi_history = []
        final_psi = None

        for n, t in enumerate(tgrid):
            # Advance one step (skip on first iteration — record initial state)
            if n > 0:
                psi = self._advance_step(psi, tgrid[n - 1], tgrid[n], V, pulse)

            # Record observables
            norm = float(np.sum(np.abs(psi) ** 2) * self.dx)
            dipole = float(np.sum(self.xgrid * np.abs(psi) ** 2) * self.dx)
            norm_history.append(norm)
            dipole_history.append(dipole)

            if self.store_psi:
                psi_history.append(psi.copy())

            if callback is not None:
                callback(t, psi)

        final_psi = psi.copy()

        return self._make_result(
            tgrid=tgrid,
            psi_history=psi_history if self.store_psi else None,
            final_psi=final_psi,
            norm_history=norm_history,
            dipole_history=dipole_history,
            wall_time=time.time() - t_start,
        )

    @abstractmethod
    def _advance_step(
        self,
        psi: np.ndarray,
        t_start: float,
        t_end: float,
        V: np.ndarray,
        pulse: callable,
    ) -> np.ndarray:
        """Advance the wavefunction by one time step.

        Parameters
        ----------
        psi : np.ndarray, shape (N,), complex
            Wavefunction at t_start.
        t_start : float
            Start of the time step (au).
        t_end : float
            End of the time step (au). t_end - t_start == self.dt.
        V : np.ndarray, shape (N,), complex
            Field-free potential (may include imaginary CAP).
        pulse : callable
            Laser pulse function.

        Returns
        -------
        np.ndarray, shape (N,), complex
            Wavefunction at t_end.
        """
        ...  # pragma: no cover

    def _validate_inputs(
        self,
        psi0: np.ndarray,
        tgrid: np.ndarray,
    ) -> np.ndarray:
        """Validate and normalise inputs.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial wavefunction.
        tgrid : np.ndarray
            Time grid.

        Returns
        -------
        np.ndarray
            Normalised copy of psi0 as complex128.

        Raises
        ------
        ValueError
            If inputs are malformed.
        """
        if psi0.ndim != 1:
            raise ValueError(f"psi0 must be 1D, got shape {psi0.shape}")
        if tgrid.ndim != 1 or len(tgrid) < 2:
            raise ValueError("tgrid must be a 1D array with at least 2 points")
        if not np.all(np.diff(tgrid) > 0):
            raise ValueError("tgrid must be monotonically increasing")

        psi0 = np.array(psi0, dtype=np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi0) ** 2) * self.dx)
        if not np.isclose(norm, 1.0, rtol=1e-6):
            warnings.warn(
                f"Initial wavefunction has norm {norm:.6f}, expected 1.0. "
                f"Normalizing automatically. Pass a pre-normalized state to "
                f"suppress this warning.",
                UserWarning,
                stacklevel=3,   # points to the user's call site, not this function
            )
            psi0 = psi0 / norm

        return psi0

    def _make_result(
        self,
        tgrid: np.ndarray,
        psi_history: list[np.ndarray] | None,
        final_psi: np.ndarray,
        norm_history: list[float],
        dipole_history: list[float],
        wall_time: float,
    ) -> PropagatorResult:
        """Assemble a PropagatorResult from accumulated history lists."""
        info = {
            "backend": self._backend_name,
            "dx": self.dx,
            "dt": self.dt,
            "n_steps": len(tgrid),
            "wall_time_s": wall_time,
            "final_psi": final_psi,
        }
        return PropagatorResult(
            t=tgrid,
            psi=np.array(psi_history) if psi_history is not None else None,
            norm=np.array(norm_history),
            dipole=np.array(dipole_history),
            info=info,
        )

    def __repr__(self) -> str:
        return (
            f"{self._backend_name}("
            f"dx={self.dx}, dt={self.dt}, "
            f"N={len(self.xgrid)}, store_psi={self.store_psi})"
        )