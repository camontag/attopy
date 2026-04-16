"""
attopy.propagators.base
=======================
Abstract base class and result dataclass for all attopy propagator backends.

All backends subclass PropagatorBase and implement the propagate() method,
returning a PropagatorResult. This ensures the API is identical regardless
of which backend is used.

Example
-------
    from attopy.propagators.splitop import SplitOperatorPropagator
    import numpy as np

    propagator = SplitOperatorPropagator(dx=0.1, dt=0.05)
    result = propagator.propagate(psi0, tgrid, H0, pulse)

    print(result.norm)    # norm at each time step
    print(result.dipole)  # <x>(t) at each time step
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PropagatorResult:
    """Container for the output of a propagation run.

    Attributes
    ----------
    t : np.ndarray, shape (Nt,)
        Time grid in atomic units.
    psi : np.ndarray, shape (Nt, N)
        Wavefunction at each time step. Complex array.
    norm : np.ndarray, shape (Nt,)
        Norm of the wavefunction at each time step. Should be 1.0
        for field-free propagation, decreasing when a CAP is present.
    dipole : np.ndarray, shape (Nt,)
        Expectation value of position <x>(t) in atomic units.
    info : dict
        Metadata about the run: backend name, wall time, step count, etc.
    """

    t: np.ndarray
    psi: np.ndarray
    norm: np.ndarray
    dipole: np.ndarray
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        # Basic shape consistency checks
        Nt = len(self.t)
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
    def dipole_acceleration(self) -> np.ndarray:
        """Numerical second derivative of dipole moment: d²<x>/dt².

        Used as input to HHG spectrum calculation.

        Returns
        -------
        np.ndarray, shape (Nt,)
            Dipole acceleration in atomic units. Edge points use
            lower-order finite differences.
        """
        dipole_velocity = np.gradient(self.dipole, self.t)
        return np.gradient(dipole_velocity, self.t)

    @property
    def final_psi(self) -> np.ndarray:
        """Wavefunction at the final time step.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        return self.psi[-1]


class PropagatorBase(ABC):
    """Abstract base class for all attopy propagator backends.

    Subclasses must implement the propagate() method. The interface is
    intentionally kept minimal — all physical setup (grid, potential,
    pulse) is passed to propagate() rather than the constructor, so
    that a single propagator instance can be reused across multiple
    simulations.

    Parameters
    ----------
    dx : float
        Spatial grid spacing in atomic units.
    dt : float
        Time step in atomic units.
    """

    def __init__(self, dx: float, dt: float, xgrid: np.ndarray, store_psi: bool = True):
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dx = dx
        self.dt = dt
        self.xgrid = xgrid
        self.store_psi = store_psi
        self._backend_name = self.__class__.__name__

    @abstractmethod
    def propagate(
        self,
        psi0: np.ndarray,
        tgrid: np.ndarray,
        H0: np.ndarray,
        pulse: callable,
        callback: callable | None = None,
    ) -> PropagatorResult:
        """Propagate an initial wavefunction forward in time.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,), complex
            Initial wavefunction. Will be normalised internally if not
            already normalised.
        tgrid : np.ndarray, shape (Nt,)
            Time points in atomic units at which to store the wavefunction.
            Must be monotonically increasing.
        H0 : np.ndarray, shape (N,) or (N, N)
            Field-free Hamiltonian. Shape (N,) implies a diagonal Hamiltonian
            (e.g. in momentum space); shape (N, N) is a full or banded matrix.
        pulse : callable
            Laser pulse function. Signature: pulse(t) -> float for linear
            polarization, or pulse(t) -> np.ndarray of shape (2,) for
            circular/elliptical polarization.
        callback : callable, optional
            Called at each time step as callback(t, psi). Use this to
            compute custom observables without storing the full wavefunction
            history. Return value is ignored.

        Returns
        -------
        PropagatorResult
        """
        ...

    def _validate_inputs(
        self,
        psi0: np.ndarray,
        tgrid: np.ndarray,
    ) -> np.ndarray:
        """Validate and normalise inputs. Called by subclasses at the
        start of propagate().

        Parameters
        ----------
        psi0 : np.ndarray
            Initial wavefunction.
        tgrid : np.ndarray
            Time grid.

        Returns
        -------
        np.ndarray
            Normalised copy of psi0 as a complex128 array.

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
            psi0 = psi0 / norm

        return psi0

    def _make_result(
        self,
        tgrid: np.ndarray,
        psi_history: list[np.ndarray],
        norm_history: list[float],
        dipole_history: list[float],
        wall_time: float,
    ) -> PropagatorResult:
        """Assemble a PropagatorResult from accumulated history lists.
        Called by subclasses at the end of propagate().
        """
        return PropagatorResult(
            t=tgrid,
            psi=np.array(psi_history),
            norm=np.array(norm_history),
            dipole=np.array(dipole_history),
            info={
                "backend": self._backend_name,
                "dx": self.dx,
                "dt": self.dt,
                "n_steps": len(tgrid),
                "wall_time_s": wall_time,
            },
        )

    def __repr__(self) -> str:
        return f"{self._backend_name}(dx={self.dx}, dt={self.dt})"