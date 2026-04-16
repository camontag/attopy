# attopy

[![Tests](https://github.com/camontag/attopy/actions/workflows/tests.yml/badge.svg)](https://github.com/camontag/attopy/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A Python toolkit for strong-field TDSE simulations, targeting attosecond and ultrafast physics research.

**attopy** provides a clean, NumPy-native interface for propagating the time-dependent Schrödinger equation (TDSE) in intense laser fields, with built-in tools for computing strong-field observables such as high-harmonic generation (HHG) spectra and above-threshold ionization (ATI) yields.

---

## Features

**Multiple propagator backends** — a common API across all solvers, swappable with a single argument:
- Split-operator FFT (default) — handles arbitrary pulse shapes including circular and elliptical polarization
- Crank-Nicolson *(Phase 1 — coming soon)*
- ITVOLT Fortran backend *(Phase 3 — optional compiled backend)*

**Laser pulse library** *(Phase 2)* — sin², Gaussian, few-cycle, bi-chromatic, and circularly polarized pulses with carrier-envelope phase control.

**Strong-field observables** *(Phase 4)* — HHG spectra, ATI photoelectron distributions, ionization yields, time-frequency analysis.

**SFA comparison module** *(Phase 5)* — Lewenstein model and classical three-step model alongside TDSE results.

**Atomic units throughout** — with a `units` module for converting laser parameters (W/cm², nm, fs, eV) to and from atomic units.

---

## Installation

```bash
pip install attopy
```

For development:

```bash
git clone https://github.com/camontag/attopy
cd attopy
pip install -e ".[dev]"
```

> **Note:** attopy is currently in early development. The API may change between versions.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`01_coherent_state_benchmark.ipynb`](notebooks/01_coherent_state_benchmark.ipynb) | Split-operator validation against exact analytic result; convergence study |

---

## Project Status

attopy is being developed as part of a PhD research project in strong-field physics. The development follows a phased plan targeting a JOSS publication.

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Complete | Split-operator backend, units module, CI |
| 1 | 🔄 In progress | Crank-Nicolson backend, Hamiltonian builder, ground state prep |
| 2 | ⏳ Planned | Laser pulse library |
| 3 | ⏳ Planned | ITVOLT Fortran backend |
| 4 | ⏳ Planned | Strong-field observables (HHG, ATI) |
| 5 | ⏳ Planned | SFA comparison module |

---

## Citation

If you use attopy in your research, please cite:

```bibtex
@software{attopy,
  author  = {Carter Montag},
  title   = {attopy: A Python toolkit for strong-field TDSE simulations},
  url     = {https://github.com/camontag/attopy},
  year    = {2026},
}
```

The split-operator FFT method is described in:

> Feit, M. D., Fleck, J. A., & Steiger, A. (1982). Solution of the Schrödinger equation by a spectral method. *Journal of Computational Physics*, 47(3), 412–433.

The optional ITVOLT backend uses:

> Schneider, R. et al. (2023). ITVOLT: An iterative solver for the time-dependent Schrödinger equation. *Computer Physics Communications*, 289, 108765.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request so we can discuss the proposed change. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

MIT — see [`LICENSE`](LICENSE) for details.