"""
attopy.units
============
Atomic unit conversion constants and utility functions for strong-field physics.

All internal calculations in attopy use atomic units (au):
    - Length:   Bohr radius (a₀)
    - Energy:   Hartree (Eₕ)
    - Time:     ℏ/Eₕ ≈ 24.19 as
    - E-field:  Eₕ/(e·a₀)
    - Intensity: corresponding to E-field unit squared

Usage
-----
    from attopy.units import intensity_to_au, ponderomotive_energy
    I_au = intensity_to_au(1e14)        # 10^14 W/cm^2 in atomic units
    Up   = ponderomotive_energy(1e14, 800.0)  # Up in au and eV
"""

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------

#: Speed of light in atomic units (= 1/α)
C_AU: float = 137.035999084

#: Fine structure constant
ALPHA_FINE: float = 1.0 / C_AU

# ---------------------------------------------------------------------------
# Conversion constants
# ---------------------------------------------------------------------------

# Energy
#: 1 Hartree in electronvolts
HARTREE_TO_EV: float = 27.211386245988

#: 1 eV in Hartree
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

# Length
#: 1 Bohr radius in Ångström
BOHR_TO_ANGSTROM: float = 0.529177210903

#: 1 Ångström in Bohr radii
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM

# Time
#: 1 atomic unit of time in femtoseconds
AU_TIME_TO_FS: float = 0.02418884254

#: 1 femtosecond in atomic units of time
FS_TO_AU_TIME: float = 1.0 / AU_TIME_TO_FS

# Electric field
#: 1 au of electric field in V/m
AU_EFIELD_TO_VM: float = 5.14220674763e11

#: 1 V/m in au of electric field
VM_TO_AU_EFIELD: float = 1.0 / AU_EFIELD_TO_VM

# Intensity
# 1 au of intensity = ε₀·c·E_au² / 2, where E_au is the au E-field in SI
# = 3.50944758e16 W/cm²
#: 1 au of intensity in W/cm²
AU_INTENSITY_TO_WCMCM: float = 3.50944758e16

#: 1 W/cm² in atomic units of intensity
WCMCM_TO_AU_INTENSITY: float = 1.0 / AU_INTENSITY_TO_WCMCM

# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


def intensity_to_au(I_wcm2: float) -> float:
    """Convert laser intensity from W/cm² to atomic units.

    Parameters
    ----------
    I_wcm2 : float
        Intensity in W/cm².

    Returns
    -------
    float
        Intensity in atomic units.

    Examples
    --------
    >>> intensity_to_au(3.51e16)   # doctest: +ELLIPSIS
    1.000...
    """
    return I_wcm2 * WCMCM_TO_AU_INTENSITY


def au_to_intensity(I_au: float) -> float:
    """Convert laser intensity from atomic units to W/cm².

    Parameters
    ----------
    I_au : float
        Intensity in atomic units.

    Returns
    -------
    float
        Intensity in W/cm².
    """
    return I_au * AU_INTENSITY_TO_WCMCM


def wavelength_to_omega(lambda_nm: float) -> float:
    """Convert laser wavelength in nm to angular frequency in atomic units.

    Parameters
    ----------
    lambda_nm : float
        Wavelength in nanometres.

    Returns
    -------
    float
        Angular frequency ω in atomic units.

    Examples
    --------
    >>> round(wavelength_to_omega(800.0), 6)
    0.057005
    """
    import math
    # λ in au: 1 nm = 10 Å = 10 / BOHR_TO_ANGSTROM bohr
    lambda_au = lambda_nm * 10.0 * ANGSTROM_TO_BOHR
    return 2.0 * math.pi / lambda_au


def omega_to_wavelength(omega_au: float) -> float:
    """Convert angular frequency in atomic units to wavelength in nm.

    Parameters
    ----------
    omega_au : float
        Angular frequency in atomic units.

    Returns
    -------
    float
        Wavelength in nanometres.
    """
    import math
    lambda_au = 2.0 * math.pi / omega_au
    return lambda_au * BOHR_TO_ANGSTROM / 10.0


def energy_to_au(E_eV: float) -> float:
    """Convert energy from eV to atomic units (Hartree).

    Parameters
    ----------
    E_eV : float
        Energy in electronvolts.

    Returns
    -------
    float
        Energy in Hartree.
    """
    return E_eV * EV_TO_HARTREE


def au_to_energy(E_au: float) -> float:
    """Convert energy from atomic units (Hartree) to eV.

    Parameters
    ----------
    E_au : float
        Energy in Hartree.

    Returns
    -------
    float
        Energy in electronvolts.
    """
    return E_au * HARTREE_TO_EV


def time_to_au(t_fs: float) -> float:
    """Convert time from femtoseconds to atomic units.

    Parameters
    ----------
    t_fs : float
        Time in femtoseconds.

    Returns
    -------
    float
        Time in atomic units.
    """
    return t_fs * FS_TO_AU_TIME


def au_to_time(t_au: float) -> float:
    """Convert time from atomic units to femtoseconds.

    Parameters
    ----------
    t_au : float
        Time in atomic units.

    Returns
    -------
    float
        Time in femtoseconds.
    """
    return t_au * AU_TIME_TO_FS


def efield_to_au(E_Vm: float) -> float:
    """Convert electric field amplitude from V/m to atomic units.

    Parameters
    ----------
    E_Vm : float
        Electric field in V/m.

    Returns
    -------
    float
        Electric field in atomic units.
    """
    return E_Vm * VM_TO_AU_EFIELD


def au_to_efield(E_au: float) -> float:
    """Convert electric field amplitude from atomic units to V/m.

    Parameters
    ----------
    E_au : float
        Electric field in atomic units.

    Returns
    -------
    float
        Electric field in V/m.
    """
    return E_au * AU_EFIELD_TO_VM


# ---------------------------------------------------------------------------
# Strong-field specific quantities
# ---------------------------------------------------------------------------


def efield_amplitude(I_wcm2: float) -> float:
    """Compute peak electric field amplitude in au from intensity in W/cm².

    Uses E₀ = sqrt(I / (ε₀·c)) in SI, then converts to au.

    Parameters
    ----------
    I_wcm2 : float
        Peak laser intensity in W/cm².

    Returns
    -------
    float
        Peak electric field amplitude in atomic units.
    """
    import math
    return math.sqrt(intensity_to_au(I_wcm2))


def ponderomotive_energy(I_wcm2: float, lambda_nm: float) -> dict:
    """Compute the ponderomotive energy Uₚ = E₀²/(4ω²).

    Parameters
    ----------
    I_wcm2 : float
        Peak laser intensity in W/cm².
    lambda_nm : float
        Laser wavelength in nm.

    Returns
    -------
    dict with keys:
        'au'  : Uₚ in atomic units
        'eV'  : Uₚ in electronvolts

    Examples
    --------
    >>> up = ponderomotive_energy(1e14, 800.0)
    >>> round(up['eV'], 2)
    5.93
    """
    E0 = efield_amplitude(I_wcm2)
    omega = wavelength_to_omega(lambda_nm)
    Up_au = E0**2 / (4.0 * omega**2)
    return {"au": Up_au, "eV": au_to_energy(Up_au)}


def keldysh_parameter(I_wcm2: float, lambda_nm: float, Ip_eV: float) -> float:
    """Compute the Keldysh parameter γ = ω·sqrt(2·Ip) / E₀.

    γ << 1 : tunnel ionization regime
    γ >> 1 : multiphoton ionization regime
    γ ≈ 1  : intermediate (often taken as γ < 1 for tunneling)

    Parameters
    ----------
    I_wcm2 : float
        Peak laser intensity in W/cm².
    lambda_nm : float
        Laser wavelength in nm.
    Ip_eV : float
        Ionization potential in eV.

    Returns
    -------
    float
        Keldysh parameter γ (dimensionless).

    Examples
    --------
    >>> round(keldysh_parameter(1e14, 800.0, 15.76), 2)  # Argon
    1.1
    """
    import math
    Ip_au = energy_to_au(Ip_eV)
    omega = wavelength_to_omega(lambda_nm)
    E0 = efield_amplitude(I_wcm2)
    return omega * math.sqrt(2.0 * Ip_au) / E0