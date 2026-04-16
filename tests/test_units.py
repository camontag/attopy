"""
Tests for attopy.units
"""
import math
import pytest
from attopy import units


# ---------------------------------------------------------------------------
# Round-trip conversion tests
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_intensity_round_trip(self):
        I = 1e14
        assert math.isclose(units.au_to_intensity(units.intensity_to_au(I)), I, rel_tol=1e-12)

    def test_energy_round_trip(self):
        E = 15.76  # Ar ionization potential in eV
        assert math.isclose(units.au_to_energy(units.energy_to_au(E)), E, rel_tol=1e-12)

    def test_time_round_trip(self):
        t = 10.0  # fs
        assert math.isclose(units.au_to_time(units.time_to_au(t)), t, rel_tol=1e-12)

    def test_wavelength_omega_round_trip(self):
        lam = 800.0  # nm
        assert math.isclose(
            units.omega_to_wavelength(units.wavelength_to_omega(lam)), lam, rel_tol=1e-12
        )

    def test_efield_round_trip(self):
        E = 1e10  # V/m
        assert math.isclose(units.au_to_efield(units.efield_to_au(E)), E, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_1au_intensity(self):
        """1 au of intensity should be ~3.509e16 W/cm²."""
        assert math.isclose(
            units.au_to_intensity(1.0), 3.50944758e16, rel_tol=1e-6
        )

    def test_1au_time_in_fs(self):
        """1 au of time should be ~24.19 as = 0.02419 fs."""
        assert math.isclose(units.au_to_time(1.0), 0.02418884254, rel_tol=1e-6)

    def test_1_hartree_in_eV(self):
        """1 Hartree should be ~27.211 eV."""
        assert math.isclose(units.au_to_energy(1.0), 27.211386245988, rel_tol=1e-8)

    def test_800nm_omega(self):
        """800 nm Ti:Sapph laser → ω ≈ 0.05700 au."""
        omega = units.wavelength_to_omega(800.0)
        assert math.isclose(omega, 0.057005, rel_tol=1e-3)

    def test_efield_amplitude_consistency(self):
        """E₀² should equal I in au (since Up = E₀²/4ω²)."""
        I_au = units.intensity_to_au(1e14)
        E0 = units.efield_amplitude(1e14)
        assert math.isclose(E0**2, I_au, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Ponderomotive energy
# ---------------------------------------------------------------------------

class TestPonderomotiveEnergy:
    def test_returns_dict_with_correct_keys(self):
        up = units.ponderomotive_energy(1e14, 800.0)
        assert "au" in up
        assert "eV" in up

    def test_au_eV_consistency(self):
        """au and eV values should be consistent via HARTREE_TO_EV."""
        up = units.ponderomotive_energy(1e14, 800.0)
        assert math.isclose(up["eV"], up["au"] * units.HARTREE_TO_EV, rel_tol=1e-12)

    def test_known_value_800nm_1e14(self):
        """800 nm, 10^14 W/cm² → Up ≈ 5.93 eV (standard reference value)."""
        up = units.ponderomotive_energy(1e14, 800.0)
        assert math.isclose(up["eV"], 5.93, rel_tol=2e-2)

    def test_scales_linearly_with_intensity(self):
        """Up ∝ I — doubling intensity should double Up."""
        up1 = units.ponderomotive_energy(1e14, 800.0)
        up2 = units.ponderomotive_energy(2e14, 800.0)
        assert math.isclose(up2["au"] / up1["au"], 2.0, rel_tol=1e-10)

    def test_scales_quadratically_with_wavelength(self):
        """Up ∝ λ² — doubling wavelength should quadruple Up."""
        up1 = units.ponderomotive_energy(1e14, 800.0)
        up2 = units.ponderomotive_energy(1e14, 1600.0)
        assert math.isclose(up2["au"] / up1["au"], 4.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Keldysh parameter
# ---------------------------------------------------------------------------

class TestKeldyshParameter:
    def test_argon_800nm_1e14(self):
        """Ar (Ip=15.76 eV), 800 nm, 10^14 W/cm² → γ ≈ 1.1."""
        gamma = units.keldysh_parameter(1e14, 800.0, 15.76)
        assert math.isclose(gamma, 1.1, rel_tol=5e-2)

    def test_tunnel_regime_high_intensity(self):
        """At very high intensity, should be well into tunnel regime (γ < 1)."""
        gamma = units.keldysh_parameter(5e14, 800.0, 15.76)
        assert gamma < 1.0

    def test_multiphoton_regime_low_intensity(self):
        """At very low intensity, should be in multiphoton regime (γ >> 1)."""
        gamma = units.keldysh_parameter(1e12, 800.0, 15.76)
        assert gamma > 1.0

    def test_positive(self):
        gamma = units.keldysh_parameter(1e14, 800.0, 15.76)
        assert gamma > 0.0

    def test_increases_with_Ip(self):
        """Higher Ip → larger γ (harder to tunnel ionize)."""
        gamma_ar = units.keldysh_parameter(1e14, 800.0, 15.76)  # Ar
        gamma_he = units.keldysh_parameter(1e14, 800.0, 24.59)  # He
        assert gamma_he > gamma_ar

    def test_decreases_with_intensity(self):
        """Higher intensity → smaller γ (deeper into tunnel regime)."""
        gamma_low = units.keldysh_parameter(1e13, 800.0, 15.76)
        gamma_high = units.keldysh_parameter(1e14, 800.0, 15.76)
        assert gamma_high < gamma_low


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_speed_of_light_au(self):
        """c in au should be ~137 (= 1/α)."""
        assert math.isclose(units.C_AU, 137.035999084, rel_tol=1e-8)

    def test_fine_structure_constant(self):
        """α ≈ 1/137."""
        assert math.isclose(units.ALPHA_FINE, 1.0 / 137.035999084, rel_tol=1e-8)

    def test_conversion_constants_are_reciprocals(self):
        assert math.isclose(units.HARTREE_TO_EV * units.EV_TO_HARTREE, 1.0, rel_tol=1e-12)
        assert math.isclose(units.BOHR_TO_ANGSTROM * units.ANGSTROM_TO_BOHR, 1.0, rel_tol=1e-12)
        assert math.isclose(units.AU_TIME_TO_FS * units.FS_TO_AU_TIME, 1.0, rel_tol=1e-12)
        assert math.isclose(units.AU_INTENSITY_TO_WCMCM * units.WCMCM_TO_AU_INTENSITY, 1.0, rel_tol=1e-12)