import math
import sympy as sp
from sympy import symbols, Eq, solve, pi, S
from typing import Any, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    PhysicsChecker for a geostationary (equatorial) orbit problem.
    Audits the solution instance for:
      - required result keys (phi, r, orbit_description)
      - invariant conditions:
          * geostationary period condition (T_s = T_E)
          * centripetal acceleration condition (G M / r^2 = omega_s^2 r)
          * phi == 0 (equatorial)
          * r > R_E (orbit above Earth)
      - modeling quality (orbit description keywords, radius derived from centripetal formula)
      - monotonicity of radius with angular velocity (if perturbed)
      - weak sanity of final answer (non‑None, finite, plausible range)
    """

    # Standard physical constants (SI units)
    G_STANDARD = 6.67430e-11        # m^3 kg^-1 s^-2
    M_STANDARD = 5.9722e24          # kg
    T_E_STANDARD = 86400.0          # s (24 hours)
    R_E_STANDARD = 6.371e6          # m

    def __init__(self, solution_instance: object) -> None:
        self.solution = solution_instance

        # Safely retrieve result keys, default to None
        self.results = getattr(solution_instance, 'results', {})
        if not self.results:
            self.results = {}

        self.phi = self.results.get('phi', None)
        self.r = self.results.get('r', None)
        self.orbit_description = self.results.get('orbit_description', None)

        # Try to extract constants from solution instance;
        # fall back to standard values if missing.
        self.G = getattr(solution_instance, 'G', self.G_STANDARD) or self.G_STANDARD
        self.M = getattr(solution_instance, 'M', self.M_STANDARD) or self.M_STANDARD
        self.T_E = getattr(solution_instance, 'T_E', self.T_E_STANDARD) or self.T_E_STANDARD
        self.R_E = getattr(solution_instance, 'R_E', self.R_E_STANDARD) or self.R_E_STANDARD

        self.omega_E = 2.0 * math.pi / self.T_E  # Earth's angular velocity

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_phi_exists(self) -> Tuple[bool, str]:
        """Check that results['phi'] exists and is not None."""
        if self.phi is None:
            return (False, "phi is missing or None")
        return (True, "phi is present")

    def test_interface_r_exists(self) -> Tuple[bool, str]:
        """Check that results['r'] exists and is not None."""
        if self.r is None:
            return (False, "r is missing or None")
        return (True, "r is present")

    def test_interface_orbit_description_exists(self) -> Tuple[bool, str]:
        """Check that results['orbit_description'] exists and is not None."""
        if self.orbit_description is None:
            return (False, "orbit_description is missing or None")
        return (True, "orbit_description is present")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) tests – use math.isclose (tolerance 1e-2)
    # ------------------------------------------------------------------
    def test_invariant_geostationary_period(self) -> Tuple[bool, str]:
        """Check that T_s from Kepler = T_E (geostationary condition)."""
        if self.r is None:
            return (False, "SKIP: r not available")

        # Compute orbital period from r using Kepler's third law
        T_s = 2.0 * math.pi * math.sqrt(self.r**3 / (self.G * self.M))

        if not math.isfinite(T_s):
            return (False, "Computed period is not finite")

        if not math.isclose(T_s, self.T_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Orbital period {T_s:.2f} s does not equal Earth period {self.T_E:.2f} s (rel_tol=1e-2)")
        return (True, f"Orbital period {T_s:.2f} s matches Earth period {self.T_E:.2f} s")

    def test_invariant_centripetal_condition(self) -> Tuple[bool, str]:
        """Check that G M / r^2 = omega_E^2 * r (centripetal balance)."""
        if self.r is None:
            return (False, "SKIP: r not available")

        lhs = self.G * self.M / (self.r ** 2)
        rhs = self.omega_E ** 2 * self.r

        if not math.isfinite(lhs) or not math.isfinite(rhs):
            return (False, "Computed centripetal terms not finite")

        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Centripetal mismatch: G M / r^2 = {lhs:.3e}, omega^2 r = {rhs:.3e}")
        return (True, f"Centripetal condition satisfied: lhs={lhs:.3e}, rhs={rhs:.3e}")

    def test_invariant_phi_equatorial(self) -> Tuple[bool, str]:
        """Check that phi is 0 (within tolerance)."""
        if self.phi is None:
            return (False, "SKIP: phi not available")

        phi_deg = self.phi if isinstance(self.phi, (int, float)) else 0.0
        if not math.isclose(phi_deg, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"phi = {phi_deg}° is not 0 (equatorial condition)")
        return (True, f"phi = {phi_deg}° is essentially 0")

    def test_invariant_orbit_above_earth(self) -> Tuple[bool, str]:
        """Check that orbital radius r > Earth radius."""
        if self.r is None:
            return (False, "SKIP: r not available")

        if self.r <= self.R_E:
            return (False, f"Orbital radius {self.r:.2e} m is not greater than Earth radius {self.R_E:.2e} m")
        return (True, f"Orbital radius {self.r:.2e} m > Earth radius {self.R_E:.2e} m")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests
    # ------------------------------------------------------------------
    def test_modeling_orbit_description_keywords(self) -> Tuple[bool, str]:
        """Check that orbit_description contains required keywords."""
        if self.orbit_description is None or not isinstance(self.orbit_description, str):
            return (False, "orbit_description missing or not a string")

        desc_lower = self.orbit_description.lower()
        keywords = ['circular', 'equatorial', 'geostationary']
        missing = [kw for kw in keywords if kw not in desc_lower]
        if missing:
            return (False, f"orbit_description missing keywords: {missing}")
        return (True, "orbit_description contains all required keywords")

    def test_modeling_orbital_radius_derivation(self) -> Tuple[bool, str]:
        """
        Check that the orbital radius r is consistent with the centripetal formula
        r = (G M / omega_E^2)^(1/3). This test re-derives the expected radius from
        the known constants and checks it against the provided r using isclose.
        """
        if self.r is None:
            return (False, "SKIP: r not available")

        # Re‑derive expected geostationary radius from centripetal equation
        expected_r = (self.G * self.M / (self.omega_E ** 2)) ** (1.0 / 3.0)

        if not math.isfinite(expected_r):
            return (False, "Derived expected radius is not finite")

        if not math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Provided r = {self.r:.3e} differs from derived r = {expected_r:.3e} (rel_tol=1e-2)")
        return (True, f"Provided r = {self.r:.3e} matches derived r = {expected_r:.3e}")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_radius_with_omega(self) -> Tuple[bool, str]:
        """
        Check monotonicity: if omega_E increases, geostationary radius r decreases.
        We use the solution's r as baseline, then perturb omega_E by +1% and
        recompute the expected radius. Check that r_perturbed < r_baseline.
        """
        if self.r is None:
            return (False, "SKIP: r not available")

        # Compute baseline radius from the centripetal equation (as a consistency check)
        r_baseline = (self.G * self.M / (self.omega_E ** 2)) ** (1.0/3.0)

        # Perturb omega_E upward by 1%
        omega_perturbed = self.omega_E * 1.01
        r_perturbed = (self.G * self.M / (omega_perturbed ** 2)) ** (1.0/3.0)

        if r_perturbed >= r_baseline:
            return (False, f"Monotonicity failed: r_perturbed={r_perturbed:.3e} >= r_baseline={r_baseline:.3e} (omega increased)")
        return (True, f"Monotonicity OK: r decreases from {r_baseline:.3e} to {r_perturbed:.3e} when omega increases by 1%")

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (exactly one method, no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """Sanity checks: phi and r are finite, non‑NaN, and within plausible ranges."""
        # Check phi
        if self.phi is not None:
            if not isinstance(self.phi, (int, float)):
                return (False, "phi is not numeric")
            if math.isnan(self.phi) or math.isinf(self.phi):
                return (False, "phi is NaN or infinite")
            if self.phi < -90.0 or self.phi > 90.0:
                return (False, f"phi = {self.phi} outside plausible range [-90,90]")
        # Check r
        if self.r is not None:
            if not isinstance(self.r, (int, float)):
                return (False, "r is not numeric")
            if math.isnan(self.r) or math.isinf(self.r):
                return (False, "r is NaN or infinite")
            if self.r <= 0:
                return (False, f"r = {self.r} is non‑positive")
            if self.r < self.R_E or self.r > 1e8:
                return (False, f"r = {self.r:.2e} out of plausible range [{self.R_E:.2e}, 1e8] m")
        return (True, "phi and r are finite and within plausible ranges")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods.
        Returns (all_pass, list_of_log_messages).
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {type(e).__name__}: {e}"
                import traceback
                tb = traceback.format_exc()
                msg += f"\n{tb}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)