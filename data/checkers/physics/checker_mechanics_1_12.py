import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits the solution for the problem:
    "Find the gravitational mass of the galaxy in solar masses
     given the Sun's orbital radius and period, and Earth's orbital radius and period."
    """
    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = getattr(solution_instance, 'U', None)
        if results is None:
            results = {}

        # Primary output
        self.M_gal_solar = results.get('M_gal_solar', None)

        # Attempt to extract given inputs from the solution instance
        self.R_ly = getattr(solution_instance, 'R_ly', None)
        self.T_yr = getattr(solution_instance, 'T_yr', None)
        self.r_lightmin = getattr(solution_instance, 'r_lightmin', None)
        self.t_yr = getattr(solution_instance, 't_yr', None)
        self.minutes_per_ly = getattr(solution_instance, 'minutes_per_ly', None)

        # Try alternative container (givens or parameters)
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = getattr(solution_instance, 'parameters', None)
        if isinstance(givens, dict):
            if self.R_ly is None:
                self.R_ly = givens.get('R_ly', None)
            if self.T_yr is None:
                self.T_yr = givens.get('T_yr', None)
            if self.r_lightmin is None:
                self.r_lightmin = givens.get('r_lightmin', None)
            if self.t_yr is None:
                self.t_yr = givens.get('t_yr', None)
            if self.minutes_per_ly is None:
                self.minutes_per_ly = givens.get('minutes_per_ly', None)

    # --------------------------------------------------------------------------
    # Interface Checks
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Verify that 'M_gal_solar' exists and is a finite number."""
        if self.M_gal_solar is None:
            return (False, "SKIP: M_gal_solar key missing or None")
        if not math.isfinite(self.M_gal_solar):
            return (False, "M_gal_solar is non-finite")
        return (True, "M_gal_solar present and finite")

    # --------------------------------------------------------------------------
    # Theorem-Based Invariant Checks (at least two)
    # --------------------------------------------------------------------------
    def test_invariant_Kepler_ratio(self) -> Tuple[bool, str]:
        """
        Physics invariant: M_gal_solar should equal (R/r)^3 * (t/T)^2
        with R converted to light minutes.
        """
        if any(v is None for v in [self.R_ly, self.T_yr, self.r_lightmin, self.t_yr, self.minutes_per_ly]):
            return (False, "SKIP: missing one or more givens (R_ly, T_yr, r_lightmin, t_yr, minutes_per_ly)")
        if self.M_gal_solar is None:
            return (False, "SKIP: M_gal_solar not available")

        R_lightmin = self.R_ly * self.minutes_per_ly
        expected_ratio = (R_lightmin / self.r_lightmin) ** 3 * (self.t_yr / self.T_yr) ** 2

        if math.isclose(self.M_gal_solar, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M_gal_solar {self.M_gal_solar:.3e} matches expected {expected_ratio:.3e} within tolerance")
        else:
            return (False, f"M_gal_solar {self.M_gal_solar:.3e} differs from expected {expected_ratio:.3e} by >1%")

    def test_invariant_velocity_ratio(self) -> Tuple[bool, str]:
        """
        Alternative invariant: M_gal_solar should equal (R/r) * (V_sun / V_earth)^2,
        where V = 2πR/T for each orbit.
        """
        if any(v is None for v in [self.R_ly, self.T_yr, self.r_lightmin, self.t_yr, self.minutes_per_ly]):
            return (False, "SKIP: missing one or more givens")
        if self.M_gal_solar is None:
            return (False, "SKIP: M_gal_solar not available")

        R_lightmin = self.R_ly * self.minutes_per_ly
        # Orbital speeds in light-minutes per year
        V_sun = 2.0 * math.pi * R_lightmin / self.T_yr
        V_earth = 2.0 * math.pi * self.r_lightmin / self.t_yr
        # Expected from theoretical derivation
        expected = (R_lightmin / self.r_lightmin) * (V_sun / V_earth) ** 2

        if math.isclose(self.M_gal_solar, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M_gal_solar {self.M_gal_solar:.3e} matches velocity-ratio prediction {expected:.3e}")
        else:
            return (False, f"M_gal_solar {self.M_gal_solar:.3e} differs from velocity-ratio prediction {expected:.3e}")

    # --------------------------------------------------------------------------
    # Modeling / Trace Checks (at least two)
    # --------------------------------------------------------------------------
    def test_modeling_unit_conversion(self) -> Tuple[bool, str]:
        """
        Check that the unit conversion factor minutes_per_ly is approximately 525960.
        This ensures the modeling step converting light-years to light-minutes used the correct factor.
        """
        if self.minutes_per_ly is None:
            return (False, "SKIP: minutes_per_ly missing")
        expected_min_per_ly = 525960.0
        if math.isclose(self.minutes_per_ly, expected_min_per_ly, rel_tol=1e-2):
            return (True, "minutes_per_ly is approximately 525960")
        else:
            return (False, f"minutes_per_ly = {self.minutes_per_ly} is not close to 525960")

    def test_modeling_orbital_speeds_subluminal(self) -> Tuple[bool, str]:
        """
        Modeling quality check: the orbital speeds of both the Earth and the Sun
        (computed from given radii and periods) must be significantly less than
        the speed of light, justifying the use of Newtonian gravity.
        Speed of light in light-minutes per year = minutes_per_ly (since 1 light-minute per minute).
        """
        if any(v is None for v in [self.R_ly, self.T_yr, self.r_lightmin, self.t_yr, self.minutes_per_ly]):
            return (False, "SKIP: missing givens to compute orbital speeds")

        R_lightmin = self.R_ly * self.minutes_per_ly
        v_earth = 2.0 * math.pi * self.r_lightmin / self.t_yr   # light-minutes per year
        v_sun   = 2.0 * math.pi * R_lightmin / self.T_yr        # light-minutes per year
        c_ly_yr = self.minutes_per_ly  # speed of light in light-minutes per year

        if v_earth < c_ly_yr and v_sun < c_ly_yr:
            return (True, f"Earth speed {v_earth:.2f} and Sun speed {v_sun:.2f} are << c ({c_ly_yr:.2f})")
        else:
            return (False, f"Orbital speeds exceed c: Earth {v_earth:.2f}, Sun {v_sun:.2f}")

    # --------------------------------------------------------------------------
    # Monotonicity Checks
    # --------------------------------------------------------------------------
    def test_monotonicity_galaxy_mass_increases_with_R(self) -> Tuple[bool, str]:
        """
        With other parameters fixed, increasing the Sun's orbital radius R should
        increase the galaxy mass estimate (since M ∝ R^3).
        We perturb R by +10% and recompute the expected ratio.
        """
        if any(v is None for v in [self.R_ly, self.T_yr, self.r_lightmin, self.t_yr, self.minutes_per_ly]):
            return (False, "SKIP: missing givens to compute monotonicity")

        # Compute baseline ratio
        R_baseline = self.R_ly * self.minutes_per_ly
        ratio_baseline = (R_baseline / self.r_lightmin) ** 3 * (self.t_yr / self.T_yr) ** 2

        # Perturb R by +10%
        R_perturbed = (self.R_ly * 1.10) * self.minutes_per_ly
        ratio_perturbed = (R_perturbed / self.r_lightmin) ** 3 * (self.t_yr / self.T_yr) ** 2

        if ratio_perturbed > ratio_baseline:
            return (True, f"Perturbed ratio {ratio_perturbed:.3e} > baseline {ratio_baseline:.3e}")
        else:
            return (False, f"Perturbation did not increase: perturbed {ratio_perturbed:.3e} <= baseline {ratio_baseline:.3e}")

    # --------------------------------------------------------------------------
    # Weak Answer Sanity (only one allowed)
    # --------------------------------------------------------------------------
    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity: M_gal_solar should be positive, finite, and within a broad
        order-of-magnitude range (1e10 to 1e12) based on the problem's known answer.
        No isclose, no re-derivation.
        """
        if self.M_gal_solar is None:
            return (False, "SKIP: M_gal_solar not available")
        if not math.isfinite(self.M_gal_solar) or self.M_gal_solar <= 0:
            return (False, "M_gal_solar must be positive and finite")
        # Broad magnitude range: expected ~1.53e11, so [1e10, 1e12] is safe.
        if 1e10 <= self.M_gal_solar <= 1e12:
            return (True, f"M_gal_solar = {self.M_gal_solar:.3e} is in plausible range [1e10, 1e12]")
        else:
            return (False, f"M_gal_solar = {self.M_gal_solar:.3e} is outside plausible range [1e10, 1e12]")

    # --------------------------------------------------------------------------
    # Test Runner
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)