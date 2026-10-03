import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audit checker for a geothermal heat engine problem.
    Checks:
      - Interface keys exist
      - Physical invariants (positive work, temperature conversion, Carnot efficiency)
      - Modeling consistency (re-derivation from first principles)
      - Monotonicity (work increases with initial temperature)
      - Weak answer sanity (magnitude range)
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely - assume they are stored as attributes or in results
        self.V_km3 = None  # volume in km^3
        self.rho = None    # density kg/m^3
        self.C = None      # specific heat J/(kg*K)
        self.T_initial_C = None   # °C
        self.T_final_C = None     # °C
        self.T_atmosphere_C = None # °C

        # The result key expected from pseudocode
        self.W_kWh = None

        # Try to extract from solution_instance's __dict__, results dict, or attributes
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            # result key
            self.W_kWh = solution_instance.results.get('W_kWh', None)
            # givens may also be in results
            self.V_km3 = solution_instance.results.get('V', None)
            self.rho = solution_instance.results.get('rho', None)
            self.C = solution_instance.results.get('C', None)
            self.T_initial_C = solution_instance.results.get('T_c_degC', None) or solution_instance.results.get('T_initial_C', None)
            self.T_final_C = solution_instance.results.get('T_f_degC', None) or solution_instance.results.get('T_final_C', None)
            self.T_atmosphere_C = solution_instance.results.get('T_a_degC', None) or solution_instance.results.get('T_atmosphere_C', None)
        else:
            # Fallback to direct attributes
            self.V_km3 = getattr(solution_instance, 'V', None)
            self.rho = getattr(solution_instance, 'rho', None)
            self.C = getattr(solution_instance, 'C', None)
            self.T_initial_C = getattr(solution_instance, 'T_c_degC', None) or getattr(solution_instance, 'T_initial_C', None)
            self.T_final_C = getattr(solution_instance, 'T_f_degC', None) or getattr(solution_instance, 'T_final_C', None)
            self.T_atmosphere_C = getattr(solution_instance, 'T_a_degC', None) or getattr(solution_instance, 'T_atmosphere_C', None)
            # result may be stored as attribute
            if self.W_kWh is None:
                self.W_kWh = getattr(solution_instance, 'W_kWh', None)

        # Derived quantities (computed inside checker for tests)
        self.V_m3 = None     # volume in m^3
        self.m = None        # mass kg
        self.Ti = None       # initial temp in K
        self.Tf = None       # final temp in K
        self.Ta = None       # atmosphere temp in K

        if self.V_km3 is not None:
            self.V_m3 = self.V_km3 * 1e9  # 1 km^3 = 1e9 m^3
        if self.rho is not None and self.V_m3 is not None:
            self.m = self.rho * self.V_m3
        if self.T_initial_C is not None:
            self.Ti = self.T_initial_C + 273.15
        if self.T_final_C is not None:
            self.Tf = self.T_final_C + 273.15
        if self.T_atmosphere_C is not None:
            self.Ta = self.T_atmosphere_C + 273.15

    # --------------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------------
    def test_interface_result_key_exists(self) -> Tuple[bool, str]:
        """Check that the expected result key 'W_kWh' exists and is numeric."""
        if self.W_kWh is None:
            return (False, "Missing result key 'W_kWh' in solution_instance")
        if not isinstance(self.W_kWh, (int, float)):
            return (False, f"'W_kWh' is not numeric, type: {type(self.W_kWh)}")
        return (True, "Result key 'W_kWh' present and numeric")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that all required givens are present (volume, density, specific heat, temperatures)."""
        missing = []
        if self.V_km3 is None:
            missing.append("V (volume in km^3)")
        if self.rho is None:
            missing.append("rho (density)")
        if self.C is None:
            missing.append("C (specific heat)")
        if self.T_initial_C is None:
            missing.append("T_initial_C (initial rock temperature in °C)")
        if self.T_final_C is None:
            missing.append("T_final_C (final rock temperature in °C)")
        if self.T_atmosphere_C is None:
            missing.append("T_atmosphere_C (atmosphere temperature in °C)")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # --------------------------------------------------------------------------
    # Invariant / theorem‑based checks
    # --------------------------------------------------------------------------
    def test_invariant_positive_work(self) -> Tuple[bool, str]:
        """Work output must be positive (engine does work)."""
        if self.W_kWh is None:
            return (False, "SKIP: W_kWh not available")
        if self.W_kWh <= 0:
            return (False, f"Work is not positive: {self.W_kWh} kWh")
        return (True, f"Work is positive: {self.W_kWh} kWh")

    def test_invariant_temperatures_valid(self) -> Tuple[bool, str]:
        """Check that temperatures are physically meaningful (Kelvin > 0, initial > final > atmosphere)."""
        if self.Ti is None or self.Tf is None or self.Ta is None:
            return (False, "SKIP: Kelvin temperatures not computable")
        if self.Ti <= 0:
            return (False, f"Initial temp (K) not positive: {self.Ti}")
        if self.Tf <= 0:
            return (False, f"Final temp (K) not positive: {self.Tf}")
        if self.Ta <= 0:
            return (False, f"Atmosphere temp (K) not positive: {self.Ta}")
        if self.Ti <= self.Tf:
            return (False, f"Initial temp {self.Ti} K <= final {self.Tf} K (rock must cool)")
        if self.Tf <= self.Ta:
            return (False, f"Final temp {self.Tf} K <= atmosphere {self.Ta} K (engine would stop)")
        return (True, "Temperatures satisfy physical ordering")

    def test_invariant_carnot_efficiency_bounds(self) -> Tuple[bool, str]:
        """Carnot efficiency must lie between 0 and 1 for each infinitesimal step (crude check)."""
        if self.Ti is None or self.Tf is None or self.Ta is None:
            return (False, "SKIP: temps not available")
        # efficiency at initial and final (the extremes)
        eta_initial = 1 - self.Ta / self.Ti
        eta_final = 1 - self.Ta / self.Tf
        if not (0 < eta_initial < 1):
            return (False, f"Initial efficiency {eta_initial:.4f} not in (0,1)")
        if not (0 < eta_final < 1):
            return (False, f"Final efficiency {eta_final:.4f} not in (0,1)")
        if eta_final <= 0:
            return (False, "Final Carnot efficiency non‑positive (engine would stop)")
        return (True, "Carnot efficiencies in valid range (0,1)")

    # --------------------------------------------------------------------------
    # Modeling / trace consistency checks
    # --------------------------------------------------------------------------
    def test_modeling_recompute_work_from_formula(self) -> Tuple[bool, str]:
        """
        Re‑compute the total work using the integrated Carnot formula.
        Compare with the stored result using isclose (lenient tolerance).
        This test is allowed because we derive the value inside the checker.
        """
        # Need all givens and the result
        if self.m is None or self.C is None or self.Ti is None or self.Tf is None or self.Ta is None:
            return (False, "SKIP: required quantities for re‑computation missing")
        if self.W_kWh is None:
            return (False, "SKIP: W_kWh missing")

        # Compute work in Joules
        delta_T = self.Ti - self.Tf
        try:
            ln_ratio = math.log(self.Ti / self.Tf)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Cannot compute log ratio: {e}")
        term = delta_T - self.Ta * ln_ratio
        W_J = self.m * self.C * term

        # Convert to kWh: 1 kWh = 3.6e6 J
        W_kWh_computed = W_J / 3.6e6

        # Compare with result
        if not math.isclose(self.W_kWh, W_kWh_computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, (f"Re‑computed work ({W_kWh_computed:.6e} kWh) "
                            f"does not match stored ({self.W_kWh:.6e} kWh) within 1%"))
        return (True, f"Re‑computed work {W_kWh_computed:.6e} kWh agrees with stored value")

    def test_modeling_mass_calculation(self) -> Tuple[bool, str]:
        """Check that mass = density * volume is used consistently."""
        if self.m is None:
            return (False, "SKIP: mass not computed")
        # We cannot know the exact mass, but we can check that if both rho and V are given,
        # the stored mass (if present) is consistent. Here we just check that mass is positive.
        if self.m <= 0:
            return (False, f"Mass must be positive, got {self.m}")
        return (True, f"Mass is positive: {self.m:.6e} kg")

    # --------------------------------------------------------------------------
    # Monotonicity test
    # --------------------------------------------------------------------------
    def test_monotonicity_initial_temp_increases_work(self) -> Tuple[bool, str]:
        """
        If initial temperature increases (keeping others fixed), total work should increase.
        We simulate a small perturbation by recomputing with a slightly higher Ti.
        """
        # Need base values
        if self.m is None or self.C is None or self.Ti is None or self.Tf is None or self.Ta is None:
            return (False, "SKIP: missing base quantities")
        if self.W_kWh is None:
            return (False, "SKIP: W_kWh missing")

        # Perturb Ti upward by 1%
        Ti_perturbed = self.Ti * 1.01   # +1%
        # Re‑compute work with perturbed Ti (keeping Tf, Ta same)
        delta_T = Ti_perturbed - self.Tf
        if delta_T <= 0:
            return (False, "Perturbed Ti < Tf, test invalid")
        try:
            ln_ratio = math.log(Ti_perturbed / self.Tf)
        except (ValueError, ZeroDivisionError):
            return (False, "Cannot compute log for perturbed ratio")
        term = delta_T - self.Ta * ln_ratio
        W_J_pert = self.m * self.C * term
        W_kWh_pert = W_J_pert / 3.6e6

        if W_kWh_pert <= self.W_kWh:
            return (False, (f"Perturbed work {W_kWh_pert:.6e} kWh not > base {self.W_kWh:.6e} kWh "
                            "despite higher initial temperature"))
        return (True, f"Monotonicity holds: perturbed work {W_kWh_pert:.6e} > base {self.W_kWh:.6e} kWh")

    # --------------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # --------------------------------------------------------------------------
    def test_answer_sanity_magnitude_range(self) -> Tuple[bool, str]:
        """Check that the answer is in a plausible range (1e12 to 1e14 kWh)."""
        if self.W_kWh is None:
            return (False, "SKIP: W_kWh missing")
        if not (1e12 <= self.W_kWh <= 1e14):
            return (False, f"Answer {self.W_kWh} kWh outside plausible range [1e12, 1e14]")
        return (True, f"Answer {self.W_kWh} kWh within plausible order of magnitude")

    # --------------------------------------------------------------------------
    # Main runner
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return (all_pass, logs)."""
        logs = []
        all_pass = True
        # Get all methods that start with 'test_'
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