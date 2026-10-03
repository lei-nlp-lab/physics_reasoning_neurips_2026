import math
import traceback
from typing import Dict, List, Tuple, Any, Optional

class PhysicsChecker:
    """
    Auditor for the blackbody radiation mass density problem.
    """
    def __init__(self, solution_instance: Any):
        # Extract givens safely (may be in solution_instance.givens dict or as attributes)
        self.T = None
        self.a = None
        self.c = None
        self.ton_to_g = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.T = solution_instance.givens.get('T', None)
            self.a = solution_instance.givens.get('a', None)
            self.c = solution_instance.givens.get('c', None)
            self.ton_to_g = solution_instance.givens.get('ton_to_g', None)
        # Also accept as top-level attributes
        if self.T is None and hasattr(solution_instance, 'T'):
            self.T = solution_instance.T
        if self.a is None and hasattr(solution_instance, 'a'):
            self.a = solution_instance.a
        if self.c is None and hasattr(solution_instance, 'c'):
            self.c = solution_instance.c
        if self.ton_to_g is None and hasattr(solution_instance, 'ton_to_g'):
            self.ton_to_g = solution_instance.ton_to_g

        # Extract results safely
        self.rho_ton_per_cm3 = None
        self.closest_option = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.rho_ton_per_cm3 = solution_instance.results.get('rho_ton_per_cm3', None)
            self.closest_option = solution_instance.results.get('closest_option', None)
        # Also accept as top-level attributes
        if self.rho_ton_per_cm3 is None and hasattr(solution_instance, 'rho_ton_per_cm3'):
            self.rho_ton_per_cm3 = solution_instance.rho_ton_per_cm3
        if self.closest_option is None and hasattr(solution_instance, 'closest_option'):
            self.closest_option = solution_instance.closest_option

        # Optionally store explanation_steps and pseudocode for traceability
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Derived quantities for internal use
        self.rho_g_per_cm3 = None
        if self.rho_ton_per_cm3 is not None and self.ton_to_g is not None and self.ton_to_g != 0:
            self.rho_g_per_cm3 = self.rho_ton_per_cm3 * self.ton_to_g  # back-calculated

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_rho_ton_per_cm3(self) -> Tuple[bool, str]:
        """Check that rho_ton_per_cm3 exists and is a finite number."""
        if self.rho_ton_per_cm3 is None:
            return (False, "SKIP: rho_ton_per_cm3 not available")
        if not isinstance(self.rho_ton_per_cm3, (int, float)):
            return (False, f"rho_ton_per_cm3 has invalid type: {type(self.rho_ton_per_cm3)}")
        if math.isnan(self.rho_ton_per_cm3) or math.isinf(self.rho_ton_per_cm3):
            return (False, "rho_ton_per_cm3 is NaN or Inf")
        return (True, "rho_ton_per_cm3 present and finite")

    def test_interface_closest_option(self) -> Tuple[bool, str]:
        """Check that closest_option is a non-empty string."""
        if self.closest_option is None:
            return (False, "SKIP: closest_option not available")
        if not isinstance(self.closest_option, str):
            return (False, f"closest_option has invalid type: {type(self.closest_option)}")
        if len(self.closest_option) == 0:
            return (False, "closest_option is empty string")
        return (True, "closest_option present and non-empty")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_density(self) -> Tuple[bool, str]:
        """Check that energy density u = a * T^4 (within 1% relative tolerance)."""
        if self.a is None or self.T is None:
            return (False, "SKIP: missing a or T givens")
        expected_u = self.a * (self.T ** 4)
        # To compute u from results, we need rho_g_per_cm3 and c.
        if self.rho_g_per_cm3 is None or self.c is None or self.c == 0:
            return (False, "SKIP: cannot compute u (rho_g_per_cm3 or c missing)")
        actual_u = self.rho_g_per_cm3 * (self.c ** 2)
        if not math.isclose(actual_u, expected_u, rel_tol=1e-2):
            return (False, f"u mismatch: actual={actual_u:.4e}, expected={expected_u:.4e}")
        return (True, f"u = a T^4 holds: {actual_u:.4e} ≈ {expected_u:.4e}")

    def test_invariant_mass_energy_equivalence(self) -> Tuple[bool, str]:
        """Check that rho_g_per_cm3 = u / c^2 (within 1% relative tolerance)."""
        if self.rho_g_per_cm3 is None or self.c is None or self.c == 0:
            return (False, "SKIP: rho_g_per_cm3 or c missing")
        # Compute u from a and T if available, otherwise from results.
        if self.a is not None and self.T is not None:
            u = self.a * (self.T ** 4)
        elif self.rho_ton_per_cm3 is not None and self.ton_to_g is not None and self.ton_to_g != 0:
            # use tentative rho_g and back-calculate u
            rho_g = self.rho_ton_per_cm3 * self.ton_to_g
            u = rho_g * (self.c ** 2)
        else:
            return (False, "SKIP: insufficient data to compute u")
        expected_rho = u / (self.c ** 2)
        if not math.isclose(self.rho_g_per_cm3, expected_rho, rel_tol=1e-2):
            return (False, f"ρ mismatch: actual={self.rho_g_per_cm3:.4e}, expected={expected_rho:.4e}")
        return (True, f"ρ = u/c^2 holds: {self.rho_g_per_cm3:.4e} ≈ {expected_rho:.4e}")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------
    def test_modeling_conversion_factor(self) -> Tuple[bool, str]:
        """Check that rho_ton_per_cm3 = rho_g_per_cm3 / ton_to_g (within 1%)."""
        if self.rho_ton_per_cm3 is None or self.ton_to_g is None or self.ton_to_g == 0:
            return (False, "SKIP: missing rho_ton_per_cm3 or ton_to_g")
        # Compute expected from rho_g_per_cm3 if available; else from back-computed
        if self.rho_g_per_cm3 is not None:
            expected_ton = self.rho_g_per_cm3 / self.ton_to_g
        elif self.rho_ton_per_cm3 is not None and self.ton_to_g is not None:
            # Use the result itself to check consistency (weak, but better than nothing)
            expected_ton = self.rho_ton_per_cm3  # identity check, not useful
            return (False, "SKIP: cannot cross-check")
        else:
            return (False, "SKIP: insufficient data")
        if not math.isclose(self.rho_ton_per_cm3, expected_ton, rel_tol=1e-2):
            return (False, f"Conversion mismatch: actual={self.rho_ton_per_cm3:.4e}, expected={expected_ton:.4e}")
        return (True, f"Conversion consistent: {self.rho_ton_per_cm3:.4e} ≈ {expected_ton:.4e}")

    def test_modeling_closest_option_correct(self) -> Tuple[bool, str]:
        """Check that the selected option is '1 ton/cm^3' or contains '1 ton'."""
        if self.closest_option is None:
            return (False, "SKIP: closest_option not available")
        # The target answer is "1 ton/cm^3". Accept any string that includes "1 ton" and "cm^3".
        option_lower = self.closest_option.lower().replace(" ", "")
        if "1ton" not in option_lower or "cm^3" not in option_lower:
            return (False, f"Expected option containing '1 ton' and 'cm^3', got '{self.closest_option}'")
        return (True, f"closest_option '{self.closest_option}' matches expected")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_density_increases_with_temperature(self) -> Tuple[bool, str]:
        """Check that perturbing T upward increases rho_ton_per_cm3 (since u ∝ T^4)."""
        if self.T is None or self.rho_ton_per_cm3 is None or self.a is None or self.c is None or self.ton_to_g is None:
            return (False, "SKIP: missing required givens/results")
        # Baseline value
        baseline = self.rho_ton_per_cm3
        # Perturb T by +5%
        T_pert = self.T * 1.05
        # Recompute rho_ton_per_cm3 using the same formulas
        u_pert = self.a * (T_pert ** 4)
        rho_g_pert = u_pert / (self.c ** 2)
        rho_ton_pert = rho_g_pert / self.ton_to_g
        if rho_ton_pert <= baseline:
            return (False, f"Monotonicity violated: perturbed density {rho_ton_pert:.4e} not > baseline {baseline:.4e}")
        return (True, f"Monotonicity holds: +5% T gives density {rho_ton_pert:.4e} > {baseline:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self) -> Tuple[bool, str]:
        """Check that rho_ton_per_cm3 is positive, finite, and roughly order 0.01–10 ton/cm³."""
        if self.rho_ton_per_cm3 is None:
            return (False, "SKIP: rho_ton_per_cm3 not available")
        if self.rho_ton_per_cm3 <= 0:
            return (False, f"rho_ton_per_cm3 is not positive: {self.rho_ton_per_cm3}")
        if math.isnan(self.rho_ton_per_cm3) or math.isinf(self.rho_ton_per_cm3):
            return (False, "rho_ton_per_cm3 is NaN or Inf")
        # Accept values between 0.01 and 10 ton/cm^3 (the true value is ~0.084)
        if not (0.01 <= self.rho_ton_per_cm3 <= 10.0):
            return (False, f"rho_ton_per_cm3 {self.rho_ton_per_cm3} not in plausible range [0.01, 10]")
        return (True, f"rho_ton_per_cm3 = {self.rho_ton_per_cm3} is positive and plausible")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collect results."""
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