import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Auditor for EM wave dispersion in metals (free-electron model).
    Tests focus on physics invariants, modeling consistency, monotonicity,
    and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract inputs and results safely with None defaults.
        Expects solution_instance to have attributes or keys: results, givens, etc.
        """
        # Safe extraction from results dict
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.k_sq = results.get('k_sq', None)
        self.omega_p = results.get('omega_p', None)
        self.omega_th = results.get('omega_th', None)

        # Also extract givens if available (for self-consistency checks)
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}

        self.n = givens.get('n', None)
        self.e = givens.get('e', None)
        self.m = givens.get('m', None)
        self.eps0 = givens.get('eps0', None)
        self.c = givens.get('c', None)
        self.omega = givens.get('omega', None)

        # For monotonicity, we need a base omega (if not given, try to infer from results)
        if self.omega is None and self.omega_p is not None and self.k_sq is not None:
            # try to extract omega from dispersion relation (if k_sq and omega_p are set)
            # but this is heuristic; best to avoid and just skip monotonicity if missing
            pass

        # Store explanation steps if present (for traceability, not for hardcoding)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all three expected result keys exist and are not None."""
        required = ['k_sq', 'omega_p', 'omega_th']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_numeric_values(self) -> Tuple[bool, str]:
        """Check that provided results are numeric (int or float)."""
        numerics = ['k_sq', 'omega_p', 'omega_th']
        bad = []
        for name in numerics:
            val = getattr(self, name)
            if val is not None and not isinstance(val, (int, float)):
                bad.append(name)
        if bad:
            return (False, f"Non-numeric keys: {bad}")
        return (True, "All result values are numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (use math.isclose with lenient tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_dispersion_relation(self) -> Tuple[bool, str]:
        """Check that k_sq = (omega^2/c^2)*(1 - omega_p^2/omega^2)."""
        if any(val is None for val in [self.k_sq, self.omega_p, self.omega, self.c]):
            return (False, "SKIP: missing required variable for dispersion check")

        # Compute expected k_sq from dispersion relation
        expected_k_sq = (self.omega**2 / self.c**2) * (1 - self.omega_p**2 / self.omega**2)

        if math.isclose(self.k_sq, expected_k_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Dispersion relation satisfied")
        else:
            return (False, f"k_sq {self.k_sq} != expected {expected_k_sq}")

    def test_invariant_omega_th_equals_omega_p(self) -> Tuple[bool, str]:
        """Check that threshold frequency equals plasma frequency."""
        if self.omega_th is None or self.omega_p is None:
            return (False, "SKIP: omega_th or omega_p missing")

        if math.isclose(self.omega_th, self.omega_p, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "omega_th == omega_p")
        else:
            return (False, f"omega_th {self.omega_th} != omega_p {self.omega_p}")

    def test_invariant_plasma_frequency_formula(self) -> Tuple[bool, str]:
        """Check that omega_p^2 = n*e^2/(m*eps0)."""
        if any(val is None for val in [self.omega_p, self.n, self.e, self.m, self.eps0]):
            return (False, "SKIP: missing givens for plasma frequency formula")

        expected = math.sqrt(self.n * self.e**2 / (self.m * self.eps0))
        if math.isclose(self.omega_p, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Plasma frequency formula holds")
        else:
            return (False, f"omega_p {self.omega_p} != expected {expected}")

    def test_invariant_sign_of_k_sq_above_threshold(self) -> Tuple[bool, str]:
        """For omega > omega_p, k_sq should be positive."""
        if self.omega is None or self.omega_p is None or self.k_sq is None:
            return (False, "SKIP: missing data for sign check")

        if self.omega > self.omega_p:
            if self.k_sq > 0:
                return (True, "k_sq positive above threshold")
            else:
                return (False, f"Expected k_sq > 0 but got {self.k_sq}")
        elif self.omega < self.omega_p:
            # below threshold k_sq should be negative (evanescent)
            if self.k_sq < 0:
                return (True, "k_sq negative below threshold")
            else:
                return (False, f"Expected k_sq < 0 but got {self.k_sq}")
        else:  # omega == omega_p exactly
            if math.isclose(self.k_sq, 0, abs_tol=1e-2):
                return (True, "k_sq ~ 0 at threshold")
            else:
                return (False, f"At threshold, expected k_sq ~ 0, got {self.k_sq}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (check consistency with explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_dielectric_limit(self) -> Tuple[bool, str]:
        """Check that when omega >> omega_p, k_sq ~ omega^2/c^2."""
        if any(val is None for val in [self.k_sq, self.omega_p, self.omega, self.c]):
            return (False, "SKIP: missing data for high frequency limit")

        # If omega is more than 3x omega_p, the factor (1 - (omega_p/omega)^2) should be close to 1
        if self.omega > 3 * self.omega_p:
            expected = self.omega**2 / self.c**2
            # k_sq should be close to expected; tolerance 5% because of rounding
            if math.isclose(self.k_sq, expected, rel_tol=5e-2, abs_tol=1e-2):
                return (True, "High frequency limit matches vacuum dispersion")
            else:
                return (False, f"High frequency: k_sq {self.k_sq}, expected ~ {expected}")
        else:
            return (True, "SKIP: not in high-frequency regime")

    def test_modeling_conductivity_form(self) -> Tuple[bool, str]:
        """Check consistency of derived conductivity if available.
        Not strictly required but checks modeling depth.
        """
        # This test is weak; only if the solution provides sigma.
        # Since the pseudocode mentions sigma, we might look for it.
        # We'll extract sigma if it exists in results.
        sigma = getattr(self, 'sigma', None)
        if sigma is None:
            # maybe store from results during init? We'll add safe extraction in init
            results = getattr(self, 'results_maybe', {})  # fallback
            sigma = results.get('sigma', None)
            if sigma is None:
                return (True, "SKIP: conductivity not provided")

        # Conductivity σ = i n e^2 / (ω m) (without damping)
        if any(val is None for val in [self.n, self.e, self.m, self.omega]):
            return (False, "SKIP: missing givens for conductivity check")

        # Compute expected conductivity magnitude (since it's complex, we check magnitude)
        expected_mag = self.n * self.e**2 / (self.omega * self.m)
        if hasattr(sigma, '__abs__'):
            sigma_mag = abs(sigma)
        else:
            sigma_mag = abs(complex(sigma)) if isinstance(sigma, complex) else abs(sigma)

        if math.isclose(sigma_mag, expected_mag, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Conductivity magnitude consistent")
        else:
            return (False, f"Sigma magnitude {sigma_mag} != {expected_mag}")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_k_sq_with_omega(self) -> Tuple[bool, str]:
        """Increase omega by 10% and verify k_sq increases."""
        if any(val is None for val in [self.omega, self.omega_p, self.c]):
            return (False, "SKIP: missing base data for monotonicity")

        # Use the dispersion relation to compute perturbed k_sq
        omega_pert = self.omega * 1.10
        k_sq_pert = (omega_pert**2 / self.c**2) * (1 - self.omega_p**2 / omega_pert**2)

        # We only have original k_sq from results; compare
        if self.k_sq is None:
            return (False, "SKIP: no baseline k_sq to compare")

        if k_sq_pert > self.k_sq:
            return (True, "k_sq increases when omega increases (monotonic)")
        else:
            return (False, f"Perturbed k_sq {k_sq_pert} not > baseline {self.k_sq}")

    def test_monotonicity_k_sq_with_omega_p(self) -> Tuple[bool, str]:
        """Increase omega_p by 5% and verify k_sq decreases (since higher plasma density reduces propagation)."""
        if any(val is None for val in [self.omega, self.omega_p, self.c, self.k_sq]):
            return (False, "SKIP: missing data for omega_p monotonicity")

        omega_p_pert = self.omega_p * 1.05
        # recompute k_sq with perturbed omega_p
        k_sq_pert = (self.omega**2 / self.c**2) * (1 - omega_p_pert**2 / self.omega**2)

        # For omega > omega_p, increasing omega_p reduces k_sq.
        if k_sq_pert < self.k_sq:
            return (True, "k_sq decreases when omega_p increases (monotonic)")
        else:
            return (False, f"Perturbed k_sq {k_sq_pert} not < baseline {self.k_sq}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basics(self) -> Tuple[bool, str]:
        """Weak sanity: all results are finite, positive (k_sq can be negative though),
        and omega_p, omega_th are positive.
        """
        # Check finiteness
        for name in ['k_sq', 'omega_p', 'omega_th']:
            val = getattr(self, name)
            if val is None:
                return (False, f"Missing {name}")
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")

        # omega_p and omega_th must be positive
        for name in ['omega_p', 'omega_th']:
            val = getattr(self, name)
            if val <= 0:
                return (False, f"{name} is not positive: {val}")

        # k_sq can be any real, but should be consistent with sign condition already checked.
        # No further checks.
        return (True, "Basic sanity passed: finite, positive frequencies")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return (all_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# Example usage (not part of required output, but for debugging)
def demo():
    class DummySolution:
        def __init__(self):
            self.results = {
                'k_sq': 1.0e12,
                'omega_p': 5.0e15,
                'omega_th': 5.0e15,
                'sigma': complex(0, 1e7)  # optional
            }
            self.givens = {
                'n': 1e28,
                'e': 1.6e-19,
                'm': 9.11e-31,
                'eps0': 8.85e-12,
                'c': 3e8,
                'omega': 1e16
            }
            self.explanation_steps = "Steps..."
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")

if __name__ == "__main__":
    demo()