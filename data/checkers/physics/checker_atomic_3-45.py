import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    A robust physics checker for neutrino interactions problem.
    Checks interface, invariants, modeling, monotonicity, and provides a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results and givens with None defaults
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        self.givens = getattr(solution_instance, 'givens', {})

        # Extract N_int from results
        self.N_int = self.results.get('N_int', None)

        # Problem constants (from problem statement)
        self.N0 = 1e6          # number of incident neutrinos
        self.sigma = 0.7e-38  # cm^2 per nucleon
        self.rho = 5.0        # g/cm^3
        self.R_cm = 6000.0 * 1e5  # 6000 km in cm = 6e8 cm
        self.N_A = 6.02214076e23  # mol^{-1}

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_N_int(self) -> Tuple[bool, str]:
        """
        Check that the result contains the key 'N_int'.
        """
        if self.N_int is None:
            return (False, "Missing 'N_int' in results")
        return (True, "Key 'N_int' present")

    def test_interface_N_int_is_numeric(self) -> Tuple[bool, str]:
        """
        Check that N_int is a number (int or float).
        """
        if self.N_int is None:
            return (False, "N_int is missing")
        if not isinstance(self.N_int, (int, float)):
            return (False, f"N_int has type {type(self.N_int).__name__}, expected int or float")
        return (True, "N_int is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physics sanity)
    # ------------------------------------------------------------------
    def test_invariant_N_int_positive(self) -> Tuple[bool, str]:
        """
        Number of interactions must be non-negative.
        """
        if self.N_int is None:
            return (False, "SKIP: N_int not available")
        if self.N_int < 0:
            return (False, f"N_int = {self.N_int} is negative")
        return (True, "N_int is non-negative")

    def test_invariant_N_int_not_exceed_incident(self) -> Tuple[bool, str]:
        """
        The number of interactions cannot exceed the number of incident neutrinos.
        """
        if self.N_int is None:
            return (False, "SKIP: N_int not available")
        if self.N_int > self.N0:
            return (False, f"N_int = {self.N_int} exceeds N0 = {self.N0:.1e}")
        return (True, "N_int ≤ N0 satisfied")

    def test_invariant_exponent_small(self) -> Tuple[bool, str]:
        """
        The exponential argument exp(-σ n L) must be very small (≪1)
        to justify linear approximation.
        """
        # Compute n = ρ * N_A  (nucleon number density)
        n = self.rho * self.N_A          # cm^{-3}
        L = 2.0 * self.R_cm              # cm
        exponent = self.sigma * n * L    # dimensionless
        threshold = 0.1                  # Exponent << 1 means <0.1
        if exponent >= threshold:
            return (False, f"Exponent σ n L = {exponent:.2e} is not ≪ 1")
        return (True, f"Exponent {exponent:.2e} is ≪ 1")

    # ------------------------------------------------------------------
    # Modeling/trace tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_interaction_formula(self) -> Tuple[bool, str]:
        """
        Verify that N_int approximately equals N0 * σ * n * L.
        Re-derive expected value from problem constants.
        """
        if self.N_int is None:
            return (False, "SKIP: N_int not available")

        # Compute expected N_int using the linear approximation
        n = self.rho * self.N_A
        L = 2.0 * self.R_cm
        expected = self.N0 * self.sigma * n * L

        # Use fuzzy matching for physics noise
        rel_tol = 0.02   # allow 2% relative error
        abs_tol = 1.0    # allow absolute difference of 1 interaction
        if not math.isclose(self.N_int, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return (False, f"N_int = {self.N_int} but expected ≈ {expected:.1f} "
                           f"(off by {abs(self.N_int - expected):.1f})")
        return (True, f"N_int = {self.N_int} matches N0 σ n L formula (≈ {expected:.1f})")

    def test_modeling_exponential_consistency(self) -> Tuple[bool, str]:
        """
        Check that N_int is consistent with the exact exponential model.
        The exact number of interactions is N0*(1 - exp(-σ n L)).
        For small exponent, the linear approximation should be close.
        Compute the exact value and compare with the given N_int.
        """
        if self.N_int is None:
            return (False, "SKIP: N_int not available")

        n = self.rho * self.N_A
        L = 2.0 * self.R_cm
        exponent = self.sigma * n * L
        exact = self.N0 * (1.0 - math.exp(-exponent))

        # Linear approximation: N_int_lin = N0 * exponent
        linear = self.N0 * exponent

        # If exponent is small, linear and exact are close.
        # We allow a relative tolerance of 10% between linear and exact
        # and also check that the given N_int is between linear and exact (or close to either).
        if exponent < 0.001:
            # In this regime, linear and exact are essentially identical; accept if close to either
            rel_tol = 0.02
            abs_tol = 1.0
            match_linear = math.isclose(self.N_int, linear, rel_tol=rel_tol, abs_tol=abs_tol)
            match_exact = math.isclose(self.N_int, exact, rel_tol=rel_tol, abs_tol=abs_tol)
            if not (match_linear or match_exact):
                return (False, f"N_int = {self.N_int} does not match linear ({linear:.2f}) "
                               f"or exact ({exact:.2f}) values")
            return (True, f"N_int = {self.N_int} consistent with linear and exact models")
        else:
            # For larger exponent, use exact value as reference
            rel_tol = 0.05
            abs_tol = 1.0
            if not math.isclose(self.N_int, exact, rel_tol=rel_tol, abs_tol=abs_tol):
                return (False, f"N_int = {self.N_int} does not match exact model value {exact:.2f}")
            return (True, f"N_int = {self.N_int} matches exact exponential model")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_increase(self) -> Tuple[bool, str]:
        """
        Verify that N_int increases with cross-section sigma.
        Baseline: use given sigma. Perturb: increase sigma by 10%.
        Compute corresponding N_int_perturbed > baseline.
        """
        n = self.rho * self.N_A
        L = 2.0 * self.R_cm
        baseline = self.N0 * self.sigma * n * L
        perturbed_sigma = self.sigma * 1.1
        perturbed = self.N0 * perturbed_sigma * n * L
        if not (perturbed > baseline):
            return (False, f"Perturbed N_int = {perturbed} should be > baseline {baseline}")
        return (True, f"Sigma increase gives N_int increase: {perturbed} > {baseline}")

    # ------------------------------------------------------------------
    # Answer sanity check (weak, broad range)
    # ------------------------------------------------------------------
    def test_answer_sanity_sensible_range(self) -> Tuple[bool, str]:
        """
        Final answer should be a small positive integer (order 10-100).
        Do NOT use isclose or rederive answer.
        """
        if self.N_int is None:
            return (False, "SKIP: N_int not available")

        # Only check sign and broad magnitude
        if not isinstance(self.N_int, (int, float)) or math.isnan(self.N_int):
            return (False, "N_int is NaN or non-numeric")
        if self.N_int <= 0:
            return (False, f"N_int = {self.N_int} is not positive")
        if self.N_int > 1000:
            return (False, f"N_int = {self.N_int} is absurdly large")
        if self.N_int < 1:
            return (False, f"N_int = {self.N_int} is too small")
        return (True, f"N_int = {self.N_int} is in a plausible range (1 - 1000)")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all methods whose name starts with 'test_'.
        Returns (all_pass, list of log messages).
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
            except Exception as exc:
                passed, msg = False, f"{name} CRASHED: {exc}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)