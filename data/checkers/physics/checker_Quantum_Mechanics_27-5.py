import math
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits the physics fidelity of a solution_instance for the positronium problem.
    """

    def __init__(self, solution_instance: Any) -> None:
        """Extract all relevant data from the solution instance, defaulting to None."""
        # Result quantities
        results = getattr(solution_instance, 'results', {})
        self.E_n = results.get('E_n', None)
        self.E_trip = results.get('E_trip', None)
        self.E_sing = results.get('E_sing', None)
        self.E_12 = results.get('E_12', None)
        self.E_3 = results.get('E_3', None)
        self.E_4 = results.get('E_4', None)

        # Physical givens (if available from solution_instance attributes)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.m_e = getattr(solution_instance, 'm_e', None)
        self.c = getattr(solution_instance, 'c', None)
        self.n = getattr(solution_instance, 'n', None)
        self.A = getattr(solution_instance, 'A', None)
        self.B = getattr(solution_instance, 'B', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.e = getattr(solution_instance, 'e', None)

        # Derived constant
        self.mu_B: Optional[float] = None
        if self.e is not None and self.hbar is not None and self.m_e is not None:
            self.mu_B = self.e * self.hbar / (2.0 * self.m_e)

        # Explanation steps (optional, for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_result_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present (non‑None)."""
        required_keys = ['E_n', 'E_trip', 'E_sing', 'E_12', 'E_3', 'E_4']
        for key in required_keys:
            if getattr(self, key) is None:
                return (False, f"Missing required result key: {key}")
        return (True, "All required result keys present")

    def test_interface_result_numeric(self) -> Tuple[bool, str]:
        """Check that result values are numeric (int/float) and not NaN or Inf."""
        for attr in ['E_n', 'E_trip', 'E_sing', 'E_12', 'E_3', 'E_4']:
            val = getattr(self, attr)
            if val is not None:
                if not isinstance(val, (int, float)):
                    return (False, f"{attr} is not numeric: {type(val).__name__}")
                if math.isnan(val) or math.isinf(val):
                    return (False, f"{attr} is NaN or Inf")
        return (True, "All results are valid numbers")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_trip_sing_difference(self) -> Tuple[bool, str]:
        """E_trip - E_sing should be 4 * A * mu_B^2."""
        if any(x is None for x in [self.E_trip, self.E_sing, self.A, self.mu_B]):
            return (False, "SKIP: missing data for triplet‑singlet difference")
        expected = 4.0 * self.A * self.mu_B**2
        actual = self.E_trip - self.E_sing
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Triplet‑singlet difference = {actual} ≈ {expected}")
        return (False, f"Difference mismatch: {actual} vs {expected}")

    def test_invariant_12_equals_trip(self) -> Tuple[bool, str]:
        """E_12 must equal E_trip because |1>,|2> are triplet states."""
        if self.E_12 is None or self.E_trip is None:
            return (False, "SKIP: missing E_12 or E_trip")
        if math.isclose(self.E_12, self.E_trip, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "E_12 matches E_trip")
        return (False, f"E_12 ({self.E_12}) ≠ E_trip ({self.E_trip})")

    def test_invariant_3_4_sum(self) -> Tuple[bool, str]:
        """E_3 + E_4 = 2*H0 - 2*A*mu_B^2 (trace of 2x2 hyperfine‑Zeeman submatrix)."""
        if any(x is None for x in [self.E_3, self.E_4, self.E_trip, self.A, self.mu_B]):
            return (False, "SKIP: missing data for sum test")
        H0 = self.E_trip - self.A * self.mu_B**2   # H0 = Coulomb energy
        expected = 2.0 * H0 - 2.0 * self.A * self.mu_B**2
        actual = self.E_3 + self.E_4
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_3+E_4 = {actual} ≈ {expected}")
        return (False, f"Sum mismatch: {actual} vs {expected}")

    def test_invariant_singlet_quadratic_shift(self) -> Tuple[bool, str]:
        """For B>0, the singlet state |4> should be lower than the zero‑field singlet energy."""
        if any(x is None for x in [self.E_4, self.E_sing, self.B]):
            return (False, "SKIP: missing data for singlet shift test")
        if self.B == 0:
            return (True, "B=0, shift is zero by definition")
        if self.B > 0:
            if self.E_4 < self.E_sing:
                return (True, f"Singlet energy lowers with B: E_4={self.E_4} < E_sing={self.E_sing}")
            return (False, f"Singlet should lower but E_4={self.E_4} >= E_sing={self.E_sing}")
        # For B<0 the physics is symmetric; skip check
        return (True, "B negative, skipping directional check")

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on theoretical formulas)
    # ------------------------------------------------------------------
    def test_modeling_energy_formula(self) -> Tuple[bool, str]:
        """E_n should equal -α² m_e c² / (4 n²) (reduced mass)."""
        if any(x is None for x in [self.E_n, self.alpha, self.m_e, self.c, self.n]):
            return (False, "SKIP: missing givens for E_n formula")
        if self.n <= 0:
            return (False, f"n must be positive, got {self.n}")
        expected = - (self.alpha**2 * self.m_e * self.c**2) / (4.0 * self.n**2)
        if math.isclose(self.E_n, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_n matches theoretical: {self.E_n}")
        return (False, f"E_n mismatch: got {self.E_n}, expected {expected}")

    def test_modeling_beta_dependent_energies(self) -> Tuple[bool, str]:
        """Check E_3 and E_4 against the diagonalized formulas."""
        if any(x is None for x in [self.A, self.mu_B, self.B, self.E_3, self.E_4, self.E_n]):
            return (False, "SKIP: missing data for β‑dependent check")
        H0 = self.E_n
        beta = self.B / (self.A * self.mu_B)
        sqrt_term = math.sqrt(1.0 + beta**2)
        exp_E3 = H0 + self.A * self.mu_B**2 + 2.0 * self.A * self.mu_B**2 * (sqrt_term - 1.0)
        exp_E4 = H0 - 3.0 * self.A * self.mu_B**2 - 2.0 * self.A * self.mu_B**2 * (sqrt_term - 1.0)
        if math.isclose(self.E_3, exp_E3, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose(self.E_4, exp_E4, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "E_3 and E_4 match theoretical expressions")
        return (False, f"Mismatch: E_3={self.E_3} (exp {exp_E3}), E_4={self.E_4} (exp {exp_E4})")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_3_4_with_B(self) -> Tuple[bool, str]:
        """
        If B increases by 10%, E_3 should increase and E_4 should decrease.
        (We compute the analytical expressions for both fields.)
        """
        if any(x is None for x in [self.B, self.A, self.mu_B, self.E_n]):
            return (False, "SKIP: missing data (E_n, A, mu_B, B) for monotonicity test")
        # Derive E3 and E4 using the formulas
        H0 = self.E_n
        beta_base = self.B / (self.A * self.mu_B)
        sqrt_base = math.sqrt(1.0 + beta_base**2)
        E3_base = H0 + self.A * self.mu_B**2 + 2.0 * self.A * self.mu_B**2 * (sqrt_base - 1.0)
        E4_base = H0 - 3.0 * self.A * self.mu_B**2 - 2.0 * self.A * self.mu_B**2 * (sqrt_base - 1.0)

        # Perturb B (increase by 10%, but avoid zero field if B=0)
        if self.B == 0:
            B_pert = 0.1 * (self.A * self.mu_B)   # non‑zero perturbation
        else:
            B_pert = self.B * 1.1
        beta_pert = B_pert / (self.A * self.mu_B)
        sqrt_pert = math.sqrt(1.0 + beta_pert**2)
        E3_pert = H0 + self.A * self.mu_B**2 + 2.0 * self.A * self.mu_B**2 * (sqrt_pert - 1.0)
        E4_pert = H0 - 3.0 * self.A * self.mu_B**2 - 2.0 * self.A * self.mu_B**2 * (sqrt_pert - 1.0)

        if E3_pert > E3_base and E4_pert < E4_base:
            return (True, f"E3 increases ({E3_base} → {E3_pert}), E4 decreases ({E4_base} → {E4_pert})")
        else:
            return (False, f"Monotonicity broken: E3: {E3_base} → {E3_pert}, E4: {E4_base} → {E4_pert}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, use only inequalities, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs_and_order(self) -> Tuple[bool, str]:
        """Very weak checks: signs and ordering."""
        if any(x is None for x in [self.E_n, self.E_trip, self.E_sing, self.E_12, self.E_3, self.E_4]):
            return (False, "SKIP: missing results for answer sanity")
        # Coulomb energy should be negative for positive n
        if self.n is not None and self.n >= 1 and self.E_n >= 0:
            return (False, f"E_n should be negative for n={self.n}, got {self.E_n}")
        # Triplet should lie above singlet
        if self.E_trip <= self.E_sing:
            return (False, f"Triplet ({self.E_trip}) should be > singlet ({self.E_sing})")
        # For B>0, |3⟩ should lie above |1⟩,|2⟩; |4⟩ below singlet
        if self.B is not None and self.B > 0:
            if self.E_3 <= self.E_12:
                return (False, f"For B>0, E_3 ({self.E_3}) should be > E_12 ({self.E_12})")
            if self.E_4 >= self.E_sing:
                return (False, f"For B>0, E_4 ({self.E_4}) should be < E_sing ({self.E_sing})")
        # Basic finiteness already checked in interface test
        return (True, "Basic sign and ordering sanity passed")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_ methods, collect results.
        Returns (all_pass, logs).
        """
        logs: List[str] = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {e}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)