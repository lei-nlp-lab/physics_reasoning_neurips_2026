import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens; assume solution_instance has .givens dict-like
        self.a = None
        self.r = None
        self.q = None
        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            self.a = givens.get('a', None)
            self.r = givens.get('r', None)
            self.q = givens.get('q', None)
        # Extract results
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        self.Q_min_1 = self.results.get('Q_min_1', None)
        self.Q_min_2 = self.results.get('Q_min_2', None)

    def run_all_tests(self):
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

    # ---------- Interface tests ----------
    def test_interface_given_keys(self):
        if self.a is None:
            return (False, "SKIP: 'a' not available in givens")
        if self.r is None:
            return (False, "SKIP: 'r' not available in givens")
        if self.q is None:
            return (False, "SKIP: 'q' not available in givens")
        return (True, "Given keys a, r, q present")

    def test_interface_result_keys(self):
        if self.Q_min_1 is None:
            return (False, "SKIP: 'Q_min_1' not in results")
        if self.Q_min_2 is None:
            return (False, "SKIP: 'Q_min_2' not in results")
        return (True, "Result keys Q_min_1 and Q_min_2 present")

    # ---------- Invariant tests (theorem-based, use math.isclose) ----------
    def test_invariant_sigma_min_zero_sub1(self):
        """For sub1, with Q = Q_min_1, sigma(theta=0) should be zero."""
        if None in (self.a, self.r, self.q, self.Q_min_1):
            return (False, "SKIP: missing a, r, q, or Q_min_1")
        # sigma(0) = (1/(4π a²)) [ Q + a q/r - a q (r² - a²)/(r-a)^3 ]
        Q = self.Q_min_1
        a = self.a
        r = self.r
        q = self.q
        try:
            sigma0 = (1/(4*math.pi*a**2)) * (Q + a*q/r - a*q*(r**2 - a**2)/(r-a)**3)
        except ZeroDivisionError:
            return (False, "Division by zero: r-a = 0")
        if math.isclose(sigma0, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma0 ~= 0 (value={sigma0})")
        else:
            return (False, f"sigma0 = {sigma0}, not close to 0")

    def test_invariant_sigma_min_zero_sub2(self):
        """For sub2, with Q = Q_min_2, sigma(theta=pi) should be zero."""
        if None in (self.a, self.r, self.q, self.Q_min_2):
            return (False, "SKIP: missing a, r, q, or Q_min_2")
        Q = self.Q_min_2
        a = self.a
        r = self.r
        q = self.q
        try:
            sigma_pi = (1/(4*math.pi*a**2)) * (Q - a*q/r + a*q*(r**2 - a**2)/(r+a)**3)
        except ZeroDivisionError:
            return (False, "Division by zero")
        if math.isclose(sigma_pi, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_pi ~= 0 (value={sigma_pi})")
        else:
            return (False, f"sigma_pi = {sigma_pi}, not close to 0")

    def test_invariant_sign_Q_min_1(self):
        """Q_min_1 must be positive for positive q."""
        if self.Q_min_1 is None:
            return (False, "SKIP: Q_min_1 missing")
        if self.Q_min_1 <= 0:
            return (False, f"Q_min_1 = {self.Q_min_1} is not positive")
        return (True, f"Q_min_1 = {self.Q_min_1} > 0")

    def test_invariant_sign_Q_min_2(self):
        """Q_min_2 must be positive (derivation yields positive)."""
        if self.Q_min_2 is None:
            return (False, "SKIP: Q_min_2 missing")
        if self.Q_min_2 <= 0:
            return (False, f"Q_min_2 = {self.Q_min_2} is not positive")
        return (True, f"Q_min_2 = {self.Q_min_2} > 0")

    # ---------- Modeling/trace tests (consistency with derivation) ----------
    def test_trace_Q_min_1_formula(self):
        """Check that Q_min_1 matches the derived formula a^2 q (3r - a) / (r (r-a)^2)."""
        if None in (self.a, self.r, self.q, self.Q_min_1):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        try:
            expected = a**2 * q * (3*r - a) / (r * (r - a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero: r-a = 0")
        if math.isclose(self.Q_min_1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_min_1 = {self.Q_min_1} matches formula {expected}")
        else:
            return (False, f"Q_min_1 = {self.Q_min_1} != formula {expected}")

    def test_trace_Q_min_2_formula(self):
        """Check that Q_min_2 matches the derived formula a^2 q (3r + a) / (r (r+a)^2)."""
        if None in (self.a, self.r, self.q, self.Q_min_2):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        try:
            expected = a**2 * q * (3*r + a) / (r * (r + a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero")
        if math.isclose(self.Q_min_2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_min_2 = {self.Q_min_2} matches formula {expected}")
        else:
            return (False, f"Q_min_2 = {self.Q_min_2} != formula {expected}")

    def test_trace_sigma_min_expression_sub1(self):
        """Re-derive sigma_min expression from given Q_min_1 and check it is zero."""
        # Same as invariant test but explicit derivation test
        return self.test_invariant_sigma_min_zero_sub1()  # Reuse

    def test_trace_sigma_min_expression_sub2(self):
        return self.test_invariant_sigma_min_zero_sub2()

    # ---------- Monotonicity tests (perturb input parameters) ----------
    def test_monotonicity_Q_min_1_with_r(self):
        """Increasing r should increase Q_min_1 (perturb r +1% and compare)."""
        if None in (self.a, self.r, self.q, self.Q_min_1):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        dr = 0.01 * r
        try:
            Q_base = a**2 * q * (3*r - a) / (r * (r - a)**2)
            Q_perturb = a**2 * q * (3*(r+dr) - a) / ((r+dr) * (r+dr - a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero in monotonicity calculation")
        if not math.isclose(self.Q_min_1, Q_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_min_1 {self.Q_min_1} does not match base formula {Q_base}")
        if Q_perturb > Q_base:
            return (True, f"Q_min_1 increases with r: from {Q_base} to {Q_perturb}")
        else:
            return (False, f"Q_min_1 does not increase: base {Q_base}, perturbed {Q_perturb}")

    def test_monotonicity_Q_min_2_with_r(self):
        """Increasing r should increase Q_min_2 (perturb r +1%)."""
        if None in (self.a, self.r, self.q, self.Q_min_2):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        dr = 0.01 * r
        try:
            Q_base = a**2 * q * (3*r + a) / (r * (r + a)**2)
            Q_perturb = a**2 * q * (3*(r+dr) + a) / ((r+dr) * (r+dr + a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero in monotonicity calculation")
        if not math.isclose(self.Q_min_2, Q_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_min_2 {self.Q_min_2} does not match base formula {Q_base}")
        if Q_perturb > Q_base:
            return (True, f"Q_min_2 increases with r: from {Q_base} to {Q_perturb}")
        else:
            return (False, f"Q_min_2 does not increase: base {Q_base}, perturbed {Q_perturb}")

    def test_monotonicity_Q_min_1_with_q(self):
        """Increasing q should increase Q_min_1 linearly (perturb q +10%)."""
        if None in (self.a, self.r, self.q, self.Q_min_1):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        dq = 0.1 * q
        try:
            Q_base = a**2 * q * (3*r - a) / (r * (r - a)**2)
            Q_perturb = a**2 * (q+dq) * (3*r - a) / (r * (r - a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero in monotonicity calculation")
        if not math.isclose(self.Q_min_1, Q_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_min_1 {self.Q_min_1} does not match base formula {Q_base}")
        if Q_perturb > Q_base:
            return (True, f"Q_min_1 increases with q: from {Q_base} to {Q_perturb}")
        else:
            return (False, f"Q_min_1 does not increase: base {Q_base}, perturbed {Q_perturb}")

    def test_monotonicity_Q_min_2_with_q(self):
        """Increasing q should increase Q_min_2 linearly (perturb q +10%)."""
        if None in (self.a, self.r, self.q, self.Q_min_2):
            return (False, "SKIP: missing data")
        a = self.a
        r = self.r
        q = self.q
        dq = 0.1 * q
        try:
            Q_base = a**2 * q * (3*r + a) / (r * (r + a)**2)
            Q_perturb = a**2 * (q+dq) * (3*r + a) / (r * (r + a)**2)
        except ZeroDivisionError:
            return (False, "Division by zero in monotonicity calculation")
        if not math.isclose(self.Q_min_2, Q_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_min_2 {self.Q_min_2} does not match base formula {Q_base}")
        if Q_perturb > Q_base:
            return (True, f"Q_min_2 increases with q: from {Q_base} to {Q_perturb}")
        else:
            return (False, f"Q_min_2 does not increase: base {Q_base}, perturbed {Q_perturb}")

    # ---------- Answer sanity (weak, at most one) ----------
    def test_answer_sanity_positive(self):
        """Weak sanity: Q_min values are positive and finite."""
        if self.Q_min_1 is None or self.Q_min_2 is None:
            return (False, "SKIP: missing result keys")
        if not (math.isfinite(self.Q_min_1) and math.isfinite(self.Q_min_2)):
            return (False, "Q_min values are not finite")
        if self.Q_min_1 <= 0 or self.Q_min_2 <= 0:
            return (False, "Q_min values should be positive")
        return (True, f"Q_min_1={self.Q_min_1}, Q_min_2={self.Q_min_2} are positive and finite")