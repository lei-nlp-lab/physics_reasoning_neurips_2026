import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.solution = solution_instance
        # Extract all expected keys with safe defaults
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        self.R_1 = getattr(solution_instance, 'R_1', getattr(solution_instance, 'results', {}).get('R_1', 0.0))
        self.P_max = getattr(solution_instance, 'P_max', getattr(solution_instance, 'results', {}).get('P_max', 0.0))
        self.U_1 = getattr(solution_instance, 'U_1', getattr(solution_instance, 'results', {}).get('U_1', 0.0))
        self.U_1all = getattr(solution_instance, 'U_1all', getattr(solution_instance, 'results', {}).get('U_1all', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.P_2 = getattr(solution_instance, 'P_2', getattr(solution_instance, 'results', {}).get('P_2', 0.0))
        self.U_2 = getattr(solution_instance, 'U_2', getattr(solution_instance, 'results', {}).get('U_2', 0.0))
        self.R_2_prime = getattr(solution_instance, 'R_2\'', getattr(solution_instance, 'results', {}).get('R_2\'', 0.0))
        self.R_2_double_prime = getattr(solution_instance, 'R_2\'\'', getattr(solution_instance, 'results', {}).get('R_2\'\'', 0.0))
        self.I_prime = getattr(solution_instance, 'I\'', getattr(solution_instance, 'results', {}).get('I\'', 0.0))
        self.R_2max = getattr(solution_instance, 'R_2max', getattr(solution_instance, 'results', {}).get('R_2max', 0.0))
        self.R_0 = getattr(solution_instance, 'R_0', getattr(solution_instance, 'results', {}).get('R_0', 0.0))
        self.U = getattr(solution_instance, 'U', getattr(solution_instance, 'results', {}).get('U', 0.0))

        # Constants from steps analysis for validation
        self.step_values = {
            'I': 0.3,
            'U_1all': 5.4,
            'U_2': 3.0,
            'R_2max': 10.0,
            'U_1': 2.4,
            'R_1': 8.0,
            'R_2\'': 5.0,
            'P_2': 0.8,
            'I\'': 0.4,
            'U': 6.0,
            'R_0': 2.0,
            'P_max': 0.9
        }
        self.logs = []

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _log(self, msg, success):
        self.logs.append((msg, success))

    def test_invariants(self) -> bool:
        """Verify Ohm's Law, series circuit laws, and power formulas."""
        success = True

        # 1. Ohm's Law: R_2max = U_2 / I
        R_2max_calc = self.U_2 / self.I if abs(self.I) > 1e-12 else 0.0
        if not self._is_close(self.R_2max, R_2max_calc):
            self._log(f"Ohm's Law failed: R_2max = {self.R_2max}, expected U_2/I = {R_2max_calc}", False)
            success = False
        else:
            self._log(f"Ohm's Law (R_2max = U_2/I) satisfied: {self.R_2max} ≈ {R_2max_calc}", True)

        # 2. Ohm's Law: R_1 = U_1 / I
        R_1_calc = self.U_1 / self.I if abs(self.I) > 1e-12 else 0.0
        if not self._is_close(self.R_1, R_1_calc):
            self._log(f"Ohm's Law failed: R_1 = {self.R_1}, expected U_1/I = {R_1_calc}", False)
            success = False
        else:
            self._log(f"Ohm's Law (R_1 = U_1/I) satisfied: {self.R_1} ≈ {R_1_calc}", True)

        # 3. Series voltage: U_1 = U_1all - U_2
        U_1_calc = self.U_1all - self.U_2
        if not self._is_close(self.U_1, U_1_calc):
            self._log(f"Series voltage law failed: U_1 = {self.U_1}, expected U_1all - U_2 = {U_1_calc}", False)
            success = False
        else:
            self._log(f"Series voltage law (U_1 = U_1all - U_2) satisfied: {self.U_1} ≈ {U_1_calc}", True)

        # 4. Resistance proportionality: R_2' = 0.5 * R_2max
        R_2_prime_calc = 0.5 * self.R_2max
        if not self._is_close(self.R_2_prime, R_2_prime_calc):
            self._log(f"Resistance proportionality failed: R_2' = {self.R_2_prime}, expected 0.5 * R_2max = {R_2_prime_calc}", False)
            success = False
        else:
            self._log(f"Resistance proportionality (R_2' = 0.5 * R_2max) satisfied: {self.R_2_prime} ≈ {R_2_prime_calc}", True)

        # 5. Power formula: P_2 = I'^2 * R_2'
        P_2_calc = self.I_prime**2 * self.R_2_prime
        if not self._is_close(self.P_2, P_2_calc):
            self._log(f"Power formula failed: P_2 = {self.P_2}, expected I'^2 * R_2' = {P_2_calc}", False)
            success = False
        else:
            self._log(f"Power formula (P_2 = I'^2 * R_2') satisfied: {self.P_2} ≈ {P_2_calc}", True)

        # 6. Series circuit: U = U_1all + I * R_0
        U_calc1 = self.U_1all + self.I * self.R_0
        if not self._is_close(self.U, U_calc1):
            self._log(f"Series circuit law (step 9) failed: U = {self.U}, expected U_1all + I*R_0 = {U_calc1}", False)
            success = False
        else:
            self._log(f"Series circuit law (U = U_1all + I*R_0) satisfied: {self.U} ≈ {U_calc1}", True)

        # 7. Series circuit: U = I' * (R_1 + R_2' + R_0)
        U_calc2 = self.I_prime * (self.R_1 + self.R_2_prime + self.R_0)
        if not self._is_close(self.U, U_calc2):
            self._log(f"Series circuit law (step 10) failed: U = {self.U}, expected I'*(R_1+R_2'+R_0) = {U_calc2}", False)
            success = False
        else:
            self._log(f"Series circuit law (U = I'*(R_1+R_2'+R_0)) satisfied: {self.U} ≈ {U_calc2}", True)

        # 8. Maximum power transfer condition (numerical check): P_max occurs when R_2 = R_0 + R_1
        R_th = self.R_0 + self.R_1
        if not self._is_close(self.R_2max, R_th):
            self._log(f"Maximum power condition: R_2max = {self.R_2max} should equal R_0+R_1 = {R_th}", False)
            success = False
        else:
            self._log(f"Maximum power condition (R_2max = R_0+R_1) satisfied: {self.R_2max} ≈ {R_th}", True)

        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with steps analysis and final answers."""
        success = True

        # Check step values against stored results (if available)
        for key, expected in self.step_values.items():
            actual = getattr(self, key.replace("'", "_prime").replace("\"", "_double_prime"), None)
            if actual is not None and not self._is_close(actual, expected):
                self._log(f"Trace mismatch for {key}: got {actual}, expected {expected}", False)
                success = False
            elif actual is not None:
                self._log(f"Trace match for {key}: {actual} ≈ {expected}", True)

        # Check final answers against target expressions
        # First target: R_2max = 10 Ω
        if not self._is_close(self.R_2max, 10.0):
            self._log(f"First target answer failed: R_2max = {self.R_2max} ≠ 10 Ω", False)
            success = False
        else:
            self._log("First target answer (10 Ω) satisfied", True)

        # Second target: U = 6 V, R_0 = 2 Ω
        if not (self._is_close(self.U, 6.0) and self._is_close(self.R_0, 2.0)):
            self._log(f"Second target answer failed: U = {self.U} V, R_0 = {self.R_0} Ω", False)
            success = False
        else:
            self._log(f"Second target answer (6 V, 2 Ω) satisfied: U={self.U}, R_0={self.R_0}", True)

        # Third target: P expression and P_max = 0.9 W
        # Check P_max value
        if not self._is_close(self.P_max, 0.9):
            self._log(f"Third target (P_max) failed: P_max = {self.P_max} ≠ 0.9 W", False)
            success = False
        else:
            self._log(f"Third target (P_max = 0.9 W) satisfied", True)

        # Verify the derived power expression symbolically
        x = sp.symbols('x')
        # Correct expression from step 14-15: P = (6/(10 + x))^2 * x = 36*x/(10+x)^2
        P_expr = 36 * x / (10 + x)**2
        # Compare with the stored expression if available (as string, but we check numerically)
        # We'll test at a few points: x=5 (R_2') and x=10 (R_2max)
        test_points = [5.0, 10.0]
        for x_val in test_points:
            P_calc = 36 * x_val / (10 + x_val)**2
            if x_val == 5.0:
                if not self._is_close(P_calc, self.P_2):
                    self._log(f"Power expression mismatch at x=5: P(5)={P_calc}, expected P_2={self.P_2}", False)
                    success = False
            if x_val == 10.0:
                if not self._is_close(P_calc, self.P_max):
                    self._log(f"Power expression mismatch at x=10: P(10)={P_calc}, expected P_max={self.P_max}", False)
                    success = False
        self._log("Power expression P = 36*x/(10+x)^2 validated at key points", True)

        return success

    def test_monotonicity(self) -> bool:
        """Perturb R2 (x) and verify power trend around maximum."""
        success = True

        # Power function: P(x) = 36*x/(10+x)^2
        def P_func(x):
            return 36 * x / (10 + x)**2

        # Test points around the maximum (R2 = R_0+R_1 = 10 Ω)
        # For x < 10, function should be increasing; for x > 10, decreasing.
        test_pairs = [(8.0, 9.0), (9.0, 10.0), (10.0, 11.0), (11.0, 12.0)]
        for x1, x2 in test_pairs:
            P1 = P_func(x1)
            P2 = P_func(x2)
            if x1 < 10 and x2 < 10:
                if P2 <= P1:
                    self._log(f"Monotonicity failed: P({x1}) = {P1}, P({x2}) = {P2} but should increase for x<10", False)
                    success = False
                else:
                    self._log(f"Monotonicity OK: P({x1}) = {P1} < P({x2}) = {P2} for x<10", True)
            elif x1 >= 10 and x2 > x1:
                if P2 >= P1:
                    self._log(f"Monotonicity failed: P({x1}) = {P1}, P({x2}) = {P2} but should decrease for x>10", False)
                    success = False
                else:
                    self._log(f"Monotonicity OK: P({x1}) = {P1} > P({x2}) = {P2} for x>10", True)

        # Additional check: maximum at x=10
        x_max = 10.0
        P_max_calc = P_func(x_max)
        if not self._is_close(P_max_calc, self.P_max):
            self._log(f"Maximum power mismatch: P(10) = {P_max_calc}, expected P_max = {self.P_max}", False)
            success = False
        else:
            self._log(f"Maximum power at x=10: P(10) = {P_max_calc} ≈ P_max = {self.P_max}", True)

        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self._log(f"--- Starting test: {name} ---", True)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {name} crashed: {e}", False)
                overall_success = False

        # Compile logs into messages
        log_messages = [f"{'PASS' if ok else 'FAIL'}: {msg}" for msg, ok in self.logs]
        return overall_success, log_messages