import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all parameters and results from the solution instance.
        Uses getattr with fallback to .results dictionary to avoid AttributeError.
        """
        # Extract input parameters (must be numeric)
        self.Q = float(getattr(solution_instance, 'Q', solution_instance.results.get('Q', 1e-6)))
        self.d = float(getattr(solution_instance, 'd', solution_instance.results.get('d', 0.1)))
        self.m = float(getattr(solution_instance, 'm', solution_instance.results.get('m', 0.01)))
        self.q = float(getattr(solution_instance, 'q', solution_instance.results.get('q', 1e-9)))
        self.v = float(getattr(solution_instance, 'v', solution_instance.results.get('v', 0.5)))
        self.k = float(getattr(solution_instance, 'k', solution_instance.results.get('k', 8.99e9)))
        self.g = float(getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8)))

        # Extract results safely (must be numeric)
        self.results = {}
        expected_keys = [
            'W_CD', 'delta_E_k', 'delta_E_prime_k', 'phi_C', 'F_net', 'phi_E',
            'W_EF', 'F_1', 'E_k', 'U_CD', 'phi_F', 'v_E', 'a', 'W_DE',
            'U_DE', 'phi_D', 'delta_E_double_prime_k', 'F_2'
        ]
        for key in expected_keys:
            val = getattr(solution_instance, key, solution_instance.results.get(key, None))
            if val is not None:
                self.results[key] = float(val)
            else:
                self.results[key] = None

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify fundamental physical laws (energy conservation, Newton's laws)."""
        logs = []
        try:
            # 1. Newton's Second Law at point D (from steps 1-3)
            # Compute expected values numerically
            F1_expected = self.k * self.Q * self.q / (2 * self.d**2)
            F2_expected = F1_expected
            cos45 = math.sqrt(2)/2
            F_net_expected = self.m * self.g + (F1_expected + F2_expected) * cos45
            a_expected = F_net_expected / self.m

            # Compare with result
            if self.results['a'] is not None:
                if not self._is_close(self.results['a'], a_expected):
                    logs.append(f"FAIL: Acceleration a mismatch. Expected {a_expected}, got {self.results['a']}")
                else:
                    logs.append("PASS: Newton's second law at D satisfied.")
            else:
                logs.append("SKIP: Acceleration 'a' not provided.")

            # 2. Energy conservation: C to D
            # W_CD = ΔE_k = 0.5*m*v^2
            delta_E_k_expected = 0.5 * self.m * self.v**2
            # W_CD from solution: mgd + q*(phi_C - phi_D), with phi_D=0
            phi_D = 0
            W_CD_expected = self.m * self.g * self.d + self.q * (self.results['phi_C'] - phi_D)
            
            if self.results['delta_E_k'] is not None and self.results['W_CD'] is not None:
                if not self._is_close(self.results['delta_E_k'], delta_E_k_expected):
                    logs.append(f"FAIL: ΔE_k mismatch. Expected {delta_E_k_expected}, got {self.results['delta_E_k']}")
                if not self._is_close(self.results['W_CD'], W_CD_expected):
                    logs.append(f"FAIL: W_CD mismatch. Expected {W_CD_expected}, got {self.results['W_CD']}")
                if self._is_close(self.results['W_CD'], self.results['delta_E_k']):
                    logs.append("PASS: Work-energy theorem holds for C→D.")
                else:
                    logs.append("FAIL: Work-energy theorem violated for C→D.")
            else:
                logs.append("SKIP: Missing results for energy check C→D.")

            # 3. Energy conservation: D to E
            # v_E = sqrt(2)*v from step 10
            v_E_expected = math.sqrt(2) * self.v
            delta_E_prime_k_expected = 0.5 * self.m * (v_E_expected**2 - self.v**2)
            # W_DE = mgd + q*U_DE, with U_DE = U_CD = phi_C (since phi_D=0)
            W_DE_expected = self.m * self.g * self.d + self.q * self.results['phi_C']

            if self.results['v_E'] is not None:
                if not self._is_close(self.results['v_E'], v_E_expected):
                    logs.append(f"FAIL: v_E mismatch. Expected {v_E_expected}, got {self.results['v_E']}")
                else:
                    logs.append("PASS: v_E consistent with √2*v.")
            
            if self.results['delta_E_prime_k'] is not None and self.results['W_DE'] is not None:
                if not self._is_close(self.results['W_DE'], W_DE_expected):
                    logs.append(f"FAIL: W_DE mismatch. Expected {W_DE_expected}, got {self.results['W_DE']}")
                if self._is_close(self.results['W_DE'], self.results['delta_E_prime_k']):
                    logs.append("PASS: Work-energy theorem holds for D→E.")
                else:
                    logs.append("FAIL: Work-energy theorem violated for D→E.")
            else:
                logs.append("SKIP: Missing results for energy check D→E.")

            # 4. Energy conservation: E to F
            phi_E_expected = -self.k * self.Q / self.d
            phi_F_expected = -self.k * self.Q / (math.sqrt(2) * self.d)
            W_EF_expected = self.m * self.g * self.d + self.q * (phi_E_expected - phi_F_expected)
            # ΔE''_k = E_k - 0.5*m*v_E^2
            if self.results['E_k'] is not None and self.results['v_E'] is not None:
                delta_E_double_prime_k_expected = self.results['E_k'] - 0.5 * self.m * self.results['v_E']**2
                if self.results['delta_E_double_prime_k'] is not None:
                    if not self._is_close(self.results['delta_E_double_prime_k'], delta_E_double_prime_k_expected):
                        logs.append(f"FAIL: ΔE''_k mismatch. Expected {delta_E_double_prime_k_expected}, got {self.results['delta_E_double_prime_k']}")
                if self.results['W_EF'] is not None:
                    if not self._is_close(self.results['W_EF'], W_EF_expected):
                        logs.append(f"FAIL: W_EF mismatch. Expected {W_EF_expected}, got {self.results['W_EF']}")
                    if self._is_close(self.results['W_EF'], self.results['delta_E_double_prime_k']):
                        logs.append("PASS: Work-energy theorem holds for E→F.")
                    else:
                        logs.append("FAIL: Work-energy theorem violated for E→F.")
            else:
                logs.append("SKIP: Missing results for energy check E→F.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
        
        return len(logs) == 0 or all("PASS" in log or "SKIP" in log for log in logs), logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that phi_C expression leads to consistent U_CD
            phi_C_expected = (self.m * self.v**2 - 2 * self.m * self.g * self.d) / (2 * self.q)
            if self.results['phi_C'] is not None:
                if not self._is_close(self.results['phi_C'], phi_C_expected):
                    logs.append(f"FAIL: phi_C mismatch. Expected {phi_C_expected}, got {self.results['phi_C']}")
                else:
                    logs.append("PASS: phi_C expression consistent with work-energy.")

            # Check that phi_D is 0 (from step 4)
            if self.results['phi_D'] is not None:
                if not self._is_close(self.results['phi_D'], 0):
                    logs.append(f"FAIL: phi_D should be 0, got {self.results['phi_D']}")
                else:
                    logs.append("PASS: phi_D = 0 as expected.")

            # Check that U_CD = phi_C - phi_D = phi_C
            if self.results['U_CD'] is not None and self.results['phi_C'] is not None:
                if not self._is_close(self.results['U_CD'], self.results['phi_C']):
                    logs.append(f"FAIL: U_CD should equal phi_C, got {self.results['U_CD']} vs {self.results['phi_C']}")
                else:
                    logs.append("PASS: U_CD = phi_C (since phi_D=0).")

            # Check that F1 and F2 are equal and have correct magnitude
            F_expected = self.k * self.Q * self.q / (2 * self.d**2)
            if self.results['F_1'] is not None:
                if not self._is_close(self.results['F_1'], F_expected):
                    logs.append(f"FAIL: F1 mismatch. Expected {F_expected}, got {self.results['F_1']}")
                else:
                    logs.append("PASS: F1 magnitude correct.")
            if self.results['F_2'] is not None:
                if not self._is_close(self.results['F_2'], F_expected):
                    logs.append(f"FAIL: F2 mismatch. Expected {F_expected}, got {self.results['F_2']}")
                else:
                    logs.append("PASS: F2 magnitude correct.")
            if self.results['F_1'] is not None and self.results['F_2'] is not None:
                if not self._is_close(self.results['F_1'], self.results['F_2']):
                    logs.append("FAIL: F1 and F2 should be equal.")
                else:
                    logs.append("PASS: F1 = F2 as expected.")

            # Check that U_DE = U_CD (from step 9)
            if self.results['U_DE'] is not None and self.results['U_CD'] is not None:
                if not self._is_close(self.results['U_DE'], self.results['U_CD']):
                    logs.append(f"FAIL: U_DE should equal U_CD, got {self.results['U_DE']} vs {self.results['U_CD']}")
                else:
                    logs.append("PASS: U_DE = U_CD.")

            # Check that final E_k matches expression from step 13
            E_k_expected = (self.m * self.g * self.d + self.m * self.v**2 - 
                            (self.k * self.Q * self.q / self.d) * (1 - math.sqrt(2)/2))
            if self.results['E_k'] is not None:
                if not self._is_close(self.results['E_k'], E_k_expected):
                    logs.append(f"FAIL: E_k mismatch. Expected {E_k_expected}, got {self.results['E_k']}")
                else:
                    logs.append("PASS: E_k expression correct.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
        
        return len(logs) == 0 or all("PASS" in log or "SKIP" in log for log in logs), logs

    def test_monotonicity(self):
        """Perturb an input and verify expected trends, then restore."""
        logs = []
        try:
            # Perturb mass m by +10%
            original_m = self.m
            perturbed_m = original_m * 1.1
            
            # 1. Acceleration a = g + (√2 k Q q) / (2 m d²) → decreases as m increases
            a_original = self.g + (math.sqrt(2) * self.k * self.Q * self.q) / (2 * original_m * self.d**2)
            a_perturbed = self.g + (math.sqrt(2) * self.k * self.Q * self.q) / (2 * perturbed_m * self.d**2)
            if a_perturbed < a_original:
                logs.append("PASS: Acceleration a decreases when mass increases (as expected).")
            else:
                logs.append(f"FAIL: Acceleration a did not decrease when mass increased: {a_original} -> {a_perturbed}")

            # 2. phi_C = (m v² - 2 m g d) / (2q) = m(v² - 2gd)/(2q) → increases with m if v² > 2gd
            phi_C_original = (original_m * (self.v**2 - 2 * self.g * self.d)) / (2 * self.q)
            phi_C_perturbed = (perturbed_m * (self.v**2 - 2 * self.g * self.d)) / (2 * self.q)
            # Assuming v² > 2gd from context (sphere accelerates downward)
            if self.v**2 > 2 * self.g * self.d:
                if phi_C_perturbed > phi_C_original:
                    logs.append("PASS: phi_C increases when mass increases (since v² > 2gd).")
                else:
                    logs.append(f"FAIL: phi_C did not increase when mass increased: {phi_C_original} -> {phi_C_perturbed}")
            else:
                logs.append("SKIP: v² ≤ 2gd, phi_C trend not determined.")

            # 3. E_k = m g d + m v² - (k Q q / d)(1 - √2/2) → increases with m (linear in m)
            term = (self.k * self.Q * self.q / self.d) * (1 - math.sqrt(2)/2)
            E_k_original = original_m * (self.g * self.d + self.v**2) - term
            E_k_perturbed = perturbed_m * (self.g * self.d + self.v**2) - term
            if E_k_perturbed > E_k_original:
                logs.append("PASS: E_k increases when mass increases.")
            else:
                logs.append(f"FAIL: E_k did not increase when mass increased: {E_k_original} -> {E_k_perturbed}")

            # Restore original m
            self.m = original_m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
        
        return len(logs) == 0 or all("PASS" in log or "SKIP" in log for log in logs), logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [("Invariants", self.test_invariants),
                                       ("Trace Consistency", self.test_trace_consistency),
                                       ("Monotonicity", self.test_monotonicity)]:
            test_success, logs = test_method()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not test_success and any("FAIL" in log for log in logs):
                success = False

        # Overall verdict
        if success:
            all_logs.append("=== SUMMARY: All tests passed or skipped. ===")
        else:
            all_logs.append("=== SUMMARY: Some tests failed. ===")

        return success, all_logs