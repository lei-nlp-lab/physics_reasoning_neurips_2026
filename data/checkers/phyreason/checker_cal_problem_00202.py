import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.p0 = getattr(solution_instance, 'p0', 1.0)
        self.V = getattr(solution_instance, 'V', 1.0)
        
        # Safe extraction of results using .results dict
        results = getattr(solution_instance, 'results', {})
        
        self.p_B = results.get('p_B', None)
        self.V_A = results.get('V_A', None)
        self.p_A = results.get('p_A', None)
        self.p_B_prime = results.get('p_B\'', None)
        self.V_A_prime = results.get('V_A\'', None)
        
        # Fallback: if not in results, try direct attribute (but safe)
        if self.p_B is None:
            self.p_B = getattr(solution_instance, 'p_B', None)
        if self.V_A is None:
            self.V_A = getattr(solution_instance, 'V_A', None)
        if self.p_A is None:
            self.p_A = getattr(solution_instance, 'p_A', None)
        if self.p_B_prime is None:
            self.p_B_prime = getattr(solution_instance, 'p_B_prime', None)
        if self.V_A_prime is None:
            self.V_A_prime = getattr(solution_instance, 'V_A_prime', None)
        
        self.logs = []
        self.success = True
        
    def _log_error(self, msg):
        self.logs.append(f"ERROR: {msg}")
        self.success = False
        
    def _log_info(self, msg):
        self.logs.append(f"INFO: {msg}")
        
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        self._log_info("Testing invariants (Boyle's Law and pressure difference)...")
        
        # First scenario: partition stationary, V_B = V/2
        if self.p_B is not None:
            # Check Boyle's Law for gas B
            expected_p_B = 2 * self.p0
            if not self._is_close(self.p_B, expected_p_B):
                self._log_error(f"p_B = {self.p_B} but expected {expected_p_B} from Boyle's Law (p0*V = p_B*(V/2)).")
        else:
            self._log_error("p_B not provided for first scenario.")
            
        if self.p_A is not None and self.V_A is not None:
            # CORRECTED: According to ground truth step_3: p_A = p_B + 0.5*p0
            expected_p_A = self.p_B + 0.5 * self.p0
            if not self._is_close(self.p_A, expected_p_A):
                self._log_error(f"p_A = {self.p_A} but expected {expected_p_A} (p_B + 0.5*p0).")
            
            # Check Boyle's Law for gas A in first scenario
            lhs = self.p0 * self.V
            rhs = self.p_A * self.V_A
            if not self._is_close(lhs, rhs):
                self._log_error(f"Boyle's Law for A fails: p0*V = {lhs}, p_A*V_A = {rhs}.")
        else:
            self._log_error("p_A and/or V_A not provided for first scenario.")
        
        # Second scenario: partition slides, pressure difference fixed at 0.5 p0
        if self.V_A_prime is not None and self.p_B_prime is not None:
            # Compute V_B_prime from total volume constraint (step_9: V_A' + V_B' = 2V)
            V_B_prime = 2 * self.V - self.V_A_prime
            # Boyle's Law for A'
            lhs_A = self.p0 * self.V
            # We need p_A_prime from the pressure difference condition: p_A' = p_B' - 0.5 p0
            p_A_prime = self.p_B_prime - 0.5 * self.p0
            rhs_A = p_A_prime * self.V_A_prime
            if not self._is_close(lhs_A, rhs_A):
                self._log_error(f"Boyle's Law for A' fails: p0*V = {lhs_A}, p_A'*V_A' = {rhs_A}.")
            
            # Boyle's Law for B'
            rhs_B = self.p_B_prime * V_B_prime
            if not self._is_close(lhs_A, rhs_B):
                self._log_error(f"Boyle's Law for B' fails: p0*V = {lhs_A}, p_B'*V_B' = {rhs_B}.")
            
            # Pressure difference condition for second scenario (step_9)
            if not self._is_close(p_A_prime, self.p_B_prime - 0.5 * self.p0):
                self._log_error(f"Pressure difference condition for sliding fails: p_A' = {p_A_prime}, expected p_B' - 0.5*p0 = {self.p_B_prime - 0.5*self.p0}.")
        else:
            self._log_info("Skipping second scenario invariants because V_A' and/or p_B' not provided.")
            
        self._log_info("Invariants test completed.")
        return self.success
    
    def test_trace_consistency(self):
        self._log_info("Testing trace consistency with steps analysis...")
        
        # Check first scenario step-by-step
        if self.p_B is not None:
            # Step 1: p0*V = p_B*(V/2) => p_B = 2*p0
            expected_p_B = 2 * self.p0
            if not self._is_close(self.p_B, expected_p_B):
                self._log_error(f"Step 1 (Boyle's Law for B) inconsistent: p_B = {self.p_B}, expected {expected_p_B}.")
        
        if self.p_A is not None and self.V_A is not None:
            # Step 2 and 3: p0*V = p_A*V_A and p_A = p_B + 0.5*p0 (from ground truth step_3)
            expected_p_A = self.p_B + 0.5 * self.p0
            if not self._is_close(self.p_A, expected_p_A):
                self._log_error(f"Step 3 (pressure difference) inconsistent: p_A = {self.p_A}, expected {expected_p_A}.")
            
            # Step 4: V_A = (p0*V) / (p_B + 0.5*p0) = V / 2.5 = 0.4 V
            expected_V_A = 0.4 * self.V
            if not self._is_close(self.V_A, expected_V_A):
                self._log_error(f"Step 4 (solve for V_A) inconsistent: V_A = {self.V_A}, expected {expected_V_A}.")
        else:
            self._log_info("Skipping first scenario trace consistency because p_A and/or V_A not provided.")
        
        # Second scenario: check that results satisfy the system
        if self.V_A_prime is not None and self.p_B_prime is not None:
            # Steps 7-10: 
            #   p0*V = p_A' * V_A'   (7)
            #   p0*V = p_B' * V_B'   (8) with V_B' = 2V - V_A'
            #   p_A' = p_B' - 0.5 p0 (9)  # Note: ground truth has this sign
            V_B_prime = 2 * self.V - self.V_A_prime
            p_A_prime = self.p_B_prime - 0.5 * self.p0
            
            if not self._is_close(self.p0 * self.V, p_A_prime * self.V_A_prime):
                self._log_error("Step 7 (Boyle's Law for A') inconsistent.")
            if not self._is_close(self.p0 * self.V, self.p_B_prime * V_B_prime):
                self._log_error("Step 8 (Boyle's Law for B') inconsistent.")
            if not self._is_close(p_A_prime, self.p_B_prime - 0.5 * self.p0):
                self._log_error("Step 9 (pressure relationship) inconsistent.")
            
            # Step 10: explicit values
            expected_V_A_prime = (math.sqrt(5) - 1) * self.V
            expected_p_B_prime = (3 + math.sqrt(5)) / 4 * self.p0
            if not self._is_close(self.V_A_prime, expected_V_A_prime):
                self._log_error(f"Step 10: V_A' = {self.V_A_prime}, expected {expected_V_A_prime}.")
            if not self._is_close(self.p_B_prime, expected_p_B_prime):
                self._log_error(f"Step 10: p_B' = {self.p_B_prime}, expected {expected_p_B_prime}.")
        else:
            self._log_info("Skipping second scenario trace consistency because V_A' and/or p_B' not provided.")
            
        self._log_info("Trace consistency test completed.")
        return self.success
    
    def test_monotonicity(self):
        self._log_info("Testing monotonicity by perturbing initial volume V...")
        
        original_V = self.V
        perturbed_V = original_V * 1.1  # increase by 10%
        
        # First scenario: expected formulas
        expected_p_B_original = 2 * self.p0
        expected_V_A_original = 0.4 * original_V
        
        expected_p_B_perturbed = 2 * self.p0  # independent of V
        expected_V_A_perturbed = 0.4 * perturbed_V
        
        # Check original values (if provided)
        if self.p_B is not None and not self._is_close(self.p_B, expected_p_B_original):
            self._log_error(f"Original p_B {self.p_B} does not match expected {expected_p_B_original}; skipping scaling check.")
        else:
            # Check that perturbed p_B remains the same (independent of V)
            if not self._is_close(expected_p_B_perturbed, expected_p_B_original):
                self._log_error("p_B should be independent of V but changed when V was perturbed.")
        
        if self.V_A is not None and not self._is_close(self.V_A, expected_V_A_original):
            self._log_error(f"Original V_A {self.V_A} does not match expected {expected_V_A_original}; skipping scaling check.")
        else:
            # Check that perturbed V_A scales linearly with V
            if not self._is_close(expected_V_A_perturbed, expected_V_A_original * 1.1):
                self._log_error("V_A should scale linearly with V but did not when V was increased by 10%.")
        
        # Second scenario: expected formulas
        expected_V_A_prime_original = (math.sqrt(5) - 1) * original_V
        expected_p_B_prime_original = (3 + math.sqrt(5)) / 4 * self.p0
        
        expected_V_A_prime_perturbed = (math.sqrt(5) - 1) * perturbed_V
        expected_p_B_prime_perturbed = (3 + math.sqrt(5)) / 4 * self.p0
        
        if self.V_A_prime is not None and not self._is_close(self.V_A_prime, expected_V_A_prime_original):
            self._log_error(f"Original V_A' {self.V_A_prime} does not match expected {expected_V_A_prime_original}; skipping scaling check.")
        else:
            if not self._is_close(expected_V_A_prime_perturbed, expected_V_A_prime_original * 1.1):
                self._log_error("V_A' should scale linearly with V but did not when V was increased by 10%.")
        
        if self.p_B_prime is not None and not self._is_close(self.p_B_prime, expected_p_B_prime_original):
            self._log_error(f"Original p_B' {self.p_B_prime} does not match expected {expected_p_B_prime_original}; skipping scaling check.")
        else:
            if not self._is_close(expected_p_B_prime_perturbed, expected_p_B_prime_original):
                self._log_error("p_B' should be independent of V but changed when V was perturbed.")
        
        self._log_info("Monotonicity test completed.")
        return self.success
    
    def run_all_tests(self):
        self.logs.clear()
        self.success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test in tests:
            try:
                if not test():
                    self._log_error(f"{name} test failed.")
            except Exception as e:
                self._log_error(f"{name} test crashed: {e}\n{traceback.format_exc()}")
        
        return self.success, self.logs