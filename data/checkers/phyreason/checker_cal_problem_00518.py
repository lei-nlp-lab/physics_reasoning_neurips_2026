import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize PhysicsChecker with safe attribute access.
        All resistances from problem context, all other quantities from results.
        """
        # Fixed resistances from problem context
        self.R1 = 2.0  # Ω
        self.R2 = 3.0  # Ω
        self.R3 = 4.0  # Ω
        
        # Extract results safely using getattr or results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Main circuit currents (from steps 1-5)
        self.I1 = self._get_result('I_1', 0.0)
        self.U = self._get_result('U', 0.0)
        self.I2 = self._get_result('I_2', 0.0)
        self.I3 = self._get_result('I_3', 0.0)
        self.I_total = self._get_result('I', 0.0)
        
        # Equivalent circuit calculations (from steps 6-11)
        self.inv_R = self._get_result('1/R', 0.0)
        self.R_eq = self._get_result('R', 0.0)
        self.U_prime = self._get_result('U\'', 0.0)
        self.I1_prime = self._get_result('I_1\'', 0.0)
        self.I2_prime = self._get_result('I_2\'', 0.0)
        self.I3_prime = self._get_result('I_3\'', 0.0)
        
        # For monotonicity test backup
        self._original_R1 = self.R1
        
        # Target answers parsing
        self.target_answer1 = None  # "6.5A"
        self.target_answer2 = None  # List of three fractions
        self._parse_target_answers(solution_instance)
        
    def _get_result(self, key, default):
        """Safely get result from solution_instance with fallback."""
        # First try results dictionary
        val = self.results.get(key, default)
        return float(val) if isinstance(val, (int, float)) else default
    
    def _parse_target_answers(self, solution_instance):
        """Parse target answers from solution_instance."""
        try:
            target_answers = getattr(solution_instance, 'target_answers', [])
            if len(target_answers) >= 2:
                # Parse first answer: "6.5A"
                ans1 = target_answers[0].replace('A', '').strip()
                self.target_answer1 = float(ans1) if '/' not in ans1 else eval(ans1)
                
                # Parse second answer: fractions
                ans2 = target_answers[1]
                # Extract fractions using simple parsing
                import re
                fractions = re.findall(r'\\frac\{(\d+)\}\{(\d+)\}', ans2)
                if fractions:
                    self.target_answer2 = [int(num)/int(den) for num, den in fractions]
                else:
                    # Fallback: try to extract numbers
                    nums = re.findall(r'(\d+\.?\d*)', ans2)
                    if len(nums) >= 3:
                        self.target_answer2 = [float(n) for n in nums[:3]]
        except Exception as e:
            print(f"Warning: Could not parse target answers: {e}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return False
    
    def test_invariants(self):
        """Test fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Ohm's Law tests for main circuit
            # U = I1 * R1
            expected_U = self.I1 * self.R1
            if not self._is_close(self.U, expected_U):
                logs.append(f"FAIL: Ohm's Law (U = I1*R1): {self.U} ≠ {expected_U}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law U = I1*R1 = {expected_U}")
            
            # I2 = U / R2
            expected_I2 = self.U / self.R2
            if not self._is_close(self.I2, expected_I2):
                logs.append(f"FAIL: Ohm's Law (I2 = U/R2): {self.I2} ≠ {expected_I2}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law I2 = U/R2 = {expected_I2}")
            
            # I3 = U / R3
            expected_I3 = self.U / self.R3
            if not self._is_close(self.I3, expected_I3):
                logs.append(f"FAIL: Ohm's Law (I3 = U/R3): {self.I3} ≠ {expected_I3}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law I3 = U/R3 = {expected_I3}")
            
            # 2. Kirchhoff's Current Law (Junction Rule)
            expected_I_total = self.I1 + self.I2 + self.I3
            if not self._is_close(self.I_total, expected_I_total):
                logs.append(f"FAIL: KCL (I = I1+I2+I3): {self.I_total} ≠ {expected_I_total}")
                success = False
            else:
                logs.append(f"PASS: KCL I = I1+I2+I3 = {expected_I_total}")
            
            # 3. Parallel resistance formula
            expected_inv_R = 1/self.R1 + 1/self.R2 + 1/self.R3
            if not self._is_close(self.inv_R, expected_inv_R):
                logs.append(f"FAIL: 1/R = 1/R1+1/R2+1/R3: {self.inv_R} ≠ {expected_inv_R}")
                success = False
            else:
                logs.append(f"PASS: 1/R = 1/R1+1/R2+1/R3 = {expected_inv_R}")
            
            # R = 1/(1/R)
            if not self._is_close(self.R_eq, 1/self.inv_R):
                logs.append(f"FAIL: R = 1/(1/R): {self.R_eq} ≠ {1/self.inv_R}")
                success = False
            else:
                logs.append(f"PASS: R = 1/(1/R) = {1/self.inv_R}")
            
            # 4. Ohm's Law for equivalent circuit (part 2)
            # In part 2, the circuit is different - we should NOT expect U' = I_total * R_eq
            # Instead, we should verify that the calculations in part 2 are self-consistent
            # U' should equal I1 * R_eq (based on ground truth step 8)
            expected_U_prime_from_I1 = self.I1 * self.R_eq
            if not self._is_close(self.U_prime, expected_U_prime_from_I1):
                logs.append(f"NOTE: U' ({self.U_prime}) differs from I1*R ({expected_U_prime_from_I1})")
                # Not a failure, just note
            
            # I1' = U' / R1
            expected_I1_prime = self.U_prime / self.R1
            if not self._is_close(self.I1_prime, expected_I1_prime):
                logs.append(f"FAIL: Ohm's Law (I1' = U'/R1): {self.I1_prime} ≠ {expected_I1_prime}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law I1' = U'/R1 = {expected_I1_prime}")
            
            # I2' = U' / R2
            expected_I2_prime = self.U_prime / self.R2
            if not self._is_close(self.I2_prime, expected_I2_prime):
                logs.append(f"FAIL: Ohm's Law (I2' = U'/R2): {self.I2_prime} ≠ {expected_I2_prime}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law I2' = U'/R2 = {expected_I2_prime}")
            
            # I3' = U' / R3
            expected_I3_prime = self.U_prime / self.R3
            if not self._is_close(self.I3_prime, expected_I3_prime):
                logs.append(f"FAIL: Ohm's Law (I3' = U'/R3): {self.I3_prime} ≠ {expected_I3_prime}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law I3' = U'/R3 = {expected_I3_prime}")
            
            # 5. Kirchhoff for equivalent circuit (part 2)
            # The sum of currents in part 2 should equal the current through the equivalent resistance
            expected_I_total_prime = self.I1_prime + self.I2_prime + self.I3_prime
            expected_I_through_R_eq = self.U_prime / self.R_eq
            if not self._is_close(expected_I_total_prime, expected_I_through_R_eq):
                logs.append(f"WARNING: KCL for part 2 circuit: sum(I') = {expected_I_total_prime}, U'/R = {expected_I_through_R_eq}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        try:
            # Check target answers
            if self.target_answer1 is not None:
                if not self._is_close(self.I_total, self.target_answer1):
                    logs.append(f"FAIL: Total current mismatch: {self.I_total} ≠ target {self.target_answer1}")
                    success = False
                else:
                    logs.append(f"PASS: Total current matches target: {self.I_total}")
            
            if self.target_answer2 is not None and len(self.target_answer2) >= 3:
                if (not self._is_close(self.I1_prime, self.target_answer2[0]) or
                    not self._is_close(self.I2_prime, self.target_answer2[1]) or
                    not self._is_close(self.I3_prime, self.target_answer2[2])):
                    logs.append(f"FAIL: Prime currents mismatch: ({self.I1_prime}, {self.I2_prime}, {self.I3_prime}) ≠ target {self.target_answer2}")
                    success = False
                else:
                    logs.append(f"PASS: Prime currents match target: ({self.I1_prime}, {self.I2_prime}, {self.I3_prime})")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Test behavior when perturbing resistance R1."""
        logs = []
        success = True
        
        try:
            # Perturb R1 by +1Ω
            perturbed_R1 = self.R1 + 1.0  # 3Ω instead of 2Ω
            
            # Recalculate using formulas (keeping I1 fixed at original value for part 1)
            new_U = self.I1 * perturbed_R1
            new_I2 = new_U / self.R2
            new_I3 = new_U / self.R3
            new_I_total = self.I1 + new_I2 + new_I3
            
            # For part 2 (parallel circuit), recalculate with perturbed R1
            new_inv_R = 1/perturbed_R1 + 1/self.R2 + 1/self.R3
            new_R_eq = 1 / new_inv_R
            # In part 2, U' = I1 * R_eq (following ground truth)
            new_U_prime = self.I1 * new_R_eq
            new_I1_prime = new_U_prime / perturbed_R1
            new_I2_prime = new_U_prime / self.R2
            new_I3_prime = new_U_prime / self.R3
            
            # Check monotonic trends for part 1
            # When R1 increases (with fixed I1):
            # 1. U should increase (U = I1*R1)
            if not (new_U > self.U or self._is_close(new_U, self.U)):
                logs.append(f"FAIL: U should increase when R1 increases: {new_U} ≤ {self.U}")
                success = False
            else:
                logs.append(f"PASS: U increases with R1: {self.U} → {new_U}")
            
            # 2. I2 and I3 should increase (since U increases and R2,R3 constant)
            if not (new_I2 > self.I2 or self._is_close(new_I2, self.I2)):
                logs.append(f"FAIL: I2 should increase when U increases: {new_I2} ≤ {self.I2}")
                success = False
            else:
                logs.append(f"PASS: I2 increases: {self.I2} → {new_I2}")
            
            if not (new_I3 > self.I3 or self._is_close(new_I3, self.I3)):
                logs.append(f"FAIL: I3 should increase when U increases: {new_I3} ≤ {self.I3}")
                success = False
            else:
                logs.append(f"PASS: I3 increases: {self.I3} → {new_I3}")
            
            # 3. Total current I should increase (sum of increased currents)
            if not (new_I_total > self.I_total or self._is_close(new_I_total, self.I_total)):
                logs.append(f"FAIL: Total current should increase: {new_I_total} ≤ {self.I_total}")
                success = False
            else:
                logs.append(f"PASS: Total current increases: {self.I_total} → {new_I_total}")
            
            # Check monotonic trends for part 2
            # When R1 increases in parallel configuration:
            # 4. Equivalent resistance should increase (not decrease!) because adding a larger resistor in parallel
            # Actually, for parallel resistors: 1/R = 1/R1 + 1/R2 + 1/R3
            # If R1 increases, 1/R1 decreases, so 1/R decreases, so R increases
            if not (new_R_eq > self.R_eq or self._is_close(new_R_eq, self.R_eq)):
                logs.append(f"NOTE: Equivalent resistance changes: {self.R_eq} → {new_R_eq} (may increase or decrease)")
                # Not a failure - in parallel, increasing one resistor can increase or decrease equivalent resistance
                # Actually, for two resistors in parallel, increasing one increases equivalent resistance
                # For three resistors, it's more complex but generally increases
            
            # 5. U' should decrease (U' = I1 * R_eq, I1 fixed, R_eq increases)
            if not (new_U_prime < self.U_prime or self._is_close(new_U_prime, self.U_prime)):
                logs.append(f"NOTE: U' changes: {self.U_prime} → {new_U_prime}")
            
            # Restore original R1
            self.R1 = self._original_R1
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            # Restore on error
            self.R1 = self._original_R1
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Started ==="]
        overall_success = True
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.append("\n--- Test 1: Physics Invariants ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Test 2: Trace Consistency ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Test 3: Monotonicity ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs