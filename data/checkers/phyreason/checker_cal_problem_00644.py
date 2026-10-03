import math
import sympy as sp
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Expected keys in solution_instance.results:
        - 'a_1': acceleration before collision
        - 'F': force before collision
        - 'a_2/a_1': ratio of accelerations
        - 'a_2': acceleration after collision
        - 'F'': force after collision
        - 'q_final': charge on each sphere after collision
        """
        # Safely extract all results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract numerical results with safe defaults
        self.a1 = getattr(solution_instance, 'a_1', self.results.get('a_1', 0.0))
        self.F_initial = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        self.ratio = getattr(solution_instance, 'a_2/a_1', self.results.get('a_2/a_1', 0.0))
        self.a2 = getattr(solution_instance, 'a_2', self.results.get('a_2', 0.0))
        self.F_final = getattr(solution_instance, "F''", self.results.get("F''", 0.0))
        self.q_final = getattr(solution_instance, 'q_final', self.results.get('q_final', 0.0))
        
        # Extract inputs safely
        self.q1_initial = getattr(solution_instance, 'q1_initial', 8.0)  # 8q
        self.q2_initial = getattr(solution_instance, 'q2_initial', -2.0) # -2q
        self.distance = getattr(solution_instance, 'distance', 1.0)      # l
        self.mass = getattr(solution_instance, 'mass', 1.0)              # m
        self.k = getattr(solution_instance, 'k', 8.987551787e9)          # Coulomb's constant
        self.q_base = getattr(solution_instance, 'q_base', 1.0)          # base charge q
        
        # For symbolic checks
        self.symbols = {
            'q': sp.symbols('q', positive=True),
            'l': sp.symbols('l', positive=True),
            'm': sp.symbols('m', positive=True),
            'k': sp.symbols('k', positive=True)
        }
        
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "PASS" if success else "FAIL"
            self.logs.append(f"{prefix}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
        return success
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        success = True
        
        # 1. Conservation of charge during contact
        total_initial = self.q1_initial + self.q2_initial  # 8q - 2q = 6q
        total_final = 2 * self.q_final  # Each sphere has q_final after collision
        
        if not math.isclose(total_initial, total_final, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Charge conservation failed: initial={total_initial}, final={total_final}", False)
        else:
            self._log(f"Charge conservation passed: {total_initial} = {total_final}", True)
        
        # 2. Newton's Second Law consistency (F = m*a)
        # Before collision
        if self.a1 != 0.0 and self.F_initial != 0.0:
            F_calc = self.mass * self.a1
            if not math.isclose(self.F_initial, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Newton's 2nd law (before) failed: F={self.F_initial}, m*a={F_calc}", False)
            else:
                self._log(f"Newton's 2nd law (before) passed", True)
        
        # After collision
        if self.a2 != 0.0 and self.F_final != 0.0:
            F_calc = self.mass * self.a2
            if not math.isclose(self.F_final, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Newton's 2nd law (after) failed: F'={self.F_final}, m*a'={F_calc}", False)
            else:
                self._log(f"Newton's 2nd law (after) passed", True)
        
        # 3. Coulomb's Law consistency
        q1_abs = abs(self.q1_initial)
        q2_abs = abs(self.q2_initial)
        F_coulomb_initial = self.k * q1_abs * q2_abs / (self.distance ** 2)
        
        if self.F_initial != 0.0:
            if not math.isclose(self.F_initial, F_coulomb_initial, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Coulomb's law (initial) failed: F={self.F_initial}, k*q1*q2/r²={F_coulomb_initial}", False)
            else:
                self._log(f"Coulomb's law (initial) passed", True)
        
        # Check final force
        q_final_abs = abs(self.q_final)
        F_coulomb_final = self.k * q_final_abs * q_final_abs / (self.distance ** 2)
        
        if self.F_final != 0.0:
            if not math.isclose(self.F_final, F_coulomb_final, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Coulomb's law (final) failed: F'={self.F_final}, k*q_final²/r²={F_coulomb_final}", False)
            else:
                self._log(f"Coulomb's law (final) passed", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        success = True
        
        # 1. Check that q_final = (8q - 2q)/2 = 3q
        expected_q_final = (self.q1_initial + self.q2_initial) / 2
        if not math.isclose(self.q_final, expected_q_final, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Final charge mismatch: expected {expected_q_final}, got {self.q_final}", False)
        else:
            self._log(f"Final charge correct: {self.q_final}", True)
        
        # 2. Check the ratio a2/a1 = 9/16
        expected_ratio = 9.0/16.0
        if not math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Ratio mismatch: expected {expected_ratio}, got {self.ratio}", False)
        else:
            self._log(f"Ratio correct: {self.ratio}", True)
        
        # 3. Verify a2 = ratio * a1
        if self.a1 != 0.0 and self.a2 != 0.0:
            a2_calc = self.ratio * self.a1
            if not math.isclose(self.a2, a2_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Acceleration consistency failed: a2={self.a2}, ratio*a1={a2_calc}", False)
            else:
                self._log(f"Acceleration consistency passed", True)
        
        # 4. Check that F'/F = 9/16 (since m and l cancel)
        if self.F_initial != 0.0 and self.F_final != 0.0:
            force_ratio = self.F_final / self.F_initial
            if not math.isclose(force_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Force ratio mismatch: expected {expected_ratio}, got {force_ratio}", False)
            else:
                self._log(f"Force ratio correct: {force_ratio}", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and verify the trend.
        We'll perturb the initial charge of sphere A from 8q to 9q.
        """
        success = True
        
        # Save original state
        original_q1 = self.q1_initial
        
        try:
            # Perturb: increase q1 from 8q to 9q
            self.q1_initial = 9.0 * self.q_base
            
            # Recalculate expected values based on physics
            # After collision: q_final' = (9q - 2q)/2 = 3.5q
            q_final_perturbed = (self.q1_initial + self.q2_initial) / 2
            
            # Ratio: a2'/a1' = (q_final'²) / (q1_initial' * |q2_initial|)
            # q1_initial' = 9q, q2_initial = -2q (use absolute value 2q for magnitude)
            ratio_perturbed = (q_final_perturbed ** 2) / (9.0 * self.q_base * 2.0 * self.q_base)
            # Simplify: (3.5q)² / (18q²) = 12.25/18 = 0.68056...
            expected_perturbed_ratio = 12.25 / 18.0  # Approximately 0.68056
            
            # Original ratio was 9/16 = 0.5625
            # Since 0.68056 > 0.5625, ratio should INCREASE when q1 increases
            
            if not math.isclose(ratio_perturbed, expected_perturbed_ratio, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                self._log(f"Perturbed ratio calculation failed: {ratio_perturbed} vs {expected_perturbed_ratio}", False)
            else:
                # Check trend: increasing q1 should increase the ratio
                original_ratio = 9.0/16.0
                if ratio_perturbed > original_ratio:
                    self._log(f"Monotonicity passed: ratio increases from {original_ratio:.4f} to {ratio_perturbed:.4f} when q1 increases", True)
                else:
                    success = False
                    self._log(f"Monotonicity failed: ratio decreased when q1 increased", False)
        
        except Exception as e:
            success = False
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
        
        finally:
            # Restore original state
            self.q1_initial = original_q1
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        self._log("=== Physics Checker Starting ===")
        
        # Run individual tests
        test_results = []
        
        self._log("\n--- Testing Invariants ---")
        test_results.append(("Invariants", self.test_invariants()))
        
        self._log("\n--- Testing Trace Consistency ---")
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        
        self._log("\n--- Testing Monotonicity ---")
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Determine overall success
        for test_name, test_success in test_results:
            if not test_success:
                overall_success = False
                self._log(f"Test '{test_name}' failed", False)
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs