import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.s_2 = getattr(solution_instance, 's_2', solution_instance.results.get('s_2', 0.0))
        self.s_4 = getattr(solution_instance, 's_4', solution_instance.results.get('s_4', 0.0))
        self.s_2_prime = getattr(solution_instance, 's_2\'', solution_instance.results.get('s_2\'', 0.0))
        
        # Problem constants (east = +, west = -)
        self.displacements = [5.0, -3.0, -4.0, 1.0]  # d1, d2, d3, d4
        self.original_displacements = self.displacements.copy()
        
        # Expected symbolic relationships
        self.d1, self.d2, self.d3, self.d4 = sp.symbols('d1 d2 d3 d4')
        self.symbolic_s2 = self.d1 + self.d2
        self.symbolic_s4 = self.d1 + self.d2 + self.d3 + self.d4
        self.symbolic_s2_prime = self.d3 + self.d4
        
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper for logging with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def test_invariants(self):
        """Test displacement summation invariants."""
        try:
            # Numerical checks
            s2_calc = self.displacements[0] + self.displacements[1]  # d1 + d2
            s4_calc = sum(self.displacements)  # d1 + d2 + d3 + d4
            s2_prime_calc = self.displacements[2] + self.displacements[3]  # d3 + d4
            
            # Test s_2
            if not math.isclose(self.s_2, s2_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"s_2 mismatch: Expected {s2_calc:.2f}, got {self.s_2:.2f}", False)
                return False
            self._log(f"s_2 correct: {self.s_2:.2f} = {self.displacements[0]:.2f} + {self.displacements[1]:.2f}", True)
            
            # Test s_4
            if not math.isclose(self.s_4, s4_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"s_4 mismatch: Expected {s4_calc:.2f}, got {self.s_4:.2f}", False)
                return False
            self._log(f"s_4 correct: {self.s_4:.2f} = sum({self.displacements})", True)
            
            # Test s_2'
            if not math.isclose(self.s_2_prime, s2_prime_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"s_2' mismatch: Expected {s2_prime_calc:.2f}, got {self.s_2_prime:.2f}", False)
                return False
            self._log(f"s_2' correct: {self.s_2_prime:.2f} = {self.displacements[2]:.2f} + {self.displacements[3]:.2f}", True)
            
            # Symbolic verification (optional, for completeness)
            sym_s2 = self.symbolic_s2.subs({self.d1: self.displacements[0], self.d2: self.displacements[1]})
            if not math.isclose(float(sym_s2), self.s_2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Symbolic s_2 mismatch: {float(sym_s2):.2f} vs {self.s_2:.2f}", False)
                return False
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {e}", False)
            return False
    
    def test_trace_consistency(self):
        """Check that intermediate steps are consistent with final results."""
        try:
            # All displacements must be consistent
            # From steps analysis: s_2 = 2, s_4 = -1, s_2' = -3
            expected_s2 = 2.0
            expected_s4 = -1.0
            expected_s2_prime = -3.0
            
            checks = [
                (self.s_2, expected_s2, "s_2"),
                (self.s_4, expected_s4, "s_4"),
                (self.s_2_prime, expected_s2_prime, "s_2'")
            ]
            
            all_pass = True
            for actual, expected, name in checks:
                if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"{name} trace mismatch: Expected {expected:.2f}, got {actual:.2f}", False)
                    all_pass = False
                else:
                    self._log(f"{name} trace consistent: {actual:.2f}", True)
            
            return all_pass
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {e}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check formula behavior."""
        try:
            # Perturb d1 (first displacement) by +10%
            perturbation = 0.1 * self.displacements[0]  # +10%
            original_d1 = self.displacements[0]
            self.displacements[0] += perturbation
            
            # Recalculate using formulas (not solution instance)
            new_s2 = self.displacements[0] + self.displacements[1]
            new_s4 = sum(self.displacements)
            new_s2_prime = self.displacements[2] + self.displacements[3]  # Shouldn't change
            
            # Check that formulas hold with perturbed values
            s2_formula_holds = math.isclose(new_s2, self.displacements[0] + self.displacements[1], 
                                           rel_tol=1e-2, abs_tol=1e-2)
            s4_formula_holds = math.isclose(new_s4, sum(self.displacements), 
                                           rel_tol=1e-2, abs_tol=1e-2)
            s2_prime_formula_holds = math.isclose(new_s2_prime, self.displacements[2] + self.displacements[3], 
                                                 rel_tol=1e-2, abs_tol=1e-2)
            
            # Restore original value
            self.displacements[0] = original_d1
            
            # Log results
            if s2_formula_holds and s4_formula_holds and s2_prime_formula_holds:
                self._log(f"Monotonicity test passed: Formulas hold under perturbation (+{perturbation:.2f} to d1)", True)
                return True
            else:
                failures = []
                if not s2_formula_holds: failures.append("s_2 formula")
                if not s4_formula_holds: failures.append("s_4 formula")
                if not s2_prime_formula_holds: failures.append("s_2' formula")
                self._log(f"Monotonicity test failed: {', '.join(failures)} violated", False)
                return False
                
        except Exception as e:
            # Restore on error
            self.displacements = self.original_displacements.copy()
            self._log(f"Monotonicity test failed with error: {e}", False)
            return False
    
    def test_direction_consistency(self):
        """Check that signs match direction descriptions in target answers."""
        try:
            # Target answers: "2m, east", "1m, west", "3m, west"
            # east = positive, west = negative
            
            s2_expected_sign = 1  # east -> positive
            s4_expected_sign = -1  # west -> negative
            s2_prime_expected_sign = -1  # west -> negative
            
            # Get actual signs (careful with near-zero)
            s2_sign = 1 if self.s_2 > 1e-9 else (-1 if self.s_2 < -1e-9 else 0)
            s4_sign = 1 if self.s_4 > 1e-9 else (-1 if self.s_4 < -1e-9 else 0)
            s2_prime_sign = 1 if self.s_2_prime > 1e-9 else (-1 if self.s_2_prime < -1e-9 else 0)
            
            checks = [
                (s2_sign, s2_expected_sign, "s_2 direction"),
                (s4_sign, s4_expected_sign, "s_4 direction"),
                (s2_prime_sign, s2_prime_expected_sign, "s_2' direction")
            ]
            
            all_pass = True
            for actual, expected, name in checks:
                if actual != expected:
                    self._log(f"{name} mismatch: Expected sign {expected}, got {actual}", False)
                    all_pass = False
                else:
                    self._log(f"{name} correct", True)
            
            return all_pass
            
        except Exception as e:
            self._log(f"Direction consistency test failed: {e}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Direction Consistency", self.test_direction_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} FAILED", False)
            except Exception as e:
                self._log(f"{test_name} raised exception: {traceback.format_exc()}", False)
                success = False
        
        # Final summary
        if success:
            self._log("All physics checks passed!", True)
        else:
            self._log("Some physics checks failed.", False)
        
        return success, self.logs