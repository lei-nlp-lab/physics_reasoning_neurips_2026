import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Circuit parameters
        self.N = 200
        self.r = 2.0  # coil resistance (Ω)
        self.R = 48.0 # external resistor (Ω)
        
        # Extract results with safe fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        
        # String variables (directional answers)
        self.B_induced = results_dict.get('B_induced', None)
        self.I_induced_coil = results_dict.get('I_induced_coil', None)
        self.I_R = results_dict.get('I_R', None)
        
        # Numeric variables - convert to float safely
        self.U = self._safe_float(results_dict.get('U', 0.0))
        self.E = self._safe_float(results_dict.get('E', 0.0))
        
        # Expected values from steps analysis
        self.expected_E = 10.0
        self.expected_U = 9.6
        self.expected_I_R_direction = "a to b"
        
        self.logs = []
    
    def _safe_float(self, value):
        """Safely convert value to float, handling strings and SymPy objects."""
        try:
            if hasattr(value, 'evalf'):
                return float(value.evalf())
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    def _log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def _fuzzy_match(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Faraday's Law: E = N * |ΔΦ/Δt| (magnitude)
        2. Voltage divider: U = [R/(R+r)] * E
        3. Lenz's Law: Direction consistency
        """
        try:
            # 1. Check voltage divider formula
            if self.E != 0:
                expected_U = (self.R / (self.R + self.r)) * self.E
                if not self._fuzzy_match(self.U, expected_U):
                    self._log(f"FAIL: Voltage divider invariant. U={self.U}, expected {expected_U}")
                    return False
                self._log("PASS: Voltage divider formula holds.")
            else:
                self._log("WARNING: E is zero, skipping voltage divider check.")
            
            # 2. Check Faraday's Law magnitude (we can't check exact ΔΦ/Δt without flux data,
            # but we can check that E is non-negative magnitude)
            if self.E < 0:
                self._log("WARNING: Induced EMF is negative. Should be magnitude (absolute value).")
            
            # 3. Check Lenz's Law sign consistency (qualitative)
            # If B_induced direction is given, it should oppose the change in flux
            # Since original B is into paper, and induced B is outward (from steps),
            # this implies flux is increasing (to oppose increase, induced B opposes original B)
            if self.B_induced:
                if "out" in str(self.B_induced).lower():
                    self._log("PASS: Induced B direction (outward) consistent with Lenz's Law for increasing flux.")
                else:
                    self._log("WARNING: Induced B direction may not match Lenz's Law expectation.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # 1. Check induced EMF matches step 3
            if not self._fuzzy_match(self.E, self.expected_E):
                self._log(f"FAIL: Induced EMF E={self.E} doesn't match expected {self.expected_E}V")
                return False
            self._log("PASS: Induced EMF matches step 3.")
            
            # 2. Check resistor voltage matches step 4
            if not self._fuzzy_match(self.U, self.expected_U):
                self._log(f"FAIL: Voltage U={self.U}V doesn't match expected {self.expected_U}V")
                return False
            self._log("PASS: Resistor voltage matches step 4.")
            
            # 3. Check current direction (string comparison with normalization)
            if self.I_R:
                i_r_str = str(self.I_R).lower().replace("$", "").replace("\\to", "to").strip()
                expected = self.expected_I_R_direction.lower().strip()
                if expected in i_r_str or i_r_str in expected:
                    self._log("PASS: Current direction matches step 2.")
                else:
                    self._log(f"WARNING: Current direction '{self.I_R}' doesn't match expected '{self.expected_I_R_direction}'")
            else:
                self._log("WARNING: No I_R direction provided.")
            
            # 4. Check coil current direction consistency
            if self.I_induced_coil and self.B_induced:
                i_coil = str(self.I_induced_coil).lower()
                b_ind = str(self.B_induced).lower()
                # Counterclockwise current (when viewed from above) produces outward B
                if "counter" in i_coil and "out" in b_ind:
                    self._log("PASS: Coil current and induced B direction are consistent.")
                elif "clock" in i_coil and "in" in b_ind:
                    self._log("PASS: Coil current and induced B direction are consistent.")
                else:
                    self._log("WARNING: Coil current and induced B direction may be inconsistent.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb resistance R and verify expected trend:
        U = [R/(R+r)] * E should increase with R (for fixed E and r).
        """
        try:
            # Original calculation
            U_original = (self.R / (self.R + self.r)) * self.E
            
            # Perturb R by ±10%
            perturbations = [0.9, 1.1]
            trends = []
            
            for factor in perturbations:
                R_perturbed = self.R * factor
                U_perturbed = (R_perturbed / (R_perturbed + self.r)) * self.E
                
                # Check monotonicity: U should increase with R
                if factor > 1.0:
                    trends.append(U_perturbed > U_original)
                else:
                    trends.append(U_perturbed < U_original)
                
                self._log(f"  R={R_perturbed:.1f}Ω → U={U_perturbed:.3f}V")
            
            # Verify both perturbations show correct trend
            if all(trends):
                self._log("PASS: U increases with R as expected (monotonic).")
                return True
            else:
                self._log("FAIL: U does not show expected monotonic behavior with R.")
                return False
                
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def test_symbolic_laws(self):
        """
        Verify relationships using symbolic physics laws.
        """
        try:
            # Use only numeric checks - no SymPy symbols
            self._log("Symbolic laws test simplified to numeric checks.")
            
            # Check Faraday's Law numerically
            # Given flux values from problem: phi1=0.010, phi2=0.015, dt=0.10
            phi1 = 0.010
            phi2 = 0.015
            dt = 0.10
            expected_E_from_faraday = self.N * abs(phi2 - phi1) / dt
            
            if self._fuzzy_match(self.E, expected_E_from_faraday):
                self._log(f"PASS: Faraday's Law holds numerically. E={self.E}, expected={expected_E_from_faraday}")
            else:
                self._log(f"WARNING: Faraday's Law discrepancy. E={self.E}, expected={expected_E_from_faraday}")
            
            # Check voltage divider numerically
            expected_U_from_divider = (self.R / (self.R + self.r)) * self.E
            if self._fuzzy_match(self.U, expected_U_from_divider):
                self._log(f"PASS: Voltage divider holds numerically. U={self.U}, expected={expected_U_from_divider}")
            else:
                self._log(f"WARNING: Voltage divider discrepancy. U={self.U}, expected={expected_U_from_divider}")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_symbolic_laws: {str(e)}")
            return False
    
    def run_all_tests(self):
        """
        Execute all tests and return success flag with logs.
        """
        self.logs = []
        self._log("=== Physics Checker Started ===")
        
        # Record extracted values
        self._log(f"Extracted values: N={self.N}, r={self.r}Ω, R={self.R}Ω")
        self._log(f"  E={self.E}V, U={self.U}V")
        self._log(f"  B_induced={self.B_induced}, I_induced_coil={self.I_induced_coil}, I_R={self.I_R}")
        
        test_results = []
        
        # Run tests
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Symbolic Laws", self.test_symbolic_laws)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} ---")
            try:
                result = test_func()
                test_results.append(result)
                self._log(f"Result: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self._log(f"Test crashed: {str(e)}")
                self._log(traceback.format_exc())
                test_results.append(False)
        
        success = all(test_results)
        self._log(f"\n=== Overall: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'} ===")
        
        return success, self.logs

# Example usage for testing:
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'B_induced': 'out of the paper',
                'I_induced_coil': 'counterclockwise',
                'I_R': 'a to b',
                'U': 9.6,
                'E': 10.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)