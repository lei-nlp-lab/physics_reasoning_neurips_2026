import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution instance
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'results', {}).get('n', 400))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'results', {}).get('B', 0.25))
        self.L1 = getattr(solution_instance, 'L1', getattr(solution_instance, 'results', {}).get('L1', 0.05))
        self.L2 = getattr(solution_instance, 'L2', getattr(solution_instance, 'results', {}).get('L2', 0.04))
        self.omega = getattr(solution_instance, 'omega', getattr(solution_instance, 'results', {}).get('omega', 100))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'results', {}).get('r', 0.1))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 9.9))
        
        # Safely extract computed results with defaults
        self.E_m = getattr(solution_instance, 'E_m', getattr(solution_instance, 'results', {}).get('E_m', 0.0))
        self.I_m = getattr(solution_instance, 'I_m', getattr(solution_instance, 'results', {}).get('I_m', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        
        # Store original values for monotonicity test restoration
        self._original_omega = self.omega
        self._original_E_m = self.E_m
        
        # Initialize test logs
        self.logs = []

    def _log_success(self, test_name):
        self.logs.append(f"✓ {test_name}")
        
    def _log_failure(self, test_name, message):
        self.logs.append(f"✗ {test_name}: {message}")

    def test_invariants(self):
        """Verify conservation laws and physical principles."""
        try:
            # 1. Check maximum EMF formula: E_m = n * B * A * omega
            A = self.L1 * self.L2
            expected_E_m = self.n * self.B * A * self.omega
            if math.isclose(self.E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log_success("Max EMF formula")
            else:
                self._log_failure("Max EMF formula", 
                                 f"E_m={self.E_m}, expected={expected_E_m}")
                return False
            
            # 2. Check Ohm's Law: I_m = E_m / (R + r)
            expected_I_m = self.E_m / (self.R + self.r)
            if math.isclose(self.I_m, expected_I_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log_success("Ohm's Law (max current)")
            else:
                self._log_failure("Ohm's Law (max current)", 
                                 f"I_m={self.I_m}, expected={expected_I_m}")
                return False
            
            # 3. Check AC effective value: I = I_m / sqrt(2)
            expected_I = self.I_m / math.sqrt(2)
            if math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log_success("AC effective current")
            else:
                self._log_failure("AC effective current", 
                                 f"I={self.I}, expected={expected_I}")
                return False
            
            # 4. Check power formula: P = I² * R
            expected_P = self.I**2 * self.R
            if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
                self._log_success("Power formula")
            else:
                self._log_failure("Power formula", 
                                 f"P={self.P}, expected={expected_P}")
                return False
            
            # 5. Symbolic verification of fundamental laws
            t = sp.symbols('t')
            N, B, A, omega = sp.symbols('N B A omega')
            emf_max_expr = N * B * A * omega
            # Substitute with actual values
            symbolic_E_m = emf_max_expr.subs({
                N: self.n, 
                B: self.B, 
                A: self.L1 * self.L2, 
                omega: self.omega
            })
            if math.isclose(self.E_m, float(symbolic_E_m), rel_tol=1e-2, abs_tol=1e-2):
                self._log_success("Symbolic EMF law")
            else:
                self._log_failure("Symbolic EMF law", 
                                 f"Symbolic mismatch: {float(symbolic_E_m)} vs {self.E_m}")
                return False
            
            return True
            
        except Exception as e:
            self._log_failure("Invariants test", f"Exception: {str(e)}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        try:
            # Step-by-step consistency from STEPS ANALYSIS
            steps_valid = True
            
            # Step 1 already verified in invariants
            A = self.L1 * self.L2
            E_m_step1 = self.n * self.B * A * self.omega
            if not math.isclose(self.E_m, E_m_step1, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Trace step 1", "E_m doesn't match step 1 calculation")
                steps_valid = False
            
            # Step 2: I_m = E_m / (R + r)
            I_m_step2 = self.E_m / (self.R + self.r)
            if not math.isclose(self.I_m, I_m_step2, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Trace step 2", "I_m doesn't match step 2 calculation")
                steps_valid = False
            
            # Step 3: I = I_m / sqrt(2)
            I_step3 = self.I_m / math.sqrt(2)
            if not math.isclose(self.I, I_step3, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Trace step 3", "I doesn't match step 3 calculation")
                steps_valid = False
            
            # Step 4: P = I² * R
            P_step4 = self.I**2 * self.R
            if not math.isclose(self.P, P_step4, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Trace step 4", "P doesn't match step 4 calculation")
                steps_valid = False
            
            if steps_valid:
                self._log_success("Trace consistency")
                return True
            else:
                return False
                
        except Exception as e:
            self._log_failure("Trace consistency", f"Exception: {str(e)}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Test that increasing omega increases E_m, I_m, I, and P proportionally."""
        try:
            # Store original state
            original_omega = self.omega
            original_E_m = self.E_m
            original_I_m = self.I_m
            original_I = self.I
            original_P = self.P
            
            # Perturb omega by +10%
            perturb_factor = 1.10
            self.omega *= perturb_factor
            
            # Recalculate using formulas (not calling solution methods)
            A = self.L1 * self.L2
            new_E_m = self.n * self.B * A * self.omega
            new_I_m = new_E_m / (self.R + self.r)
            new_I = new_I_m / math.sqrt(2)
            new_P = new_I**2 * self.R
            
            # Check monotonic trends
            trends_valid = True
            
            # E_m should increase by same factor as omega (direct proportionality)
            expected_E_m_ratio = perturb_factor
            actual_E_m_ratio = new_E_m / original_E_m
            if not math.isclose(actual_E_m_ratio, expected_E_m_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Monotonicity E_m", 
                                 f"Ratio {actual_E_m_ratio} ≠ {expected_E_m_ratio}")
                trends_valid = False
            
            # I_m should increase proportionally to E_m (Ohm's Law with constant R_total)
            expected_I_m_ratio = perturb_factor
            actual_I_m_ratio = new_I_m / original_I_m
            if not math.isclose(actual_I_m_ratio, expected_I_m_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Monotonicity I_m", 
                                 f"Ratio {actual_I_m_ratio} ≠ {expected_I_m_ratio}")
                trends_valid = False
            
            # I should increase proportionally to I_m
            expected_I_ratio = perturb_factor
            actual_I_ratio = new_I / original_I
            if not math.isclose(actual_I_ratio, expected_I_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Monotonicity I", 
                                 f"Ratio {actual_I_ratio} ≠ {expected_I_ratio}")
                trends_valid = False
            
            # P should increase by square of current ratio (P ∝ I²)
            expected_P_ratio = perturb_factor**2
            actual_P_ratio = new_P / original_P
            if not math.isclose(actual_P_ratio, expected_P_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log_failure("Monotonicity P", 
                                 f"Ratio {actual_P_ratio} ≠ {expected_P_ratio}")
                trends_valid = False
            
            # Restore original values
            self.omega = original_omega
            self.E_m = original_E_m
            self.I_m = original_I_m
            self.I = original_I
            self.P = original_P
            
            if trends_valid:
                self._log_success("Monotonicity")
                return True
            else:
                return False
                
        except Exception as e:
            # Restore on exception
            self.omega = self._original_omega
            self.E_m = self._original_E_m
            self._log_failure("Monotonicity", f"Exception: {str(e)}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            if not test_func():
                success = False
        
        return success, self.logs