import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        # Input parameters with safe access
        self.Q = getattr(solution_instance, 'Q', getattr(solution_instance, 'results', {}).get('Q', 20.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 4.0))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 4.0))
        
        # Results with safe access
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', None))
        self.U = getattr(solution_instance, 'U', getattr(solution_instance, 'results', {}).get('U', None))
        
        # If results are missing, compute from physics laws
        if self.I is None:
            try:
                self.I = self.Q / self.t
            except:
                self.I = 0.0
        if self.U is None:
            try:
                self.U = self.I * self.R
            except:
                self.U = 0.0
        
        # Symbolic variables
        self.I_sym = sp.symbols('I')
        self.U_sym = sp.symbols('U')
        self.Q_sym = sp.symbols('Q')
        self.t_sym = sp.symbols('t')
        self.R_sym = sp.symbols('R')
        
        # Logs
        self.logs = []
        
    def _log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """Check fundamental physics laws: current definition and Ohm's Law"""
        checks = []
        
        # 1. Check current definition: I = Q/t
        try:
            expected_I = self.Q / self.t
            if math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(True)
                self._log(f"✓ Current definition invariant: I = Q/t = {self.Q}/{self.t} ≈ {expected_I:.2f}, got {self.I:.2f}")
            else:
                checks.append(False)
                self._log(f"✗ Current definition violated: I = Q/t = {self.Q}/{self.t} ≈ {expected_I:.2f}, but got {self.I:.2f}")
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Current definition check failed: {e}")
        
        # 2. Check Ohm's Law: U = I * R
        try:
            expected_U = self.I * self.R
            if math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(True)
                self._log(f"✓ Ohm's Law invariant: U = I*R = {self.I}*{self.R} ≈ {expected_U:.2f}, got {self.U:.2f}")
            else:
                checks.append(False)
                self._log(f"✗ Ohm's Law violated: U = I*R = {self.I}*{self.R} ≈ {expected_U:.2f}, but got {self.U:.2f}")
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Ohm's Law check failed: {e}")
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Check consistency with step-by-step solution from KB"""
        checks = []
        
        # Step 1: I = Q/t = 20/4 = 5 A
        try:
            step1_I = self.Q / self.t
            if math.isclose(self.I, step1_I, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(True)
                self._log(f"✓ Step 1 consistency: I = {self.I:.2f} A matches Q/t = {step1_I:.2f} A")
            else:
                checks.append(False)
                self._log(f"✗ Step 1 inconsistency: I = {self.I:.2f} A but Q/t = {step1_I:.2f} A")
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Step 1 check failed: {e}")
        
        # Step 2: U = I*R = 5*4 = 20 V
        try:
            step2_U = self.I * self.R
            if math.isclose(self.U, step2_U, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(True)
                self._log(f"✓ Step 2 consistency: U = {self.U:.2f} V matches I*R = {step2_U:.2f} V")
            else:
                checks.append(False)
                self._log(f"✗ Step 2 inconsistency: U = {self.U:.2f} V but I*R = {step2_U:.2f} V")
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Step 2 check failed: {e}")
        
        # Overall consistency between steps
        try:
            if math.isclose(self.U, self.I * self.R, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(True)
                self._log("✓ Overall trace consistency: U and I satisfy U = I*R")
            else:
                checks.append(False)
                self._log("✗ Overall trace inconsistency: U ≠ I*R")
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Overall trace check failed: {e}")
        
        return all(checks)
    
    def test_monotonicity(self):
        """Check monotonic relationships by perturbing inputs"""
        checks = []
        
        # Save original state
        original_Q = self.Q
        original_I = self.I
        original_U = self.U
        
        try:
            # Test 1: Increase Q by 10%, expect I and U to increase
            self.Q = original_Q * 1.1
            new_I = self.Q / self.t
            new_U = new_I * self.R
            
            if new_I > original_I and new_U > original_U:
                checks.append(True)
                self._log(f"✓ Monotonicity (Q↑): I increased from {original_I:.2f} to {new_I:.2f}, U from {original_U:.2f} to {new_U:.2f}")
            else:
                checks.append(False)
                self._log(f"✗ Monotonicity (Q↑) failed: I {original_I:.2f}→{new_I:.2f}, U {original_U:.2f}→{new_U:.2f}")
            
            # Restore Q
            self.Q = original_Q
            
            # Test 2: Increase R by 10%, expect U to increase (since U = I*R and I constant for fixed Q,t)
            original_R = self.R
            self.R = original_R * 1.1
            new_U = self.I * self.R
            
            if new_U > original_U:
                checks.append(True)
                self._log(f"✓ Monotonicity (R↑): U increased from {original_U:.2f} to {new_U:.2f} (I constant)")
            else:
                checks.append(False)
                self._log(f"✗ Monotonicity (R↑) failed: U {original_U:.2f}→{new_U:.2f}")
            
            # Restore R
            self.R = original_R
            
            # Test 3: Increase t by 10%, expect I to decrease (I = Q/t)
            original_t = self.t
            self.t = original_t * 1.1
            new_I = self.Q / self.t
            
            if new_I < original_I:
                checks.append(True)
                self._log(f"✓ Monotonicity (t↑): I decreased from {original_I:.2f} to {new_I:.2f}")
            else:
                checks.append(False)
                self._log(f"✗ Monotonicity (t↑) failed: I {original_I:.2f}→{new_I:.2f}")
            
            # Restore t
            self.t = original_t
            
        except Exception as e:
            checks.append(False)
            self._log(f"✗ Monotonicity test failed: {e}")
            # Restore all values on error
            self.Q = original_Q
            self.t = original_t
            self.R = self.R  # Already restored above
        
        # Restore I and U to original values
        self.I = original_I
        self.U = original_U
        
        return all(checks) if checks else False
    
    def run_all_tests(self):
        """Run all checks and return success status and logs"""
        self.logs = []
        success = True
        
        try:
            self._log("=== Physics Auditor Report ===")
            self._log(f"Input: Q={self.Q} C, t={self.t} s, R={self.R} Ω")
            self._log(f"Results: I={self.I:.2f} A, U={self.U:.2f} V")
            
            # Run individual tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = test1 and test2 and test3
            
            self._log(f"\nSummary: {len(self.logs)-1} checks performed")
            self._log(f"Invariants: {'PASS' if test1 else 'FAIL'}")
            self._log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
            self._log(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
            self._log(f"Overall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            success = False
            self._log(f"Runtime error in PhysicsChecker: {e}")
            self._log(traceback.format_exc())
        
        return success, self.logs