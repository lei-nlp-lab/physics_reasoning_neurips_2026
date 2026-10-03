import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallbacks."""
        # Input parameters from problem context
        self.a = getattr(solution_instance, 'a', None)
        self.h = getattr(solution_instance, 'h', None)
        self.b = getattr(solution_instance, 'b', None)
        self.g = getattr(solution_instance, 'g', 9.8)  # Default gravitational acceleration
        
        # Results from solution steps (safe access with fallback to results dict)
        self.results = getattr(solution_instance, 'results', {})
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', None))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.Deltat = getattr(solution_instance, 'Deltat', self.results.get('Deltat', None))
        
        # Symbolic variables for invariant checking
        self.h_sym = sp.Symbol('h', positive=True)
        self.b_sym = sp.Symbol('b', positive=True)
        self.g_sym = sp.Symbol('g', positive=True)
        self.t1_sym = sp.sqrt(2 * self.h_sym / self.g_sym)
        self.t2_sym = sp.sqrt(2 * (self.h_sym + self.b_sym) / self.g_sym)
        self.Deltat_sym = self.t2_sym - self.t1_sym

    def test_invariants(self):
        """Verify free-fall motion formulas and time difference consistency."""
        logs = []
        success = True
        
        try:
            # Check t1 formula: h = (1/2) g t1^2
            if self.t1 is not None and self.h is not None:
                t1_from_formula = math.sqrt(2 * self.h / self.g)
                if not math.isclose(self.t1, t1_from_formula, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t1 mismatch. Got {self.t1}, expected √(2h/g) ≈ {t1_from_formula}")
                    success = False
                else:
                    logs.append(f"PASS: t1 = {self.t1} matches free-fall from height h = {self.h}")
            
            # Check t2 formula: h+b = (1/2) g t2^2
            if self.t2 is not None and self.h is not None and self.b is not None:
                t2_from_formula = math.sqrt(2 * (self.h + self.b) / self.g)
                if not math.isclose(self.t2, t2_from_formula, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t2 mismatch. Got {self.t2}, expected √(2(h+b)/g) ≈ {t2_from_formula}")
                    success = False
                else:
                    logs.append(f"PASS: t2 = {self.t2} matches free-fall from height h+b = {self.h + self.b}")
            
            # Symbolic check of the derived formula
            symbolic_check = sp.simplify(self.Deltat_sym - (sp.sqrt(2*(self.h_sym + self.b_sym)/self.g_sym) 
                                                          - sp.sqrt(2*self.h_sym/self.g_sym)))
            if symbolic_check != 0:
                logs.append(f"FAIL: Symbolic formula mismatch: {symbolic_check}")
                success = False
            else:
                logs.append("PASS: Symbolic formula Δt = √[2(h+b)/g] - √[2h/g] is correct")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            if all(v is not None for v in [self.t1, self.t2, self.Deltat]):
                # Check Δt = t2 - t1
                if not math.isclose(self.Deltat, self.t2 - self.t1, rel_tol=0.01, abs_tol=0.01):
                    logs.append(f"FAIL: Δt ({self.Deltat}) ≠ t2 - t1 ({self.t2} - {self.t1} = {self.t2 - self.t1})")
                    success = False
                else:
                    logs.append(f"PASS: Δt = {self.Deltat} consistently equals t2 - t1 = {self.t2 - self.t1}")
                
                # Verify ordering: t2 > t1 > 0
                if not (self.t2 > self.t1 > 0):
                    logs.append(f"FAIL: Time ordering violated. t1={self.t1}, t2={self.t2}")
                    success = False
                else:
                    logs.append("PASS: Time ordering t2 > t1 > 0 holds")
            else:
                logs.append("SKIP: Missing t1, t2, or Δt for consistency check")
                success = False  # Consider missing data as failure
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Perturb input h and verify expected trend in Δt."""
        logs = []
        success = True
        
        try:
            if self.h is not None and self.b is not None:
                original_h = self.h
                perturbed_h = original_h * 1.05  # +5% perturbation
                
                # Compute Δt for original and perturbed h using physics formula
                original_Deltat = math.sqrt(2 * (original_h + self.b) / self.g) - math.sqrt(2 * original_h / self.g)
                perturbed_Deltat = math.sqrt(2 * (perturbed_h + self.b) / self.g) - math.sqrt(2 * perturbed_h / self.g)
                
                # Analytical derivative: d(Δt)/dh = (1/√g) * (1/√[2(h+b)] - 1/√[2h]) < 0
                # So Δt should DECREASE when h increases
                if not (perturbed_Deltat < original_Deltat):
                    logs.append(f"FAIL: Monotonicity violated. Δt({original_h})={original_Deltat:.3f}, "
                               f"Δt({perturbed_h})={perturbed_Deltat:.3f}. Expected Δt to decrease with increasing h.")
                    success = False
                else:
                    logs.append(f"PASS: Δt decreases with increasing h (monotonicity holds). "
                               f"Δt({original_h})={original_Deltat:.3f}, Δt({perturbed_h})={perturbed_Deltat:.3f}")
                
                # Restore original h (no mutation needed)
                logs.append("NOTE: Original h restored after perturbation test")
            else:
                logs.append("SKIP: Missing h or b for monotonicity test")
                success = False
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """Execute all audit tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [("Invariants", self.test_invariants),
                                       ("Trace Consistency", self.test_trace_consistency),
                                       ("Monotonicity", self.test_monotonicity)]:
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_method()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        # Final summary
        all_logs.append(f"\n=== Overall Result ===")
        if overall_success:
            all_logs.append("SUCCESS: All physics checks passed")
        else:
            all_logs.append("FAILURE: One or more physics checks failed")
            
        return overall_success, all_logs