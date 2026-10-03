import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 15.0))
        self.M = getattr(solution_instance, 'M', getattr(solution_instance, 'inputs', {}).get('M', 55.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'inputs', {}).get('a', 0.2))
        self.g = getattr(solution_instance, 'g', 10.0)  # Based on calculation match
        
        # Extract results safely from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.F = results.get('F', 0.0)
        self.F_N = results.get('F_N', 0.0)
        self.N_A = results.get('N/A', 0.0)  # Reaction force magnitude
        
        # Target answer parsing
        self.target_force = 204.0
        
        # Log storage
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((success, msg))
        return success

    def test_invariants(self):
        try:
            # Symbolic Newton's 2nd Law for system
            m, M, a, g = sp.symbols('m M a g', positive=True)
            F_sym = sp.Symbol('F')
            F_N_sym = sp.Symbol('F_N')
            
            # System equation: 2F - (M+m)g = (M+m)a
            system_eq = sp.Eq(2*F_sym, (M + m)*(a + g))
            
            # Person equation: F + F_N - Mg = Ma
            person_eq = sp.Eq(F_sym + F_N_sym, M*(a + g))
            
            # Platform equation: F - F_N - mg = ma
            platform_eq = sp.Eq(F_sym - F_N_sym, m*(a + g))
            
            # Check system equation numerically
            expected_F = (self.M + self.m) * (self.a + self.g) / 2
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"System Newton's 2nd Law fails: F={self.F}, expected={expected_F}", False)
            
            # Check person equation numerically
            expected_F_N = self.M * (self.a + self.g) - self.F
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Person Newton's 2nd Law fails: F_N={self.F_N}, expected={expected_F_N}", False)
            
            # Check platform equation numerically
            lhs_platform = self.F - self.F_N
            rhs_platform = self.m * (self.a + self.g)
            if not math.isclose(lhs_platform, rhs_platform, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Platform Newton's 2nd Law fails: F-F_N={lhs_platform}, m(a+g)={rhs_platform}", False)
            
            # Newton's 3rd Law: Force on platform by person equals -F_N
            if not math.isclose(self.F_N, self.N_A, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Newton's 3rd Law fails: F_N={self.F_N}, reaction={self.N_A}", False)
            
            return self._log("All Newton's law invariants satisfied", True)
        except Exception as e:
            return self._log(f"Invariant test error: {str(e)}", False)

    def test_trace_consistency(self):
        try:
            # Check step 1: F = (M+m)(a+g)/2
            expected_F = (self.M + self.m) * (self.a + self.g) / 2
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Step 1 inconsistency: F={self.F}, formula gives {expected_F}", False)
            
            # Check step 2: F_N = M(a+g) - F
            expected_F_N = self.M * (self.a + self.g) - self.F
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Step 2 inconsistency: F_N={self.F_N}, formula gives {expected_F_N}", False)
            
            # Check step 3: Reaction equals F_N
            if not math.isclose(self.F_N, self.N_A, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Step 3 inconsistency: F_N={self.F_N}, reaction={self.N_A}", False)
            
            # Check final answer matches target
            if not math.isclose(self.N_A, self.target_force, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Final answer mismatch: {self.N_A} vs target {self.target_force}", False)
            
            return self._log("Trace consistency verified", True)
        except Exception as e:
            return self._log(f"Trace consistency error: {str(e)}", False)

    def test_monotonicity(self):
        original_M = self.M
        try:
            # Perturb M by +10%
            perturbed_M = original_M * 1.1
            
            # Recalculate using formulas
            F_perturbed = (perturbed_M + self.m) * (self.a + self.g) / 2
            F_N_perturbed = perturbed_M * (self.a + self.g) - F_perturbed
            
            # Check monotonic trend: Larger M should yield larger F_N
            if not (F_N_perturbed > self.F_N):
                return self._log(f"Monotonicity fails: F_N={self.F_N} -> {F_N_perturbed} for M={original_M}->{perturbed_M}", False)
            
            # Restore original value
            self.M = original_M
            return self._log(f"Monotonicity OK: F_N increases with M ({self.F_N:.1f} -> {F_N_perturbed:.1f})", True)
        except Exception as e:
            self.M = original_M  # Restore on error
            return self._log(f"Monotonicity test error: {str(e)}", False)

    def run_all_tests(self):
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.logs.append((False, f"Test {test.__name__} crashed: {traceback.format_exc()}"))
                success = False
        
        return success, self.logs