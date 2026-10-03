import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Inputs
        self.h = getattr(solution_instance, 'h', 0.0)
        self.R = getattr(solution_instance, 'R', 0.0)
        self.g = getattr(solution_instance, 'g', 0.0)
        self.G = getattr(solution_instance, 'G', 0.0)
        
        # Results (with safe fallback to results dictionary)
        results = getattr(solution_instance, 'results', {})
        self.M = results.get('M', getattr(solution_instance, 'M', 0.0))
        self.a = results.get('a', getattr(solution_instance, 'a', 0.0))
        self.v = results.get('v', getattr(solution_instance, 'v', 0.0))
        
        # Note: Removed SymPy symbols since they're not used in the tests
    
    def test_invariants(self):
        """
        Verify physical invariants: gravitational law, centripetal force, etc.
        """
        logs = []
        try:
            # Check for zero division - use tolerance instead of exact zero
            if abs(self.G) < 1e-30:
                logs.append("WARNING: G is very small, may cause division issues")
                # Don't return - continue with tests
            if abs(self.R + self.h) < 1e-30:
                logs.append("WARNING: R+h is very small, may cause division issues")
                # Don't return - continue with tests
            
            # 1. Check M = g * R^2 / G (skip if G is too small)
            if abs(self.G) > 1e-30:
                M_expected = self.g * self.R**2 / self.G
                if not math.isclose(self.M, M_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: M mismatch. Got {self.M}, expected {M_expected}")
                else:
                    logs.append("PASS: M = g R² / G")
            else:
                logs.append("SKIP: M check (G too small)")
            
            # 2. Check a = g * R^2 / (R + h)^2
            if abs(self.R + self.h) > 1e-30:
                a_expected = self.g * self.R**2 / (self.R + self.h)**2
                if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a mismatch. Got {self.a}, expected {a_expected}")
                else:
                    logs.append("PASS: a = g R² / (R+h)²")
            else:
                logs.append("SKIP: a check (R+h too small)")
            
            # 3. Check v = R * sqrt(g / (R + h))
            if abs(self.R + self.h) > 1e-30 and self.g >= 0:
                v_expected = self.R * math.sqrt(abs(self.g) / (self.R + self.h))
                if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v mismatch. Got {self.v}, expected {v_expected}")
                else:
                    logs.append("PASS: v = R √[g/(R+h)]")
            else:
                logs.append("SKIP: v check (R+h too small or g negative)")
            
            # 4. Check consistency of centripetal acceleration: a = v² / (R + h)
            if abs(self.R + self.h) > 1e-30:
                a_from_v = self.v**2 / (self.R + self.h)
                if not math.isclose(self.a, a_from_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a ≠ v²/(R+h). a={self.a}, v²/(R+h)={a_from_v}")
                else:
                    logs.append("PASS: a = v²/(R+h) (centripetal condition)")
            else:
                logs.append("SKIP: a = v²/(R+h) check (R+h too small)")
            
            # 5. Check gravitational force equals centripetal force (with Earth mass M)
            # F_grav = G * M / (R+h)^2 should equal a (for unit satellite mass)
            if abs(self.G) > 1e-30 and abs(self.R + self.h) > 1e-30:
                a_from_M = self.G * self.M / (self.R + self.h)**2
                if not math.isclose(self.a, a_from_M, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a ≠ G*M/(R+h)². a={self.a}, computed {a_from_M}")
                else:
                    logs.append("PASS: a = G*M/(R+h)² (Newton's law)")
            else:
                logs.append("SKIP: a = G*M/(R+h)² check (G or R+h too small)")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check that intermediate results (M, a, v) are mutually consistent.
        """
        logs = []
        try:
            # Check for zero division - use tolerance
            if abs(self.G) < 1e-30:
                logs.append("WARNING: G is very small, may cause division issues")
            if abs(self.R + self.h) < 1e-30:
                logs.append("WARNING: R+h is very small, may cause division issues")
            
            # 1. From M, compute a and compare
            if abs(self.G) > 1e-30 and abs(self.R + self.h) > 1e-30:
                a_from_M = self.G * self.M / (self.R + self.h)**2
                if not math.isclose(self.a, a_from_M, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a inconsistent with M. a={self.a}, from M: {a_from_M}")
                else:
                    logs.append("PASS: a consistent with M")
            else:
                logs.append("SKIP: a from M check (G or R+h too small)")
            
            # 2. From a, compute v and compare
            if abs(self.R + self.h) > 1e-30 and self.a >= 0:
                v_from_a = math.sqrt(abs(self.a) * (self.R + self.h))
                if not math.isclose(self.v, v_from_a, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v inconsistent with a. v={self.v}, from a: {v_from_a}")
                else:
                    logs.append("PASS: v consistent with a")
            else:
                logs.append("SKIP: v from a check (R+h too small or a negative)")
            
            # 3. From v, compute M and compare (via a)
            if abs(self.R + self.h) > 1e-30 and abs(self.G) > 1e-30:
                a_from_v = self.v**2 / (self.R + self.h)
                M_from_v = a_from_v * (self.R + self.h)**2 / self.G
                if not math.isclose(self.M, M_from_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: M inconsistent with v. M={self.M}, from v: {M_from_v}")
                else:
                    logs.append("PASS: M consistent with v")
            else:
                logs.append("SKIP: M from v check (R+h or G too small)")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """
        Perturb input h and verify expected trends in a and v.
        """
        logs = []
        original_h = self.h
        try:
            # Check for zero division
            if abs(self.G) < 1e-30:
                logs.append("WARNING: G is very small")
            if abs(self.R + original_h) < 1e-30:
                logs.append("WARNING: R+h is very small")
            
            # Increase h by 1% (but ensure we don't multiply by zero)
            if abs(original_h) > 1e-30:
                perturbed_h = original_h * 1.01
            else:
                perturbed_h = 1.0  # Default if h is zero
            
            # Recompute a and v using the derived formulas (not from solution instance)
            if abs(self.R + original_h) > 1e-30:
                a_original = self.g * self.R**2 / (self.R + original_h)**2
                v_original = self.R * math.sqrt(abs(self.g) / (self.R + original_h))
            else:
                a_original = 0.0
                v_original = 0.0
                logs.append("SKIP: original a,v computation (R+h too small)")
            
            if abs(self.R + perturbed_h) > 1e-30:
                a_perturbed = self.g * self.R**2 / (self.R + perturbed_h)**2
                v_perturbed = self.R * math.sqrt(abs(self.g) / (self.R + perturbed_h))
            else:
                a_perturbed = 0.0
                v_perturbed = 0.0
                logs.append("SKIP: perturbed a,v computation (R+h too small)")
            
            # Check a decreases as h increases (since a ∝ 1/(R+h)²)
            if a_original > 0 and a_perturbed > 0:
                if a_perturbed >= a_original:
                    logs.append(f"FAIL: a does not decrease when h increases. a(original)={a_original}, a(perturbed)={a_perturbed}")
                else:
                    logs.append("PASS: a decreases with increasing h")
            else:
                logs.append("SKIP: a monotonicity check (a values non-positive)")
            
            # Check v decreases as h increases (since v ∝ 1/√(R+h))
            if v_original > 0 and v_perturbed > 0:
                if v_perturbed >= v_original:
                    logs.append(f"FAIL: v does not decrease when h increases. v(original)={v_original}, v(perturbed)={v_perturbed}")
                else:
                    logs.append("PASS: v decreases with increasing h")
            else:
                logs.append("SKIP: v monotonicity check (v values non-positive)")
            
            # M should be independent of h; check with perturbed h (using formula)
            if abs(self.G) > 1e-30:
                M_original = self.M
                M_perturbed = self.g * self.R**2 / self.G  # same as original
                if not math.isclose(M_original, M_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: M changed when h perturbed. M(original)={M_original}, M(perturbed)={M_perturbed}")
                else:
                    logs.append("PASS: M independent of h")
            else:
                logs.append("SKIP: M independence check (G too small)")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Ensure no side effects: restore original h (though we only used local copies)
            pass
        return logs
    
    def run_all_tests(self):
        """
        Run all tests and return success flag and logs.
        """
        all_logs = []
        all_logs.append("=== PhysicsChecker started ===")
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            logs = test_func()
            all_logs.extend(logs)
            # Check for any failure or error messages
            for log in logs:
                if log.startswith(("FAIL", "ERROR")):
                    success = False
        
        all_logs.append(f"\n=== Overall: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, all_logs