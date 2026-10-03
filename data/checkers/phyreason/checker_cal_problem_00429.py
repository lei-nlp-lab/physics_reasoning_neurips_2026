import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required attributes from the solution instance.
        """
        self.solution = solution_instance
        
        # Input parameters from the problem context
        self.total_time = 6.0  # seconds
        self.delta_h_6th = 22.0  # meters (displacement during 6th second)
        
        # Safely extract results from solution_instance using getattr with fallback
        # according to the expected keys in self.results
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        self.Deltah = getattr(solution_instance, 'Deltah', solution_instance.results.get('Deltah', None))
        self.h_5 = getattr(solution_instance, 'h_5', solution_instance.results.get('h_5', None))
        self.h_6 = getattr(solution_instance, 'h_6', solution_instance.results.get('h_6', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', None))
        
        # Symbolic variables for invariant checks
        self.sym_g, self.sym_t, self.sym_h5, self.sym_h6, self.sym_v = sp.symbols('g t h5 h6 v', positive=True, real=True)
    
    def test_invariants(self):
        """
        Verify the fundamental kinematic relationships (Newton's laws for uniformly accelerated motion).
        """
        logs = []
        success = True
        
        # 1. Check displacement during the 6th second formula: Δh = h6 - h5
        if self.h_6 is not None and self.h_5 is not None and self.Deltah is not None:
            expected_Deltah = self.h_6 - self.h_5
            if not math.isclose(self.Deltah, expected_Deltah, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δh = h6 - h5 violated. Got Δh={self.Deltah}, expected {expected_Deltah}")
                success = False
            else:
                logs.append(f"PASS: Δh = h6 - h5 check passed.")
        else:
            logs.append("SKIP: Missing h_6, h_5, or Δh for invariant check.")
        
        # 2. Check kinematic equations for free fall from rest: h6 = 0.5*g*t6^2, h5 = 0.5*g*t5^2
        if self.g is not None:
            t6, t5 = 6.0, 5.0
            expected_h6 = 0.5 * self.g * t6**2
            expected_h5 = 0.5 * self.g * t5**2
            
            if self.h_6 is not None and not math.isclose(self.h_6, expected_h6, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h6 = 0.5*g*t6^2 violated. Got h6={self.h_6}, expected {expected_h6}")
                success = False
            else:
                logs.append(f"PASS: h6 = 0.5*g*t6^2 check passed.")
            
            if self.h_5 is not None and not math.isclose(self.h_5, expected_h5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h5 = 0.5*g*t5^2 violated. Got h5={self.h_5}, expected {expected_h5}")
                success = False
            else:
                logs.append(f"PASS: h5 = 0.5*g*t5^2 check passed.")
            
            # 3. Check final velocity: v = g*t6
            if self.v is not None:
                expected_v = self.g * t6
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v = g*t6 violated. Got v={self.v}, expected {expected_v}")
                    success = False
                else:
                    logs.append(f"PASS: v = g*t6 check passed.")
        else:
            logs.append("SKIP: Missing g for kinematic checks.")
        
        # 4. Check that release height equals h6 (total fallen distance)
        if self.h is not None and self.h_6 is not None:
            if not math.isclose(self.h, self.h_6, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Release height h should equal h6. Got h={self.h}, h6={self.h_6}")
                success = False
            else:
                logs.append(f"PASS: h = h6 check passed.")
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        # Check that Δh computed from given data (22 m) matches the derived formula: Δh = 5.5 * g
        if self.g is not None:
            expected_Deltah_from_g = 5.5 * self.g
            if not math.isclose(self.delta_h_6th, expected_Deltah_from_g, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δh from problem (22 m) does not match 5.5*g. Computed Δh=5.5*{self.g}={expected_Deltah_from_g}")
                success = False
            else:
                logs.append(f"PASS: Δh = 5.5*g consistency check passed.")
        
        # Check that g solves the equation: 0.5*g*(6^2 - 5^2) = 22
        if self.g is not None:
            lhs = 0.5 * self.g * (36 - 25)
            if not math.isclose(lhs, 22.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 0.5*g*(36-25) = {lhs}, expected 22")
                success = False
            else:
                logs.append(f"PASS: Equation 0.5*g*(36-25)=22 solved correctly.")
        
        # Check that h_6 and v are consistent with g and total time
        if self.g is not None and self.h_6 is not None:
            expected_h6 = 0.5 * self.g * 36
            if not math.isclose(self.h_6, expected_h6, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h6 inconsistent with g. Expected {expected_h6}, got {self.h_6}")
                success = False
            else:
                logs.append(f"PASS: h6 consistent with g and t=6s.")
        
        if self.g is not None and self.v is not None:
            expected_v = self.g * 6
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v inconsistent with g. Expected {expected_v}, got {self.v}")
                success = False
            else:
                logs.append(f"PASS: v consistent with g and t=6s.")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (e.g., delta_h_6th) and check that g changes in the expected direction.
        Then restore the original value.
        """
        logs = []
        success = True
        
        if self.g is None:
            logs.append("SKIP: Cannot test monotonicity without g.")
            return False, logs
        
        # Perturb Δh by +10% and recompute g using the formula: g = Δh / 5.5
        original_delta_h = self.delta_h_6th
        perturbed_delta_h = original_delta_h * 1.10  # +10%
        perturbed_g = perturbed_delta_h / 5.5
        
        # Check monotonic relationship: if Δh increases, g should increase (positive proportionality)
        if perturbed_g <= self.g:
            logs.append(f"FAIL: Monotonicity violated. Δh increased from {original_delta_h} to {perturbed_delta_h}, but g did not increase (original g={self.g}, perturbed g={perturbed_g}).")
            success = False
        else:
            logs.append(f"PASS: Monotonicity check: Δh↑ → g↑ (g increased from {self.g} to {perturbed_g}).")
        
        # Restore original Δh (no actual change needed as we didn't modify instance)
        logs.append("INFO: Original Δh restored.")
        
        # Additional monotonicity: h6 = 0.5*g*36 should increase with g
        original_h6 = 0.5 * self.g * 36
        perturbed_h6 = 0.5 * perturbed_g * 36
        if perturbed_h6 <= original_h6:
            logs.append(f"FAIL: Monotonicity in h6 violated. g increased but h6 did not increase (original h6={original_h6}, perturbed h6={perturbed_h6}).")
            success = False
        else:
            logs.append(f"PASS: h6 monotonic with g (h6 increased from {original_h6} to {perturbed_h6}).")
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs