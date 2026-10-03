import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary quantities from the solution instance.
        Do NOT directly access attributes; use getattr with fallback to results dict.
        """
        self.solution = solution_instance
        
        # Extract t_1 (time rowing upstream before realizing) and v_1 (river flow speed)
        # Using safe access: first try attribute, then results dict, then default 0.0
        self.t_1 = getattr(solution_instance, 't_1', 
                          getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', 
                          getattr(solution_instance, 'results', {}).get('v_1', 0.0))
        
        # Constants from the problem
        self.t_2 = 3600.0        # time rowing downstream (1 hour in seconds)
        self.distance = 5400.0   # catch-up distance from bridge in meters
        
        # For monotonicity test, store original v_1
        self.original_v1 = self.v_1
        
    def test_invariants(self):
        """
        Check physical invariants:
        1. The wooden block's motion: v_1 * (t_1 + t_2) should equal 5400 m.
        2. Consistency of boat's motion: t_1 should equal t_2 (from Newton's laws and constant velocity).
        """
        logs = []
        try:
            # Invariant 1: Wooden block displacement
            block_displacement = self.v_1 * (self.t_1 + self.t_2)
            if math.isclose(block_displacement, self.distance, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Wooden block displacement invariant holds.")
            else:
                logs.append(f"FAIL: Wooden block displacement mismatch. "
                           f"Expected {self.distance}, got {block_displacement}.")
            
            # Invariant 2: t_1 should equal t_2 (from boat's motion equations)
            if math.isclose(self.t_1, self.t_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Time symmetry (t_1 = t_2) holds.")
            else:
                logs.append(f"FAIL: Time symmetry broken. t_1 = {self.t_1}, t_2 = {self.t_2}.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        Here, we verify that the computed v_1 matches the expected 0.75 m/s,
        and that t_1 is consistent with the given t_2.
        """
        logs = []
        try:
            # Check final answer
            expected_v1 = 0.75
            if math.isclose(self.v_1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: River flow speed v_1 matches expected value (0.75 m/s).")
            else:
                logs.append(f"FAIL: River flow speed v_1 mismatch. "
                           f"Expected {expected_v1}, got {self.v_1}.")
            
            # Check that t_1 is 3600 s (as derived from the equations)
            expected_t1 = 3600.0
            if math.isclose(self.t_1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Upstream rowing time t_1 matches expected 3600 s.")
            else:
                logs.append(f"FAIL: Upstream rowing time t_1 mismatch. "
                           f"Expected {expected_t1}, got {self.t_1}.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            
        return logs
    
    def test_monotonicity(self):
        """
        Perturb the input distance (5400 m) and verify that v_1 changes proportionally.
        This checks the linear relationship v_1 = distance / (2 * t_2) when t_1 = t_2.
        We do NOT modify the solution instance; only compute with formulas.
        """
        logs = []
        try:
            # Perturb distance by +1%
            perturb_factor = 1.01
            D_original = self.distance
            D_perturbed = D_original * perturb_factor
            
            # The derived relationship: v_1 = D / (t_1 + t_2) and t_1 = t_2
            # So v_1 should be proportional to D.
            t_2 = self.t_2
            # Assuming t_1 = t_2 (as required by physics)
            if math.isclose(self.t_1, t_2, rel_tol=1e-2, abs_tol=1e-2):
                expected_v1_perturbed = D_perturbed / (2 * t_2)
                # Compute the ratio
                if self.original_v1 != 0:
                    ratio_original = D_original / self.original_v1
                    ratio_perturbed = D_perturbed / expected_v1_perturbed
                    # Both ratios should equal 2*t_2
                    if math.isclose(ratio_original, ratio_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append("PASS: Monotonicity test shows correct linear relationship.")
                    else:
                        logs.append(f"FAIL: Monotonicity test failed. "
                                   f"Ratios: original={ratio_original}, perturbed={ratio_perturbed}.")
                else:
                    logs.append("SKIP: Original v_1 is zero, cannot compute ratio.")
            else:
                logs.append("SKIP: t_1 ≠ t_2, monotonicity test not applicable.")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return a tuple (success_bool, logs_list).
        Success is defined as all tests passing without any FAIL messages.
        """
        all_logs = []
        
        # Run each test
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL"):
                success = False
                break
        
        return success, all_logs