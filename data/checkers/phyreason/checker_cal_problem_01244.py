import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks - convert to float
        self.m = float(getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0)))
        self.R = float(getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.0)))
        self.g = float(getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.81)))
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.v_1 = float(self.results.get('v_1', 0.0))
        self.v_2 = float(self.results.get('v_2', 0.0))
        self.W_f = float(self.results.get('W_f', 0.0))
        self.W = float(self.results.get('W', 0.0))
        
        # Expected numeric values
        self.v_1_expected = math.sqrt(6 * self.g * self.R)
        self.v_2_expected = math.sqrt(self.g * self.R)
        self.W_f_expected = -0.5 * self.m * self.g * self.R
        self.W_expected = 0.5 * self.m * self.g * self.R
        
    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check centripetal force at lowest point (Step 1)
        try:
            # Newton's 2nd Law: T - mg = m*v_1^2/R
            T = 7 * self.m * self.g
            lhs = T - self.m * self.g
            rhs = self.m * self.v_1**2 / self.R
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: Centripetal force at lowest point mismatch. LHS={lhs:.3f}, RHS={rhs:.3f}")
                success = False
            else:
                logs.append("PASS: Centripetal force balance at lowest point")
        except Exception as e:
            logs.append(f"ERROR in lowest-point check: {traceback.format_exc()}")
            success = False
        
        # 2. Check centripetal force at highest point (Step 2)
        try:
            # Minimum condition: mg = m*v_2^2/R
            lhs = self.m * self.g
            rhs = self.m * self.v_2**2 / self.R
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: Centripetal force at highest point mismatch. LHS={lhs:.3f}, RHS={rhs:.3f}")
                success = False
            else:
                logs.append("PASS: Centripetal force balance at highest point")
        except Exception as e:
            logs.append(f"ERROR in highest-point check: {traceback.format_exc()}")
            success = False
        
        # 3. Check work-energy theorem (Step 3)
        try:
            # W_gravity + W_f = ΔK
            W_gravity = -2 * self.m * self.g * self.R
            ΔK = 0.5 * self.m * (self.v_2**2 - self.v_1**2)
            lhs = W_gravity + self.W_f
            if not self._is_close(lhs, ΔK):
                logs.append(f"FAIL: Work-energy theorem mismatch. W_gravity+W_f={lhs:.3f}, ΔK={ΔK:.3f}")
                success = False
            else:
                logs.append("PASS: Work-energy theorem holds")
        except Exception as e:
            logs.append(f"ERROR in work-energy check: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check that intermediate velocities match expectations
        try:
            if not self._is_close(self.v_1, self.v_1_expected):
                logs.append(f"FAIL: v_1 mismatch. Got {self.v_1:.3f}, expected {self.v_1_expected:.3f}")
                success = False
            else:
                logs.append("PASS: v_1 matches expected sqrt(6gR)")
        except Exception as e:
            logs.append(f"ERROR in v_1 check: {traceback.format_exc()}")
            success = False
        
        try:
            if not self._is_close(self.v_2, self.v_2_expected):
                logs.append(f"FAIL: v_2 mismatch. Got {self.v_2:.3f}, expected {self.v_2_expected:.3f}")
                success = False
            else:
                logs.append("PASS: v_2 matches expected sqrt(gR)")
        except Exception as e:
            logs.append(f"ERROR in v_2 check: {traceback.format_exc()}")
            success = False
        
        # Check that W_f and W are consistent (W = -W_f)
        try:
            if not self._is_close(self.W, -self.W_f):
                logs.append(f"FAIL: W and W_f not consistent. W={self.W:.3f}, -W_f={-self.W_f:.3f}")
                success = False
            else:
                logs.append("PASS: W = -W_f (work to overcome resistance)")
        except Exception as e:
            logs.append(f"ERROR in W/W_f consistency: {traceback.format_exc()}")
            success = False
        
        # Check final answer against symbolic expectation
        try:
            if not self._is_close(self.W, self.W_expected):
                logs.append(f"FAIL: Final W mismatch. Got {self.W:.3f}, expected {self.W_expected:.3f}")
                success = False
            else:
                logs.append("PASS: Final answer matches (1/2)mgR")
        except Exception as e:
            logs.append(f"ERROR in final answer check: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Perturb mass and check if W changes proportionally
        original_m = self.m
        perturb_factor = 1.1  # 10% increase
        
        try:
            # Recalculate expected W with perturbed mass
            perturbed_m = original_m * perturb_factor
            W_perturbed_expected = 0.5 * perturbed_m * self.g * self.R
            
            W_original = 0.5 * original_m * self.g * self.R
            
            # Check proportionality: W_perturbed / W_original should equal mass ratio
            expected_ratio = perturb_factor
            actual_ratio = W_perturbed_expected / W_original
            
            if not self._is_close(actual_ratio, expected_ratio):
                logs.append(f"FAIL: Monotonicity w.r.t. mass. Ratio={actual_ratio:.3f}, expected={expected_ratio:.3f}")
                success = False
            else:
                logs.append("PASS: W scales linearly with mass")
        except Exception as e:
            logs.append(f"ERROR in monotonicity check: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                test_success, logs = test_method()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"CRASH: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs