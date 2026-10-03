import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr and dict.get to avoid AttributeError.
        """
        self.solution = solution_instance
        # Extract inputs with fallback defaults
        self.v0 = getattr(solution_instance, 'v0', 
                         getattr(solution_instance, 'results', {}).get('v0', 0.0))
        self.g = getattr(solution_instance, 'g', 
                        getattr(solution_instance, 'results', {}).get('g', 9.8))
        self.m = getattr(solution_instance, 'm', 
                        getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', 
                        getattr(solution_instance, 'results', {}).get('q', 1.0))
        
        # Extract results using key mapping with fallbacks
        results = getattr(solution_instance, 'results', {})
        self.x = results.get('x', 0.0)
        self.v_y = results.get('v_y', 0.0)
        self.PM = results.get('PM', 0.0)
        self.t = results.get('t', 0.0)
        self.theta = results.get('theta', 0.0)
        self.v = results.get('v', 0.0)
        
        # For symbolic checks
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.g_sym = sp.Symbol('g', positive=True)
        
    def fuzzy_equal(self, a, b):
        """Fuzzy numerical comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        """
        Verify kinematic invariants based on KB theorems.
        Focus on vector decomposition and constant acceleration kinematics.
        """
        logs = []
        success = True
        
        try:
            # 1. Check that tan(theta) = v0/v_y should give 45 degrees
            if self.v_y != 0:
                tan_calc = self.v0 / self.v_y
                tan_expected = math.tan(math.radians(45))  # tan(45°) = 1
                if not self.fuzzy_equal(tan_calc, tan_expected):
                    logs.append(f"FAIL: tan(theta) = v0/v_y = {tan_calc:.4f}, expected ~{tan_expected:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: tan(theta) = v0/v_y ≈ 1 (45°)")
            else:
                logs.append("WARN: v_y = 0, cannot check angle condition")
            
            # 2. Check v = sqrt(v0^2 + v_y^2) from vector addition theorem
            v_calc = math.sqrt(self.v0**2 + self.v_y**2)
            if not self.fuzzy_equal(self.v, v_calc):
                logs.append(f"FAIL: v = {self.v:.4f}, expected sqrt(v0^2 + v_y^2) = {v_calc:.4f}")
                success = False
            else:
                logs.append(f"PASS: v = sqrt(v0^2 + v_y^2) ≈ {self.v:.4f}")
            
            # 3. Check t = v_y / g from constant acceleration kinematics
            if self.g != 0:
                t_calc = self.v_y / self.g
                if not self.fuzzy_equal(self.t, t_calc):
                    logs.append(f"FAIL: t = {self.t:.4f}, expected v_y/g = {t_calc:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: t = v_y/g ≈ {self.t:.4f}")
            
            # 4. Check x = v0 * t from constant velocity kinematics
            x_calc = self.v0 * self.t
            if not self.fuzzy_equal(self.x, x_calc):
                logs.append(f"FAIL: x = {self.x:.4f}, expected v0*t = {x_calc:.4f}")
                success = False
            else:
                logs.append(f"PASS: x = v0*t ≈ {self.x:.4f}")
            
            # 5. Check PM = sqrt(2) * x from geometric relationship
            PM_calc = math.sqrt(2) * self.x
            if not self.fuzzy_equal(self.PM, PM_calc):
                logs.append(f"FAIL: PM = {self.PM:.4f}, expected sqrt(2)*x = {PM_calc:.4f}")
                success = False
            else:
                logs.append(f"PASS: PM = sqrt(2)*x ≈ {self.PM:.4f}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify the entire calculation chain from steps analysis.
        """
        logs = []
        success = True
        
        try:
            # Step-by-step verification using symbolic expressions as fallback
            
            # Step 2: v_y should equal v0 (since tan45=1)
            if not self.fuzzy_equal(self.v_y, self.v0):
                logs.append(f"INCONSISTENT: v_y = {self.v_y:.4f} ≠ v0 = {self.v0:.4f}")
                success = False
            else:
                logs.append(f"CONSISTENT: v_y = v0 = {self.v0:.4f}")
            
            # Step 3: v = sqrt(2)*v0
            v_expected = math.sqrt(2) * self.v0
            if not self.fuzzy_equal(self.v, v_expected):
                logs.append(f"INCONSISTENT: v = {self.v:.4f} ≠ sqrt(2)*v0 = {v_expected:.4f}")
                success = False
            else:
                logs.append(f"CONSISTENT: v = sqrt(2)*v0 ≈ {self.v:.4f}")
            
            # Step 5: t = v0/g (since v_y = v0)
            t_expected = self.v0 / self.g if self.g != 0 else 0
            if not self.fuzzy_equal(self.t, t_expected):
                logs.append(f"INCONSISTENT: t = {self.t:.4f} ≠ v0/g = {t_expected:.4f}")
                success = False
            else:
                logs.append(f"CONSISTENT: t = v0/g ≈ {self.t:.4f}")
            
            # Step 6: x = v0^2/g
            x_expected = self.v0**2 / self.g if self.g != 0 else 0
            if not self.fuzzy_equal(self.x, x_expected):
                logs.append(f"INCONSISTENT: x = {self.x:.4f} ≠ v0^2/g = {x_expected:.4f}")
                success = False
            else:
                logs.append(f"CONSISTENT: x = v0^2/g ≈ {self.x:.4f}")
            
            # Step 7: PM = sqrt(2)*v0^2/g
            PM_expected = math.sqrt(2) * self.v0**2 / self.g if self.g != 0 else 0
            if not self.fuzzy_equal(self.PM, PM_expected):
                logs.append(f"INCONSISTENT: PM = {self.PM:.4f} ≠ sqrt(2)*v0^2/g = {PM_expected:.4f}")
                success = False
            else:
                logs.append(f"CONSISTENT: PM = sqrt(2)*v0^2/g ≈ {self.PM:.4f}")
                
            # Check target answers
            target_v = f"${math.sqrt(2):.4f}v_{{0}}$"
            target_PM = f"${math.sqrt(2):.4f}v_{{0}}^{{2}}/g$"
            logs.append(f"Target answers: v ≈ {target_v}, PM ≈ {target_PM}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb initial velocity v0 and check output trends.
        Recalculate using formulas from steps analysis.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            v0_orig = self.v0
            
            # Test with 10% increase in v0
            v0_perturbed = v0_orig * 1.10
            
            # Recalculate expected values using formulas from steps
            v_y_new = v0_perturbed  # from step 2
            v_new = math.sqrt(2) * v0_perturbed  # from step 3
            t_new = v0_perturbed / self.g  # from step 5
            x_new = v0_perturbed**2 / self.g  # from step 6
            PM_new = math.sqrt(2) * v0_perturbed**2 / self.g  # from step 7
            
            # Check monotonic relationships
            # v should increase with v0
            if not (v_new > self.v):
                logs.append(f"MONOTONICITY VIOLATION: v did not increase when v0 increased")
                success = False
            else:
                logs.append(f"MONOTONIC: v increases with v0 (v: {self.v:.4f} → {v_new:.4f})")
            
            # PM should increase with v0 (quadratically)
            if not (PM_new > self.PM):
                logs.append(f"MONOTONICITY VIOLATION: PM did not increase when v0 increased")
                success = False
            else:
                logs.append(f"MONOTONIC: PM increases quadratically with v0 (PM: {self.PM:.4f} → {PM_new:.4f})")
            
            # Check proportionality
            # v should be proportional to v0
            ratio_v = v_new / self.v if self.v != 0 else 0
            ratio_v0 = v0_perturbed / v0_orig if v0_orig != 0 else 0
            if not self.fuzzy_equal(ratio_v, ratio_v0):
                logs.append(f"PROPORTIONALITY ISSUE: v/v0 ratio changed from {ratio_v0:.4f} to {ratio_v:.4f}")
                success = False
            else:
                logs.append(f"PROPORTIONAL: v ∝ v0 (ratio constant at ~{ratio_v:.4f})")
            
            # PM should be proportional to v0^2
            ratio_PM = PM_new / self.PM if self.PM != 0 else 0
            ratio_v0_sq = (v0_perturbed**2) / (v0_orig**2) if v0_orig != 0 else 0
            if not self.fuzzy_equal(ratio_PM, ratio_v0_sq):
                logs.append(f"PROPORTIONALITY ISSUE: PM/v0^2 ratio changed")
                success = False
            else:
                logs.append(f"PROPORTIONAL: PM ∝ v0^2 (ratio constant at ~{ratio_PM:.4f})")
            
            # Restore original v0
            self.v0 = v0_orig
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Restore original v0 if possible
            try:
                self.v0 = v0_orig
            except:
                pass
            
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        all_logs.append("=" * 60)
        all_logs.append("PhysicsChecker: Starting comprehensive audit")
        all_logs.append(f"Inputs: v0={self.v0:.4f}, g={self.g:.4f}, m={self.m:.4f}, q={self.q:.4f}")
        all_logs.append(f"Results: v={self.v:.4f}, v_y={self.v_y:.4f}, t={self.t:.4f}, x={self.x:.4f}, PM={self.PM:.4f}, theta={self.theta:.4f}")
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.append("\n--- Test 1: Kinematic Invariants ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Test 2: Trace Consistency ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Test 3: Monotonicity & Proportionality ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        all_logs.append("\n" + "=" * 60)
        all_logs.append(f"Overall audit result: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs