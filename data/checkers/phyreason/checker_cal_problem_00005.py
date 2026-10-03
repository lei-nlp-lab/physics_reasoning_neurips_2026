import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results without direct attribute access."""
        self.solution = solution_instance
        
        # Safe extraction of inputs (numeric only)
        self.h = getattr(solution_instance, 'h', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.theta = getattr(solution_instance, 'theta', 10 * math.pi / 180)
        
        # Safe extraction of results (must be numeric floats)
        self.results = getattr(solution_instance, 'results', {})
        self.v_y = self.results.get('v_y', None)
        self.v_0 = self.results.get('v_0', None)
        self.v_min = self.results.get('v_min', None)
    
    def test_invariants(self):
        """Verify kinematic invariants and skipping condition using numeric comparisons only."""
        logs = []
        success = True
        
        try:
            # 1. Check vertical velocity from free fall: v_y = √(2gh)
            if self.v_y is not None:
                expected_v_y = math.sqrt(2 * self.g * self.h)
                if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_y mismatch. Expected √(2gh)={expected_v_y:.3f}, got {self.v_y:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: v_y = √(2gh) = {self.v_y:.3f} m/s")
            else:
                logs.append("SKIP: v_y not provided")
            
            # 2. Check condition for skipping: v_y/v_0 ≤ tan(θ) (equality for minimum)
            if self.v_0 is not None and self.v_y is not None:
                ratio = self.v_y / self.v_0
                tan_theta = math.tan(self.theta)
                # For minimum velocity, we expect equality
                if not math.isclose(ratio, tan_theta, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Skipping condition not at equality. v_y/v_0 = {ratio:.3f}, tan(θ) = {tan_theta:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: v_y/v_0 = tan(θ) = {tan_theta:.3f}")
            else:
                logs.append("SKIP: v_0 or v_y not provided")
            
            # 3. Check v_min formula: v_min = √(2gh)/tan(θ)
            if self.v_min is not None:
                expected_v_min = math.sqrt(2 * self.g * self.h) / math.tan(self.theta)
                if not math.isclose(self.v_min, expected_v_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_min mismatch. Expected √(2gh)/tan(θ)={expected_v_min:.3f}, got {self.v_min:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: v_min = √(2gh)/tan(θ) = {self.v_min:.3f} m/s")
            else:
                logs.append("SKIP: v_min not provided")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer using numeric comparisons."""
        logs = []
        success = True
        
        try:
            # Check that v_0 equals v_min (since thrown horizontally)
            if self.v_0 is not None and self.v_min is not None:
                if not math.isclose(self.v_0, self.v_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_0={self.v_0:.3f} != v_min={self.v_min:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: v_0 = v_min = {self.v_0:.3f} m/s")
            else:
                logs.append("SKIP: v_0 or v_min not provided")
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Check that v_min increases with h and decreases with θ using numeric perturbations."""
        logs = []
        success = True
        
        try:
            # Original v_min
            v_min_original = math.sqrt(2 * self.g * self.h) / math.tan(self.theta)
            
            # Perturb h upward by 10%
            h_perturbed = self.h * 1.1
            v_min_h_up = math.sqrt(2 * self.g * h_perturbed) / math.tan(self.theta)
            
            if not v_min_h_up > v_min_original:
                logs.append(f"FAIL: Monotonicity in h violated. h={self.h:.3f}->{h_perturbed:.3f}, v_min={v_min_original:.3f}->{v_min_h_up:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_min increases with h ({v_min_original:.3f} -> {v_min_h_up:.3f})")
            
            # Perturb θ upward by 10% (but keep < π/2)
            theta_perturbed = min(self.theta * 1.1, 1.5)  # Cap at ~85°
            v_min_theta_up = math.sqrt(2 * self.g * self.h) / math.tan(theta_perturbed)
            
            if not v_min_theta_up < v_min_original:
                logs.append(f"FAIL: Monotonicity in θ violated. θ={self.theta:.3f}->{theta_perturbed:.3f}, v_min={v_min_original:.3f}->{v_min_theta_up:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_min decreases with θ ({v_min_original:.3f} -> {v_min_theta_up:.3f})")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs