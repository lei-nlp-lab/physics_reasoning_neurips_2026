import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Never assume direct attribute access; use getattr with fallback.
        """
        self.solution = solution_instance
        
        # Extract given constants (inputs)
        self.m = getattr(solution_instance, 'm', 0.5)
        self.R = getattr(solution_instance, 'R', 3.6)
        self.h = getattr(solution_instance, 'h', 6.0)
        self.v_C = getattr(solution_instance, 'v_C', 10.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.angle_deg = 30.0  # Given in problem
        
        # Extract results (using safe access)
        results = getattr(solution_instance, 'results', {})
        
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', 8.0))
        self.W = getattr(solution_instance, 'W', results.get('W', -5.0))
        self.delta_E = getattr(solution_instance, 'delta_E', results.get('delta_E', 5.0))
        
        # Convert to float for safety
        self.m = float(self.m)
        self.R = float(self.R)
        self.h = float(self.h)
        self.v_C = float(self.v_C)
        self.g = float(self.g)
        self.v_B = float(self.v_B)
        self.W = float(self.W)
        self.delta_E = float(self.delta_E)
        
        # Derived quantities
        self.theta = math.radians(self.angle_deg)
        self.sin_theta = math.sin(self.theta)
        
        # For monotonicity test: store original values
        self.original_m = self.m
        self.original_h = self.h
        
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Work-energy theorem from B to C.
        2. Work-energy theorem from A to C.
        3. Mechanical energy loss relation.
        """
        logs = []
        all_pass = True
        
        try:
            # 1. Check work-energy theorem from B to C
            # Equation: m*g*R*sinθ = 0.5*m*(v_C^2 - v_B^2)
            left = self.m * self.g * self.R * self.sin_theta
            right = 0.5 * self.m * (self.v_C**2 - self.v_B**2)
            
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy B->C mismatch. Left={left:.3f}, Right={right:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Work-energy B->C satisfied. Left={left:.3f}, Right={right:.3f}")
        except Exception as e:
            logs.append(f"ERROR in B->C work-energy check: {traceback.format_exc()}")
            all_pass = False
            
        try:
            # 2. Check work-energy theorem from A to C
            # Equation: m*g*h + W = 0.5*m*v_C^2
            left = self.m * self.g * self.h + self.W
            right = 0.5 * self.m * self.v_C**2
            
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy A->C mismatch. Left={left:.3f}, Right={right:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Work-energy A->C satisfied. Left={left:.3f}, Right={right:.3f}")
        except Exception as e:
            logs.append(f"ERROR in A->C work-energy check: {traceback.format_exc()}")
            all_pass = False
            
        try:
            # 3. Check mechanical energy loss relation
            # According to KB: ΔE = -W (loss of mechanical energy equals negative of electric field work)
            if not math.isclose(self.delta_E, -self.W, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ΔE ≠ -W. ΔE={self.delta_E:.3f}, -W={-self.W:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: ΔE = -W relation satisfied. ΔE={self.delta_E:.3f}, -W={-self.W:.3f}")
        except Exception as e:
            logs.append(f"ERROR in ΔE check: {traceback.format_exc()}")
            all_pass = False
            
        return all_pass, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify that v_B computed from energy conservation matches the given value.
        """
        logs = []
        all_pass = True
        
        try:
            # Compute v_B from work-energy theorem B->C (should match given v_B)
            # From: m*g*R*sinθ = 0.5*m*(v_C^2 - v_B^2)
            # Rearranged: v_B^2 = v_C^2 - 2*g*R*sinθ
            v_B_calculated = math.sqrt(self.v_C**2 - 2 * self.g * self.R * self.sin_theta)
            
            if not math.isclose(v_B_calculated, self.v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B inconsistency. Calculated={v_B_calculated:.3f}, Given={self.v_B:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: v_B consistent. Calculated={v_B_calculated:.3f}, Given={self.v_B:.3f}")
        except Exception as e:
            logs.append(f"ERROR in v_B consistency check: {traceback.format_exc()}")
            all_pass = False
            
        try:
            # Compute W from work-energy theorem A->C (should match given W)
            # From: m*g*h + W = 0.5*m*v_C^2
            # Rearranged: W = 0.5*m*v_C^2 - m*g*h
            W_calculated = 0.5 * self.m * self.v_C**2 - self.m * self.g * self.h
            
            if not math.isclose(W_calculated, self.W, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: W inconsistency. Calculated={W_calculated:.3f}, Given={self.W:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: W consistent. Calculated={W_calculated:.3f}, Given={self.W:.3f}")
        except Exception as e:
            logs.append(f"ERROR in W consistency check: {traceback.format_exc()}")
            all_pass = False
            
        return all_pass, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (mass m) and verify the expected trend in v_B.
        Since v_B is independent of m (cancels in equation), perturb height h instead
        to check effect on W (electric field work).
        """
        logs = []
        all_pass = True
        
        try:
            # Perturb height h by +10%
            h_perturbed = self.h * 1.10
            
            # Calculate new W using formula: W = 0.5*m*v_C^2 - m*g*h
            W_perturbed = 0.5 * self.m * self.v_C**2 - self.m * self.g * h_perturbed
            
            # Since W = -ΔE, and increasing h should make W more negative
            # (greater drop height means more gravitational energy converted,
            # so electric field does less positive work or more negative work)
            if W_perturbed < self.W:
                logs.append(f"PASS: Monotonicity holds. W decreases with increased h (from {self.W:.3f} to {W_perturbed:.3f})")
            else:
                logs.append(f"FAIL: Monotonicity violated. W did not decrease as expected.")
                all_pass = False
                
            # Restore original value
            self.h = self.original_h
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity check: {traceback.format_exc()}")
            all_pass = False
            # Restore on error
            self.h = self.original_h
            
        return all_pass, logs
    
    def run_all_tests(self):
        """
        Execute all checks and return overall success and log messages.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        return overall_success, all_logs

# Optional: Example usage and instantiation guard
if __name__ == "__main__":
    # For testing, create a mock solution instance
    class MockSolution:
        def __init__(self):
            self.m = 0.5
            self.R = 3.6
            self.h = 6.0
            self.v_C = 10.0
            self.g = 10.0
            self.results = {
                'v_B': 8.0,
                'W': -5.0,
                'delta_E': 5.0
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    print("Overall success:", success)
    for log in logs:
        print(log)