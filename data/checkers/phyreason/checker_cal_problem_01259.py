import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Safe extraction of results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Acceleration phase inputs
        self.v = getattr(solution_instance, 'v', 30.0)  # final velocity after acceleration
        self.t_acc = getattr(solution_instance, 't', 15.0)  # acceleration time
        self.v0_acc = 0.0  # initial velocity (from rest)
        
        # Braking phase inputs
        self.x_brake = getattr(solution_instance, 'x', 90.0)  # braking distance
        self.v0_brake = self.v  # initial braking velocity equals final acceleration velocity
        self.v_brake = 0.0  # final velocity after braking
        
        # Extract computed accelerations
        self.a1 = self.results.get('a_1', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        
        # Store original values for monotonicity test
        self.original_v = self.v
        self.original_a1 = self.a1
        self.original_a2 = self.a2

    def test_invariants(self):
        """
        Verify kinematic equations for both motion phases.
        """
        logs = []
        success = True
        
        try:
            # Phase 1: Acceleration from rest to v in time t_acc
            # v = v0 + a1 * t
            v_calculated = self.v0_acc + self.a1 * self.t_acc
            if not math.isclose(v_calculated, self.v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration phase velocity mismatch. Computed: {v_calculated:.3f}, Expected: {self.v:.3f}")
                success = False
            else:
                logs.append("PASS: Acceleration phase velocity equation satisfied.")
            
            # Phase 2: Braking from v to 0 over distance x_brake
            # v_brake^2 = v0_brake^2 + 2 * a2 * x_brake
            # Since v_brake = 0: 0 = v0_brake^2 + 2 * a2 * x_brake
            left_side = self.v_brake**2
            right_side = self.v0_brake**2 + 2 * self.a2 * self.x_brake
            if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Braking phase energy equation mismatch. Left: {left_side:.3f}, Right: {right_side:.3f}")
                success = False
            else:
                logs.append("PASS: Braking phase energy equation satisfied.")
            
            # Additional check: time symmetry (optional but insightful)
            # t_brake = (v_brake - v0_brake) / a2
            t_brake = (self.v_brake - self.v0_brake) / self.a2 if self.a2 != 0 else float('inf')
            # x_brake = 0.5 * (v0_brake + v_brake) * t_brake
            if t_brake != float('inf'):
                x_check = 0.5 * (self.v0_brake + self.v_brake) * t_brake
                if not math.isclose(x_check, self.x_brake, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARNING: Braking distance inconsistent with time. Computed: {x_check:.3f}, Expected: {self.x_brake:.3f}")
                else:
                    logs.append("PASS: Braking distance consistent with time.")
            
        except Exception as e:
            logs.append(f"ERROR in invariant test: {str(e)}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate calculations with final results.
        """
        logs = []
        success = True
        
        try:
            # Check acceleration phase using displacement formula
            # x_acc = v0_acc * t_acc + 0.5 * a1 * t_acc^2
            x_acc = 0.5 * self.a1 * (self.t_acc**2)
            # From velocity: x_acc = 0.5 * (v0_acc + v) * t_acc
            x_acc_alt = 0.5 * (self.v0_acc + self.v) * self.t_acc
            if not math.isclose(x_acc, x_acc_alt, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration displacement mismatch. Method1: {x_acc:.3f}, Method2: {x_acc_alt:.3f}")
                success = False
            else:
                logs.append(f"PASS: Acceleration displacement consistent: {x_acc:.3f} m")
            
            # Check braking acceleration sign (should be negative)
            if self.a2 >= 0:
                logs.append("FAIL: Braking acceleration should be negative (deceleration).")
                success = False
            else:
                logs.append("PASS: Braking acceleration is negative (deceleration).")
            
            # Magnitude check for braking acceleration (should be 5 m/s^2)
            if not math.isclose(abs(self.a2), 5.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Braking acceleration magnitude mismatch. |a2| = {abs(self.a2):.3f}, Expected: 5.0")
                success = False
            else:
                logs.append("PASS: Braking acceleration magnitude correct.")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency test: {str(e)}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input and verify the expected trend in outputs.
        """
        logs = []
        success = True
        
        try:
            # Perturb the final velocity v by +10%
            perturbed_v = self.v * 1.1
            
            # Recompute accelerations using the same formulas
            new_a1 = perturbed_v / self.t_acc
            new_a2 = -(perturbed_v**2) / (2 * self.x_brake)
            
            # Check trends
            # a1 should increase (direct proportionality)
            if not (new_a1 > self.a1):
                logs.append(f"FAIL: a1 did not increase with v. Original: {self.a1:.3f}, New: {new_a1:.3f}")
                success = False
            else:
                logs.append(f"PASS: a1 increased from {self.a1:.3f} to {new_a1:.3f} with v.")
            
            # |a2| should increase (quadratic dependence)
            if not (abs(new_a2) > abs(self.a2)):
                logs.append(f"FAIL: |a2| did not increase with v. Original: {abs(self.a2):.3f}, New: {abs(new_a2):.3f}")
                success = False
            else:
                logs.append(f"PASS: |a2| increased from {abs(self.a2):.3f} to {abs(new_a2):.3f} with v.")
            
            # Restore original values
            self.v = self.original_v
            self.a1 = self.original_a1
            self.a2 = self.original_a2
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            # Restore on error
            self.v = self.original_v
            self.a1 = self.original_a1
            self.a2 = self.original_a2
            success = False
        
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = ["=== Physics Checker Start ==="]
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        all_logs.append(f"=== Physics Checker End: {'ALL PASS' if overall_success else 'SOME FAILURES'} ===")
        return overall_success, all_logs