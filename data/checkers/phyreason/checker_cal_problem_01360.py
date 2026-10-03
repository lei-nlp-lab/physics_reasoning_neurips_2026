import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0))
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h = getattr(solution_instance, 'h', 1.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.sin_theta = math.sin(math.radians(self.theta))
        
        # Expected values from steps analysis
        self.expected_v = 2.0
        self.expected_a = 6.0
        self.expected_v_B = 4.0
        self.expected_t = 1.0
        
        # Test logs
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def test_invariants(self):
        """Verify energy conservation and Newton's laws from KB"""
        success = True
        
        # 1. Energy conservation during A's fall (Step 1)
        try:
            # Potential energy loss: A falls h, B rises h*sinθ
            delta_U = self.m * self.g * self.h - self.m * self.g * self.h * self.sin_theta
            # Kinetic energy gain: both masses move with speed v
            delta_K = 0.5 * (2 * self.m) * self.v**2
            
            if not math.isclose(delta_U, delta_K, rel_tol=0.02, abs_tol=0.02):
                success = self._log(f"Energy conservation mismatch: ΔU={delta_U:.3f}, ΔK={delta_K:.3f}", False)
            else:
                self._log(f"Energy conservation holds: ΔU≈ΔK={delta_U:.3f}", True)
        except Exception as e:
            self._log(f"Energy check failed: {str(e)}", False)
            success = False
        
        # 2. Newton's second law for B on incline (Step 2)
        try:
            # Net force = m*g*sinθ, acceleration = g*sinθ
            expected_a = self.g * self.sin_theta
            if not math.isclose(self.a, expected_a, rel_tol=0.02, abs_tol=0.02):
                success = self._log(f"Newton's 2nd law mismatch: a={self.a:.3f}, g*sinθ={expected_a:.3f}", False)
            else:
                self._log(f"Newton's 2nd law holds: a=g*sinθ={self.a:.3f}", True)
        except Exception as e:
            self._log(f"Newton's law check failed: {str(e)}", False)
            success = False
        
        # 3. Energy conservation for B's return (Step 3)
        try:
            # When B returns to bottom: KE_final = PE_gained + KE_initial
            # B gained PE = m*g*h*sinθ when A fell, plus initial KE from v
            total_energy = self.m * self.g * self.h * self.sin_theta + 0.5 * self.m * self.v**2
            final_KE = 0.5 * self.m * self.v_B**2
            
            if not math.isclose(total_energy, final_KE, rel_tol=0.02, abs_tol=0.02):
                success = self._log(f"Return energy mismatch: E_total={total_energy:.3f}, KE_final={final_KE:.3f}", False)
            else:
                self._log(f"Return energy conservation holds", True)
        except Exception as e:
            self._log(f"Return energy check failed: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        success = True
        
        # Check if step results match expected values
        checks = [
            (self.v, self.expected_v, "v (A's speed before impact)"),
            (self.a, self.expected_a, "a (B's acceleration)"),
            (self.v_B, self.expected_v_B, "v_B (B's return speed)"),
            (self.t, self.expected_t, "t (return time)")
        ]
        
        for actual, expected, desc in checks:
            if not math.isclose(actual, expected, rel_tol=0.02, abs_tol=0.02):
                success = self._log(f"{desc}: {actual:.3f} ≠ expected {expected:.3f}", False)
            else:
                self._log(f"{desc}: {actual:.3f} ≈ expected", True)
        
        # Verify kinematic consistency (Step 5)
        try:
            # Using v_B = v + a*t with sign convention: upward negative, downward positive
            # Initial velocity upward: v = -2 m/s, acceleration downward: a = 6 m/s²
            # Final velocity downward: v_B = 4 m/s
            # Equation: 4 = -2 + 6*t → t = 1
            calculated_t = (self.v_B - (-self.expected_v)) / self.a  # v_B = -v + a*t
            if not math.isclose(self.t, calculated_t, rel_tol=0.02, abs_tol=0.02):
                success = self._log(f"Kinematic inconsistency: t={self.t:.3f}, (v_B+v)/a={calculated_t:.3f}", False)
            else:
                self._log(f"Kinematics consistent: t=(v_B+v)/a={self.t:.3f}", True)
        except Exception as e:
            self._log(f"Kinematic check failed: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends"""
        original_h = self.h
        success = True
        
        try:
            # Test 1: Increase height h
            self.h *= 1.1  # +10%
            
            # Recalculate using step formulas
            # Step 1: v = sqrt(g*h*(1-sinθ))
            new_v = math.sqrt(self.g * self.h * (1 - self.sin_theta))
            
            # Step 3: v_B = sqrt(v² + 2*g*h*sinθ)
            new_v_B = math.sqrt(new_v**2 + 2 * self.g * self.h * self.sin_theta)
            
            # Step 5: t = (v_B + v)/a  (with sign convention)
            new_t = (new_v_B + new_v) / self.a
            
            # Check monotonicity: increasing h should increase all velocities and time
            if not (new_v > self.expected_v and new_v_B > self.expected_v_B and new_t > self.expected_t):
                success = self._log(f"Monotonicity violated for h↑: v={new_v:.3f}, v_B={new_v_B:.3f}, t={new_t:.3f}", False)
            else:
                self._log(f"Monotonicity holds for h↑: all outputs increase", True)
            
            # Restore original h
            self.h = original_h
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
            self.h = original_h  # Ensure restoration even on error
        
        return success
    
    def run_all_tests(self):
        """Execute all checks and return summary"""
        self.logs = []  # Clear previous logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Final summary
        if overall_success:
            self._log("All physics checks passed!", True)
        else:
            self._log("Some physics checks failed", False)
        
        return overall_success, self.logs