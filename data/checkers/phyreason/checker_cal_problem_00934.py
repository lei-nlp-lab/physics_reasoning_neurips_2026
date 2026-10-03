import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.solution = solution_instance
        
        # Extract results with safe defaults
        results = getattr(solution_instance, 'results', {})
        self.B = results.get('B', 0.0)
        self.E = results.get('E', 0.0)
        
        # Use the SAME numerical values that were used in the solution
        # These should match what was substituted in the code
        self.m = 9.11e-31  # electron mass (from solution code)
        self.q = 1.6e-19   # elementary charge (from solution code)
        self.v0 = 1e6      # initial velocity (from solution code)
        self.a_val = 0.1   # horizontal displacement (from solution code)
        self.b_val = 0.05  # vertical displacement (from solution code)
        
        # Intermediate values computed from geometry and kinematics
        self.R = (self.a_val**2 + self.b_val**2) / (2 * self.a_val) if self.a_val != 0 else 0.0
        self.t = self.a_val / self.v0 if self.v0 != 0 else 0.0
        self.a_m = (2 * self.b_val) / (self.t**2) if self.t != 0 else 0.0

    def test_invariants(self):
        """Verify fundamental physical laws: Lorentz force and Newton's second law."""
        logs = []
        
        try:
            # 1. Check Lorentz force provides centripetal force: q*v0*B = m*v0^2/R
            lhs_lorentz = self.q * abs(self.v0) * self.B
            rhs_centripetal = self.m * self.v0**2 / self.R if self.R != 0 else 0.0
            
            if math.isclose(lhs_lorentz, rhs_centripetal, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Lorentz force = centripetal force satisfied")
            else:
                logs.append(f"✗ Lorentz force mismatch: q*v0*B={lhs_lorentz}, m*v0²/R={rhs_centripetal}")
                
        except Exception as e:
            logs.append(f"✗ Invariant check failed: {traceback.format_exc()}")
            
        try:
            # 2. Check Newton's second law for electric field: q*E = m*a_m
            lhs_force = self.q * self.E
            rhs_ma = self.m * self.a_m
            
            if math.isclose(lhs_force, rhs_ma, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's second law (electric) satisfied")
            else:
                logs.append(f"✗ Newton's second law mismatch: q*E={lhs_force}, m*a_m={rhs_ma}")
                
        except Exception as e:
            logs.append(f"✗ Newton's law check failed: {traceback.format_exc()}")
            
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # 1. Check geometry: (a-R)^2 + b^2 = R^2
            lhs_geom = (self.a_val - self.R)**2 + self.b_val**2
            rhs_geom = self.R**2
            
            if math.isclose(lhs_geom, rhs_geom, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Geometry constraint satisfied")
            else:
                logs.append(f"✗ Geometry mismatch: (a-R)²+b²={lhs_geom}, R²={rhs_geom}")
                
        except Exception as e:
            logs.append(f"✗ Geometry check failed: {traceback.format_exc()}")
            
        try:
            # 2. Check derived B from steps
            B_expected = (2 * self.a_val * self.m * self.v0) / (self.q * (self.a_val**2 + self.b_val**2))
            if math.isclose(self.B, B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ B matches derived formula")
            else:
                logs.append(f"✗ B mismatch: result={self.B}, expected={B_expected}")
                
        except Exception as e:
            logs.append(f"✗ B consistency check failed: {traceback.format_exc()}")
            
        try:
            # 3. Check derived E from steps
            if self.a_val != 0:
                E_expected = (2 * self.b_val * self.m * self.v0**2) / (self.q * self.a_val**2)
                if math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ E matches derived formula")
                else:
                    logs.append(f"✗ E mismatch: result={self.E}, expected={E_expected}")
            else:
                logs.append("⚠ Skipped E check: a=0 leads to division by zero")
                
        except Exception as e:
            logs.append(f"✗ E consistency check failed: {traceback.format_exc()}")
            
        return logs

    def test_monotonicity(self):
        """Perturb inputs and verify physical trends."""
        logs = []
        
        try:
            # Test: Increase a, B should decrease (from B ∝ a/(a²+b²))
            perturbed_a = self.a_val * 1.1  # +10%
            perturbed_B = (2 * perturbed_a * self.m * self.v0) / (self.q * (perturbed_a**2 + self.b_val**2))
            
            if perturbed_B < self.B:
                logs.append("✓ Monotonicity: B decreases when a increases (as expected)")
            else:
                logs.append(f"✗ Monotonicity violation: B did not decrease when a increased")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity check failed: {traceback.format_exc()}")
            
        try:
            # Test: Increase b, E should increase (from E ∝ b/a²)
            perturbed_b = self.b_val * 1.1  # +10%
            perturbed_E = (2 * perturbed_b * self.m * self.v0**2) / (self.q * self.a_val**2)
            
            if perturbed_E > self.E:
                logs.append("✓ Monotonicity: E increases when b increases (as expected)")
            else:
                logs.append(f"✗ Monotonicity violation: E did not increase when b increased")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity check failed: {traceback.format_exc()}")
            
        return logs

    def run_all_tests(self):
        """Execute all checks and return (success_bool, logs_list)."""
        all_logs = []
        success = True
        
        try:
            all_logs.append("=== Invariant Tests ===")
            invariant_logs = self.test_invariants()
            all_logs.extend(invariant_logs)
            if any("✗" in log for log in invariant_logs):
                success = False
                
            all_logs.append("\n=== Trace Consistency Tests ===")
            trace_logs = self.test_trace_consistency()
            all_logs.extend(trace_logs)
            if any("✗" in log for log in trace_logs):
                success = False
                
            all_logs.append("\n=== Monotonicity Tests ===")
            monotonicity_logs = self.test_monotonicity()
            all_logs.extend(monotonicity_logs)
            if any("✗" in log for log in monotonicity_logs):
                success = False
                
        except Exception as e:
            all_logs.append(f"✗ Critical failure in test suite: {traceback.format_exc()}")
            success = False
            
        return success, all_logs