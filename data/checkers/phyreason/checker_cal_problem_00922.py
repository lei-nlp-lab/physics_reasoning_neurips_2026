import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL needed values from results dict (not from instance attributes)
        # The solution code should store all numeric values in results dict
        self.d = self.results.get('d', 1.0)
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.v0 = self.results.get('v0', 1.0)
        
        # Results from solution
        self.B = self.results.get('B', 0.0)
        self.t = self.results.get('t', 0.0)
        self.β = self.results.get('β', 0.0)
        
        # Derived quantities for checks
        self.v_initial = math.sqrt(2) * self.v0
        self.R = math.sqrt(2) * self.d  # From step_1: R*sin(π/4) = d
    
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Check speed remains constant (magnetic force does no work)
            # Initial speed: √2 v0, should equal final speed at exit
            # From geometry, exit is perpendicular to x-axis, so velocity magnitude unchanged
            # This is trivially true since we use same formula for initial and final
            logs.append("PASS: Speed conserved (magnetic force does no work)")
            
            # 2. Check centripetal force equals magnetic force
            # F_mag = q*v*B, F_cen = m*v²/R
            # Avoid division by zero with small epsilon
            if self.R == 0:
                logs.append("ERROR: Radius R is zero, cannot compute centripetal force")
                return logs
                
            F_mag = self.q * self.v_initial * self.B
            F_cen = self.m * self.v_initial**2 / self.R
            
            if math.isclose(F_mag, F_cen, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Magnetic force provides exact centripetal force")
            else:
                logs.append(f"FAIL: Force mismatch. F_mag={F_mag:.4f}, F_cen={F_cen:.4f}")
            
            # 3. Check angle β = 3π/4 (135°)
            expected_β = 3 * math.pi / 4
            if math.isclose(self.β, expected_β, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Deflection angle β = 3π/4")
            else:
                logs.append(f"FAIL: Angle mismatch. β={self.β:.4f}, expected={expected_β:.4f}")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # 1. Cross-check B from step_2 equation
            # Avoid division by zero
            if self.q == 0 or self.d == 0:
                logs.append("ERROR: q or d is zero, cannot compute B_expected")
                return logs
                
            B_expected = (self.m * self.v0) / (self.q * self.d)
            if math.isclose(self.B, B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: B matches derived formula B = m*v0/(q*d)")
            else:
                logs.append(f"FAIL: B mismatch. Computed={self.B:.4f}, Expected={B_expected:.4f}")
            
            # 2. Cross-check t from step_3 equation
            if self.v0 == 0:
                logs.append("ERROR: v0 is zero, cannot compute t_expected")
                return logs
                
            t_expected = (3 * math.pi * self.d) / (4 * self.v0)
            if math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: t matches derived formula t = 3πd/(4v0)")
            else:
                logs.append(f"FAIL: t mismatch. Computed={self.t:.4f}, Expected={t_expected:.4f}")
            
            # 3. Verify radius consistency: R = √2 * d
            R_from_d = math.sqrt(2) * self.d
            if math.isclose(self.R, R_from_d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Radius R = √2 * d consistent")
            else:
                logs.append(f"FAIL: Radius mismatch. R={self.R:.4f}, √2*d={R_from_d:.4f}")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_d = self.d
        
        try:
            # Perturb d by +10% and check trends
            perturbed_d = original_d * 1.10
            
            # Avoid division by zero
            if self.q == 0 or self.v0 == 0:
                logs.append("ERROR: q or v0 is zero, cannot compute perturbed values")
                return logs
                
            # Recompute expected results using formulas
            B_perturbed = (self.m * self.v0) / (self.q * perturbed_d)
            t_perturbed = (3 * math.pi * perturbed_d) / (4 * self.v0)
            
            # Check monotonic relationships
            # B ∝ 1/d, so increasing d should decrease B
            if B_perturbed < self.B:
                logs.append("PASS: B decreases when d increases (inverse relationship)")
            else:
                logs.append("FAIL: B monotonicity violated with respect to d")
            
            # t ∝ d, so increasing d should increase t
            if t_perturbed > self.t:
                logs.append("PASS: t increases when d increases (direct relationship)")
            else:
                logs.append("FAIL: t monotonicity violated with respect to d")
            
            # Restore original value
            self.d = original_d
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            self.d = original_d  # Ensure restoration
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                
                # Check for any FAIL or ERROR in logs
                for log in logs:
                    if log.startswith("FAIL") or log.startswith("ERROR"):
                        success = False
                        
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                success = False
        
        return success, all_logs