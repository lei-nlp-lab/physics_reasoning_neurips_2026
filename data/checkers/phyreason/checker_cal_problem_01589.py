import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.m = getattr(solution_instance, 'm', 0.1)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.E = getattr(solution_instance, 'E', 1.0)
        self.R = getattr(solution_instance, 'R', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        
        # Safely extract results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.x_PB = results.get('x_PB', 0.0)
        self.theta = results.get('theta', 0.0)  # in degrees
        self.v_C = results.get('v_C', 0.0)
        self.v_B = results.get('v_B', 0.0)
        self.E_km = results.get('E_km', 0.0)
        
    def test_invariants(self):
        """Verify conservation laws and Newton's laws from KB."""
        logs = []
        success = True
        
        try:
            # 1. Centripetal condition at point C (from step_1)
            expected_v_C = math.sqrt(self.g * self.R)
            if not math.isclose(self.v_C, expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Centripetal condition at C. v_C={self.v_C:.3f}, expected={expected_v_C:.3f}")
                success = False
            else:
                logs.append(f"PASS: Centripetal condition at C verified (v_C={self.v_C:.3f} m/s)")
            
            # 2. Work-energy from B to C (from step_2)
            expected_v_B = math.sqrt(self.v_C**2 + 4*self.g*self.R)
            if not math.isclose(self.v_B, expected_v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy B→C. v_B={self.v_B:.3f}, expected={expected_v_B:.3f}")
                success = False
            else:
                logs.append(f"PASS: Work-energy B→C verified (v_B={self.v_B:.3f} m/s)")
            
            # 3. Work-energy from P to B (from step_3)
            lhs = (self.q*self.E - self.mu*self.m*self.g) * self.x_PB
            rhs = 0.5 * self.m * self.v_B**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy P→B. LHS={lhs:.3f} J, RHS={rhs:.3f} J")
                success = False
            else:
                logs.append(f"PASS: Work-energy P→B verified (x_PB={self.x_PB:.3f} m)")
            
            # 4. Force balance at kinetic energy maximum (from step_4)
            expected_theta_deg = math.degrees(math.atan(self.q*self.E/(self.m*self.g)))
            if not math.isclose(self.theta, expected_theta_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Force balance for max KE. θ={self.theta:.1f}°, expected={expected_theta_deg:.1f}°")
                success = False
            else:
                logs.append(f"PASS: Force balance for max KE verified (θ={self.theta:.1f}°)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check that v_C from centripetal condition matches step_1
            expected_v_C = math.sqrt(self.g * self.R)
            if not math.isclose(self.v_C, expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_C inconsistent with centripetal law")
                success = False
            
            # Check that v_B from two different paths match
            v_B_from_C = math.sqrt(self.v_C**2 + 4*self.g*self.R)
            v_B_from_P = math.sqrt(2*(self.q*self.E - self.mu*self.m*self.g)*self.x_PB/self.m)
            
            if not math.isclose(v_B_from_C, v_B_from_P, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B inconsistent. From C: {v_B_from_C:.3f}, From P: {v_B_from_P:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_B consistent from both paths ({v_B_from_C:.3f} m/s)")
            
            # Check E_km calculation (step_5)
            theta_rad = math.radians(self.theta)
            expected_E_km = (0.5*self.m*self.v_B**2 + 
                           self.q*self.E*self.R*math.sin(theta_rad) - 
                           self.m*self.g*self.R*(1 - math.cos(theta_rad)))
            
            if not math.isclose(self.E_km, expected_E_km, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_km inconsistent. Computed={self.E_km:.3f} J, Expected={expected_E_km:.3f} J")
                success = False
            else:
                logs.append(f"PASS: E_km calculation verified ({self.E_km:.3f} J)")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb parameters and check physical trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase mass by 10%
            m_original = self.m
            self.m *= 1.10
            
            # Recalculate x_PB with new mass (keeping v_B fixed from original conditions)
            # Actually v_B depends on m through the energy balance, but for monotonicity
            # we should check the direct effect: x_PB = (0.5*m*v_B^2)/(qE - μmg)
            # As m increases, denominator decreases (since μmg increases), so x_PB should increase.
            new_x_PB_m = (0.5 * self.m * self.v_B**2) / (self.q * self.E - self.mu * self.m * self.g)
            self.m = m_original  # restore
            
            if new_x_PB_m <= self.x_PB:
                logs.append("FAIL: x_PB does not increase with increased mass")
                success = False
            else:
                logs.append(f"PASS: x_PB increases with mass ({self.x_PB:.3f}→{new_x_PB_m:.3f} m)")
            
            # Test 2: Increase electric field by 20%
            E_original = self.E
            self.E *= 1.20
            
            # As E increases, denominator (qE - μmg) increases, so x_PB should DECREASE
            new_x_PB_E = (0.5 * self.m * self.v_B**2) / (self.q * self.E - self.mu * self.m * self.g)
            self.E = E_original  # restore
            
            if new_x_PB_E >= self.x_PB:
                logs.append("FAIL: x_PB does not decrease with increased E field")
                success = False
            else:
                logs.append(f"PASS: x_PB decreases with E field ({self.x_PB:.3f}→{new_x_PB_E:.3f} m)")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result."""
        all_logs = []
        overall_success = True
        
        test_functions = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            try:
                test_success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"CRITICAL ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("ALL TESTS PASSED ✓")
        else:
            all_logs.append("SOME TESTS FAILED ✗")
        
        return overall_success, all_logs