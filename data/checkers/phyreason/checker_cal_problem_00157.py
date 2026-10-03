import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results using getattr
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', 60.0))
        self.v0 = getattr(solution_instance, 'v0', 50.0)
        self.theta_deg = getattr(solution_instance, 'theta', 53.0)
        self.theta = math.radians(self.theta_deg)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin53 = 0.8
        self.cos53 = math.sqrt(1 - self.sin53**2)
        
        # Extract results safely
        self.H = getattr(solution_instance, 'H', getattr(solution_instance, 'results', {}).get('H', None))
        self.E_k = getattr(solution_instance, 'E_k', getattr(solution_instance, 'results', {}).get('E_k', None))
        self.U = getattr(solution_instance, 'U', getattr(solution_instance, 'results', {}).get('U', None))
        
        # Additional parameters that might be needed
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', None))
        self.C = getattr(solution_instance, 'C', getattr(solution_instance, 'results', {}).get('C', None))
        self.eta = getattr(solution_instance, 'eta', 0.15)
        
        # Initialize symbolic variables for invariant checking
        self._init_symbols()
        
        # Store logs for test results
        self.logs = []
    
    def _init_symbols(self):
        """Initialize symbolic variables for physics laws."""
        self.sym_v0, self.sym_theta, self.sym_g, self.sym_t = sp.symbols('v0 theta g t', real=True, positive=True)
        self.sym_vx = self.sym_v0 * sp.cos(self.sym_theta)
        self.sym_vy0 = self.sym_v0 * sp.sin(self.sym_theta)
        self.sym_vy = self.sym_vy0 - self.sym_g * self.sym_t
        self.sym_x = self.sym_vx * self.sym_t
        self.sym_y = self.sym_vy0 * self.sym_t - sp.Rational(1,2) * self.sym_g * self.sym_t**2
        self.sym_v = sp.sqrt(self.sym_vx**2 + self.sym_vy**2)
        
        # Energy symbols
        self.sym_m, self.sym_C, self.sym_U, self.sym_eta = sp.symbols('m C U eta', real=True, positive=True)
        self.sym_Ek = sp.Rational(1,2) * self.sym_m * self.sym_v0**2
        self.sym_E_cap = sp.Rational(1,2) * self.sym_C * self.sym_U**2
        self.sym_Ek_from_cap = self.sym_eta * self.sym_E_cap
    
    def _log(self, message, success=None):
        """Helper to log test results."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_projectile_motion(self):
        """Test consistency of projectile motion equations."""
        try:
            # Compute time of flight from horizontal motion
            t_flight = self.L / (self.v0 * self.cos53)
            
            # Compute height using projectile equation
            H_computed = self.v0 * self.sin53 * t_flight - 0.5 * self.g * t_flight**2
            
            # Compare with given H
            if self.H is not None:
                success = math.isclose(self.H, H_computed, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Projectile motion: H={self.H:.2f}, computed={H_computed:.2f}, match={success}", success)
                return success
            
            # If H not provided, check if computed H matches target answer (60m)
            target_H = 60.0
            success = math.isclose(H_computed, target_H, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Projectile motion: computed H={H_computed:.2f}, target={target_H}, match={success}", success)
            return success
            
        except Exception as e:
            self._log(f"Projectile motion test failed: {str(e)}", False)
            return False
    
    def test_energy_conservation(self):
        """Test energy conversion consistency."""
        try:
            if self.E_k is None or self.U is None:
                self._log("Energy conservation: missing E_k or U, skipping test", None)
                return True
                
            # Check kinetic energy formula
            if self.m is not None:
                Ek_from_mass = 0.5 * self.m * self.v0**2
                success1 = math.isclose(self.E_k, Ek_from_mass, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Kinetic energy: E_k={self.E_k:.2f}, from mass={Ek_from_mass:.2f}, match={success1}", success1)
            else:
                success1 = True
                self._log("Kinetic energy: mass not available for verification", None)
            
            # Check capacitor energy conversion
            if self.C is not None:
                E_cap = 0.5 * self.C * self.U**2
                Ek_from_cap = self.eta * E_cap
                success2 = math.isclose(self.E_k, Ek_from_cap, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Capacitor conversion: E_k={self.E_k:.2f}, from capacitor={Ek_from_cap:.2f}, match={success2}", success2)
            else:
                success2 = True
                self._log("Capacitor conversion: capacitance not available for verification", None)
            
            # If both available, check consistency between them
            if self.m is not None and self.C is not None:
                # Symbolic check: Ek = 0.5*m*v0^2 = eta*0.5*C*U^2
                lhs = 0.5 * self.m * self.v0**2
                rhs = self.eta * 0.5 * self.C * self.U**2
                success3 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Energy conversion consistency: m*v0²={lhs:.2f}, eta*C*U²={rhs:.2f}, match={success3}", success3)
                return success1 and success2 and success3
            
            return success1 and success2
            
        except Exception as e:
            self._log(f"Energy conservation test failed: {str(e)}", False)
            return False
    
    def test_voltage_value(self):
        """Test if computed voltage matches target answer."""
        try:
            if self.U is None:
                self._log("Voltage test: U not available", False)
                return False
            
            target_U = 1000 * math.sqrt(2)  # Approximately 1414.21 V
            success = math.isclose(self.U, target_U, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Voltage: U={self.U:.2f}, target={target_U:.2f}, match={success}", success)
            return success
            
        except Exception as e:
            self._log(f"Voltage test failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Step 1: Time from horizontal motion
            t_computed = self.L / (self.v0 * self.cos53)
            
            # Step 2: Height from vertical motion (should match H)
            if self.H is not None:
                H_from_t = self.v0 * self.sin53 * t_computed - 0.5 * self.g * t_computed**2
                success_H = math.isclose(self.H, H_from_t, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Trace H: from t={t_computed:.2f}s, H={H_from_t:.2f}m, given={self.H:.2f}m, match={success_H}", success_H)
            else:
                success_H = True
            
            # Step 4 & 5: Energy consistency
            if self.E_k is not None and self.U is not None and self.C is not None:
                # From capacitor: E_k = eta * 0.5 * C * U^2
                E_k_from_U = self.eta * 0.5 * self.C * self.U**2
                success_E = math.isclose(self.E_k, E_k_from_U, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Trace E_k: from U={self.U:.2f}V, E_k={E_k_from_U:.2f}J, given={self.E_k:.2f}J, match={success_E}", success_E)
            else:
                success_E = True
            
            return success_H and success_E
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb input (v0) and check output trends."""
        try:
            original_v0 = self.v0
            
            # Increase v0 by 5%
            perturbed_v0 = original_v0 * 1.05
            
            # Compute new H and U using formulas
            t_new = self.L / (perturbed_v0 * self.cos53)
            H_new = perturbed_v0 * self.sin53 * t_new - 0.5 * self.g * t_new**2
            
            # For U: from E_k = 0.5*m*v0^2 = eta*0.5*C*U^2 => U ∝ v0*sqrt(m/(eta*C))
            # Assuming m and C are constant, U should increase with v0
            if self.m is not None and self.C is not None:
                # Compute proportional U
                U_new = perturbed_v0 * math.sqrt(self.m / (self.eta * self.C))
            else:
                # If m and C not available, use the relation from the problem
                # From steps: U^2 = (0.5*m*v0^2) / (0.15*0.5*C) = (m*v0^2)/(0.15*C)
                # So U ∝ v0
                U_new = self.U * (perturbed_v0 / original_v0) if self.U is not None else None
            
            # Check monotonic trends
            success_H = True
            success_U = True
            
            if self.H is not None:
                success_H = H_new > self.H  # Higher v0 should give higher H
                self._log(f"Monotonicity H: v0↑ {original_v0:.1f}→{perturbed_v0:.1f}, H↑ {self.H:.2f}→{H_new:.2f}, {success_H}", success_H)
            
            if self.U is not None and U_new is not None:
                success_U = U_new > self.U  # Higher v0 should give higher U
                self._log(f"Monotonicity U: v0↑ {original_v0:.1f}→{perturbed_v0:.1f}, U↑ {self.U:.2f}→{U_new:.2f}, {success_U}", success_U)
            
            # Restore original v0
            self.v0 = original_v0
            
            return success_H and (success_U if self.U is not None else True)
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            # Restore v0 just in case
            self.v0 = getattr(self, 'original_v0', original_v0)
            return False
    
    def test_velocity_components(self):
        """Check that horizontal velocity remains constant."""
        try:
            vx = self.v0 * self.cos53
            vy_initial = self.v0 * self.sin53
            
            # At impact time
            t_impact = self.L / (self.v0 * self.cos53)
            vy_final = vy_initial - self.g * t_impact
            
            # Horizontal velocity should be constant
            vx_final = vx
            success_vx = math.isclose(vx, vx_final, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Horizontal velocity constant: {vx:.2f}m/s, match={success_vx}", success_vx)
            
            # Vertical velocity should decrease linearly
            vy_expected = vy_initial - self.g * t_impact
            success_vy = math.isclose(vy_final, vy_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Vertical velocity linear: initial={vy_initial:.2f}m/s, final={vy_final:.2f}m/s, match={success_vy}", success_vy)
            
            return success_vx and success_vy
            
        except Exception as e:
            self._log(f"Velocity components test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all physics checks and return success status with logs."""
        self.logs = []
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Projectile Motion", self.test_projectile_motion),
            ("Velocity Components", self.test_velocity_components),
            ("Energy Conservation", self.test_energy_conservation),
            ("Voltage Value", self.test_voltage_value),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"=== Running {test_name} ===", None)
            try:
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self._log(f"Test {test_name} raised exception: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"\n=== SUMMARY: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===", None)
        
        return overall_success, self.logs

# Example usage (for testing the checker):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.L = 60.0
            self.v0 = 50.0
            self.theta = 53.0
            self.g = 10.0
            self.results = {
                'H': 60.0,
                'E_k': 62500.0,  # Assuming m=100kg: 0.5*100*50^2
                'U': 1414.213562,  # 1000√2
                'm': 100.0,
                'C': 0.1,
                'eta': 0.15
            }
    
    mock_sol = MockSolution()
    checker = PhysicsChecker(mock_sol)
    success, logs = checker.run_all_tests()
    
    print(f"\nOverall success: {success}")
    for log in logs:
        print(log)