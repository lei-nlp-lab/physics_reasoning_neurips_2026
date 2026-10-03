import math
import sympy as sp
from sympy import symbols, Eq, sin, cos, pi

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs (constants from problem statement)
        self.m = 75.0  # kg
        self.v0 = 2.0  # m/s
        self.theta_deg = 30.0  # degrees
        self.theta = math.radians(self.theta_deg)
        self.x = 60.0  # m
        self.t = 5.0  # s
        self.g = 10.0  # m/s^2
        
        # Safe extraction of results from solution instance
        # Using getattr with fallback to results dictionary
        self.a = getattr(solution_instance, 'a', 
                        getattr(solution_instance, 'results', {}).get('a', 0.0))
        self.f = getattr(solution_instance, 'f', 
                        getattr(solution_instance, 'results', {}).get('f', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', 
                          getattr(solution_instance, 'results', {}).get('F_N', 0.0))
        self.F_Nprime = getattr(solution_instance, 'F_Nprime', 
                               getattr(solution_instance, 'results', {}).get('F_Nprime', 0.0))
        
        # Expected values from steps analysis
        self.a_expected = 4.0  # m/s^2
        self.f_expected = 75.0  # N
        self.F_N_expected = 650.0  # N
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper method to log messages"""
        self.logs.append({
            'message': message,
            'is_error': is_error
        })
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison with 1% tolerance"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants based on KB theorems"""
        try:
            # Test 1: Newton's Second Law (net force = m*a)
            # Along incline: mg sinθ - f = m*a
            net_force = self.m * self.g * math.sin(self.theta) - self.f
            expected_net_force = self.m * self.a
            
            if not self._is_close(net_force, expected_net_force):
                self._log(f"Newton's 2nd Law violation: net_force = {net_force:.2f} N, "
                         f"m*a = {expected_net_force:.2f} N", is_error=True)
                return False
            else:
                self._log(f"Newton's 2nd Law satisfied: net_force = {net_force:.2f} N ≈ "
                         f"m*a = {expected_net_force:.2f} N")
            
            # Test 2: Newton's Third Law (F_N' = F_N)
            if not self._is_close(self.F_Nprime, self.F_N):
                self._log(f"Newton's 3rd Law violation: F_N' = {self.F_Nprime:.2f} N, "
                         f"F_N = {self.F_N:.2f} N", is_error=True)
                return False
            else:
                self._log(f"Newton's 3rd Law satisfied: F_N' = {self.F_Nprime:.2f} N ≈ "
                         f"F_N = {self.F_N:.2f} N")
            
            # Test 3: Normal force calculation F_N = mg cosθ
            expected_FN = self.m * self.g * math.cos(self.theta)
            if not self._is_close(self.F_N, expected_FN):
                self._log(f"Normal force formula violation: F_N = {self.F_N:.2f} N, "
                         f"mg cosθ = {expected_FN:.2f} N", is_error=True)
                return False
            else:
                self._log(f"Normal force formula satisfied: F_N = {self.F_N:.2f} N ≈ "
                         f"mg cosθ = {expected_FN:.2f} N")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", is_error=True)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Test 1: Verify acceleration from kinematic equation
            # a = 2*(x - v0*t)/t^2
            a_calc = 2 * (self.x - self.v0 * self.t) / (self.t ** 2)
            
            if not self._is_close(self.a, a_calc):
                self._log(f"Acceleration inconsistency: computed a = {self.a:.2f} m/s², "
                         f"kinematic formula gives a = {a_calc:.2f} m/s²", is_error=True)
                return False
            else:
                self._log(f"Acceleration consistent: computed a = {self.a:.2f} m/s² ≈ "
                         f"kinematic formula a = {a_calc:.2f} m/s²")
            
            # Test 2: Verify resistance force f = mg sinθ - ma
            f_calc = self.m * self.g * math.sin(self.theta) - self.m * self.a
            
            if not self._is_close(self.f, f_calc):
                self._log(f"Resistance force inconsistency: computed f = {self.f:.2f} N, "
                         f"formula gives f = {f_calc:.2f} N", is_error=True)
                return False
            else:
                self._log(f"Resistance force consistent: computed f = {self.f:.2f} N ≈ "
                         f"formula f = {f_calc:.2f} N")
            
            # Test 3: Check against expected values from steps analysis
            checks = [
                ('acceleration', self.a, self.a_expected, 'm/s²'),
                ('resistance force', self.f, self.f_expected, 'N'),
                ('normal force', self.F_N, self.F_N_expected, 'N'),
                ('reaction force', self.F_Nprime, self.F_N_expected, 'N')
            ]
            
            all_pass = True
            for name, actual, expected, unit in checks:
                if not self._is_close(actual, expected):
                    self._log(f"{name} mismatch: {actual:.2f} {unit} vs expected {expected:.2f} {unit}", 
                             is_error=True)
                    all_pass = False
                else:
                    self._log(f"{name} matches expected: {actual:.2f} {unit}")
            
            return all_pass
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", is_error=True)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and verify physical trends"""
        original_mass = self.m
        perturbed = False
        
        try:
            # Perturb mass by +10%
            self.m = original_mass * 1.1
            
            # Recalculate expected values using formulas
            a_perturbed = 2 * (self.x - self.v0 * self.t) / (self.t ** 2)
            f_perturbed = self.m * self.g * math.sin(self.theta) - self.m * a_perturbed
            F_N_perturbed = self.m * self.g * math.cos(self.theta)
            
            # Check monotonicity trends:
            # 1. Acceleration should remain the same (independent of mass in this kinematic formula)
            if not self._is_close(a_perturbed, self.a_expected):
                self._log(f"Acceleration should be independent of mass but changed: "
                         f"{a_perturbed:.2f} m/s² vs original {self.a_expected:.2f} m/s²", 
                         is_error=True)
                perturbed = True
            
            # 2. Both f and F_N should increase with mass (direct proportionality)
            if f_perturbed <= self.f_expected * 1.05:  # Allow 5% margin for rounding
                self._log(f"Resistance force should increase with mass: "
                         f"perturbed {f_perturbed:.2f} N vs original {self.f_expected:.2f} N", 
                         is_error=True)
                perturbed = True
            else:
                self._log(f"Resistance force monotonicity OK: increases with mass "
                         f"({f_perturbed:.2f} N > {self.f_expected:.2f} N)")
            
            if F_N_perturbed <= self.F_N_expected * 1.05:
                self._log(f"Normal force should increase with mass: "
                         f"perturbed {F_N_perturbed:.2f} N vs original {self.F_N_expected:.2f} N", 
                         is_error=True)
                perturbed = True
            else:
                self._log(f"Normal force monotonicity OK: increases with mass "
                         f"({F_N_perturbed:.2f} N > {self.F_N_expected:.2f} N)")
            
            # Restore original mass
            self.m = original_mass
            
            if perturbed:
                return False
            else:
                return True
                
        except Exception as e:
            # Restore original mass even if error occurs
            self.m = original_mass
            self._log(f"Error in test_monotonicity: {str(e)}", is_error=True)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs"""
        self.logs = []  # Clear previous logs
        
        # Run tests
        test_results = []
        
        test1 = self.test_invariants()
        test_results.append(('Physical Invariants', test1))
        
        test2 = self.test_trace_consistency()
        test_results.append(('Trace Consistency', test2))
        
        test3 = self.test_monotonicity()
        test_results.append(('Monotonicity', test3))
        
        # Determine overall success
        success = all(result for _, result in test_results)
        
        # Summary log
        self._log("\n" + "="*50)
        self._log("PHYSICS AUDIT SUMMARY")
        for test_name, passed in test_results:
            status = "PASS" if passed else "FAIL"
            self._log(f"{test_name}: {status}")
        self._log(f"\nOverall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs