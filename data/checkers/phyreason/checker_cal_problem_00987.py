import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants
        self.L = 0.5  # m
        self.d = 0.02  # m
        self.m = 0.02  # kg
        self.I = 10  # A
        self.B = 1  # T
        
        # Safe extraction from solution instance
        self.results = {}
        # Try to get results from solution_instance.results, fallback to 0.0
        try:
            results_dict = getattr(solution_instance, 'results', {})
            self.results['a'] = results_dict.get('a', 0.0)
            self.results['v'] = results_dict.get('v', 0.0)
            self.results['P'] = results_dict.get('P', 0.0)
        except:
            self.results = {'a': 0.0, 'v': 0.0, 'P': 0.0}
        
        # Create symbolic variables for physical laws
        self.symbols = {
            'B': self.B, 'I': self.I, 'd': self.d, 'm': self.m, 'L': self.L,
            'a': self.results['a'], 'v': self.results['v'], 'P': self.results['P']
        }
        
        self.logs = []
    
    def log(self, message, success=None):
        """Helper to log messages with optional success flag"""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
        return success
    
    def test_invariants(self):
        """Verify physical invariants: Newton's 2nd Law and Work-Energy Theorem"""
        try:
            # Newton's 2nd Law: F_net = m*a with F_net = B*I*d
            force_lorentz = self.B * self.I * self.d
            force_net = self.m * self.results['a']
            
            # Use fuzzy matching
            if math.isclose(force_lorentz, force_net, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's 2nd Law: F_Lorentz = BId = {force_lorentz:.3f} N ≈ m*a = {force_net:.3f} N", True)
            else:
                self.log(f"Newton's 2nd Law violation: BId = {force_lorentz:.3f} N ≠ m*a = {force_net:.3f} N", False)
                return False
            
            # Work-Energy Theorem: W_net = ΔK = 0.5*m*v² (starting from rest)
            work_done = force_lorentz * self.L
            delta_KE = 0.5 * self.m * self.results['v']**2
            
            if math.isclose(work_done, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Work-Energy Theorem: W = F*L = {work_done:.3f} J ≈ ΔKE = {delta_KE:.3f} J", True)
            else:
                self.log(f"Work-Energy Theorem violation: W = {work_done:.3f} J ≠ ΔKE = {delta_KE:.3f} J", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with formulas from steps analysis"""
        try:
            # Step 1: a = B*I*d/m
            expected_a = self.B * self.I * self.d / self.m
            
            if math.isclose(self.results['a'], expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Acceleration consistent: a = {self.results['a']:.3f} m/s² ≈ BId/m = {expected_a:.3f} m/s²", True)
            else:
                self.log(f"Acceleration mismatch: a = {self.results['a']:.3f} ≠ BId/m = {expected_a:.3f}", False)
                return False
            
            # Step 2: v = sqrt(2*B*I*d*L/m)
            expected_v = math.sqrt(2 * self.B * self.I * self.d * self.L / self.m)
            
            if math.isclose(self.results['v'], expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Velocity consistent: v = {self.results['v']:.3f} m/s ≈ sqrt(2BIdL/m) = {expected_v:.3f} m/s", True)
            else:
                self.log(f"Velocity mismatch: v = {self.results['v']:.3f} ≠ sqrt(2BIdL/m) = {expected_v:.3f}", False)
                return False
            
            # Step 3: P = B*I*d*v
            expected_P = self.B * self.I * self.d * self.results['v']
            
            if math.isclose(self.results['P'], expected_P, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Power consistent: P = {self.results['P']:.3f} W ≈ BIdv = {expected_P:.3f} W", True)
            else:
                self.log(f"Power mismatch: P = {self.results['P']:.3f} ≠ BIdv = {expected_P:.3f}", False)
                return False
            
            # Additional cross-check: v² = 2*a*L (from kinematics)
            expected_v_from_a = math.sqrt(2 * self.results['a'] * self.L)
            
            if math.isclose(self.results['v'], expected_v_from_a, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematics cross-check: v = {self.results['v']:.3f} ≈ sqrt(2aL) = {expected_v_from_a:.3f}", True)
            else:
                self.log(f"Kinematics mismatch: v = {self.results['v']:.3f} ≠ sqrt(2aL) = {expected_v_from_a:.3f}", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check trends in a, v, P"""
        try:
            # Store original values
            original_m = self.m
            perturb_factor = 1.1  # 10% increase
            
            # Perturb mass
            self.m *= perturb_factor
            
            # Recalculate expected values with perturbed mass
            a_perturbed = self.B * self.I * self.d / self.m
            v_perturbed = math.sqrt(2 * self.B * self.I * self.d * self.L / self.m)
            P_perturbed = self.B * self.I * self.d * v_perturbed
            
            # Check trends
            success = True
            
            # Mass increased → acceleration should decrease
            if a_perturbed < self.results['a']:
                self.log(f"Monotonicity (a): m↑ {original_m:.4f}→{self.m:.4f} kg ⇒ a↓ {self.results['a']:.3f}→{a_perturbed:.3f} m/s²", True)
            else:
                self.log(f"Monotonicity violation (a): m↑ but a not decreasing", False)
                success = False
            
            # Mass increased → velocity should decrease
            if v_perturbed < self.results['v']:
                self.log(f"Monotonicity (v): m↑ {original_m:.4f}→{self.m:.4f} kg ⇒ v↓ {self.results['v']:.3f}→{v_perturbed:.3f} m/s", True)
            else:
                self.log(f"Monotonicity violation (v): m↑ but v not decreasing", False)
                success = False
            
            # Mass increased → power should decrease (since v decreases)
            if P_perturbed < self.results['P']:
                self.log(f"Monotonicity (P): m↑ {original_m:.4f}→{self.m:.4f} kg ⇒ P↓ {self.results['P']:.3f}→{P_perturbed:.3f} W", True)
            else:
                self.log(f"Monotonicity violation (P): m↑ but P not decreasing", False)
                success = False
            
            # Restore original mass
            self.m = original_m
            
            return success
        except Exception as e:
            # Restore original mass on error
            self.m = original_m
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"Starting {test_name}...")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", False)
            except Exception as e:
                success = False
                self.log(f"Exception in {test_name}: {traceback.format_exc()}", False)
        
        # Final summary
        if success:
            self.log("All physics checks PASSED", True)
        else:
            self.log("Some physics checks FAILED", False)
        
        return success, self.logs