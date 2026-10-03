import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants from the problem statement
        self.r = 0.6  # m
        self.R = 1.0  # m
        self.h = 1.0  # m
        self.m = 2.0  # kg
        self.omega = 5.0  # rad/s
        self.g = 10.0  # m/s^2
        
        # Safely extract results from solution_instance using getattr and .get()
        self.results = {}
        # Use safe attribute access: try solution_instance.attr, then solution_instance.results.get()
        keys_to_extract = ['P', 'x', 't', 'Tcostheta', 'v_0', 'sintheta', 's']
        for key in keys_to_extract:
            # First try direct attribute, then results dict, then default to 0.0
            val = getattr(solution_instance, key, solution_instance.results.get(key, 0.0))
            self.results[key] = float(val) if val is not None else 0.0
        
        # For convenience, define the extracted values as attributes
        self.P = self.results.get('P', 0.0)
        self.x = self.results.get('x', 0.0)
        self.t = self.results.get('t', 0.0)
        self.Tcosθ = self.results.get('Tcostheta', 0.0)
        self.v0 = self.results.get('v_0', 0.0)
        self.sinθ = self.results.get('sintheta', 0.0)
        self.s = self.results.get('s', 0.0)
        
        self.logs = []
    
    def log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def test_invariants(self):
        """Verify physical invariants: kinematics, centripetal force, power."""
        success = True
        try:
            # 1. Kinematics: time of flight from free fall
            t_expected = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Time of flight mismatch: {self.t} vs {t_expected}", False)
                success = False
            else:
                self.log(f"Time of flight ok: {self.t:.4f}", True)
            
            # 2. Linear velocity from angular velocity
            v0_expected = self.omega * self.R
            if not math.isclose(self.v0, v0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Initial velocity mismatch: {self.v0} vs {v0_expected}", False)
                success = False
            else:
                self.log(f"Initial velocity ok: {self.v0:.4f}", True)
            
            # 3. Horizontal displacement
            x_expected = self.v0 * self.t
            if not math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Horizontal displacement mismatch: {self.x} vs {x_expected}", False)
                success = False
            else:
                self.log(f"Horizontal displacement ok: {self.x:.4f}", True)
            
            # 4. Distance to center O (3D Pythagorean theorem)
            s_expected = math.sqrt(self.R**2 + self.x**2 + self.h**2)
            if not math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Distance to center O mismatch: {self.s} vs {s_expected}", False)
                success = False
            else:
                self.log(f"Distance to center O ok: {self.s:.4f}", True)
            
            # 5. sinθ geometry
            sinθ_expected = self.r / self.R
            if not math.isclose(self.sinθ, sinθ_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"sinθ mismatch: {self.sinθ} vs {sinθ_expected}", False)
                success = False
            else:
                self.log(f"sinθ ok: {self.sinθ:.4f}", True)
            
            # 6. Centripetal force: Tcosθ = m * v0^2 / R
            Tcosθ_expected = self.m * self.v0**2 / self.R
            if not math.isclose(self.Tcosθ, Tcosθ_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Tcosθ mismatch: {self.Tcosθ} vs {Tcosθ_expected}", False)
                success = False
            else:
                self.log(f"Tcosθ ok: {self.Tcosθ:.4f}", True)
            
            # 7. Power: P = T * v0 * sinθ, with T = Tcosθ / cosθ, and cosθ = sqrt(1 - sinθ^2)
            cosθ = math.sqrt(1 - self.sinθ**2)
            T_expected = self.Tcosθ / cosθ
            P_expected = T_expected * self.v0 * self.sinθ
            if not math.isclose(self.P, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Power mismatch: {self.P} vs {P_expected}", False)
                success = False
            else:
                self.log(f"Power ok: {self.P:.4f}", True)
                
        except Exception as e:
            self.log(f"Error in invariants: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer using the same physics formulas."""
        success = True
        try:
            # Compute target values using the same physics as test_invariants
            # Time of flight
            t_target = math.sqrt(2 * self.h / self.g)
            # Initial velocity
            v0_target = self.omega * self.R
            # Horizontal displacement
            x_target = v0_target * t_target
            # Distance to center O
            s_target = math.sqrt(self.R**2 + x_target**2 + self.h**2)
            # sinθ
            sinθ_target = self.r / self.R
            cosθ_target = math.sqrt(1 - sinθ_target**2)
            # Tcosθ
            Tcosθ_target = self.m * v0_target**2 / self.R
            # T
            T_target = Tcosθ_target / cosθ_target
            # Power
            P_target = T_target * v0_target * sinθ_target
            
            # Check final distance s
            if not math.isclose(self.s, s_target, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final distance s does not match computed target: {self.s} vs {s_target}", False)
                success = False
            else:
                self.log(f"Final distance s matches computed target: {s_target:.4f} m", True)
            
            # Check final power P
            if not math.isclose(self.P, P_target, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Final power P does not match computed target: {self.P} vs {P_target}", False)
                success = False
            else:
                self.log(f"Final power P matches computed target: {P_target:.4f} W", True)
                
        except Exception as e:
            self.log(f"Error in trace consistency: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input (mass m) and check trend in power P using numeric calculations."""
        original_m = self.m
        success = True
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recompute power using the formulas (numeric)
            v0 = self.omega * self.R
            sinθ = self.r / self.R
            cosθ = math.sqrt(1 - sinθ**2)
            Tcosθ = perturbed_m * v0**2 / self.R
            T = Tcosθ / cosθ
            P_perturbed = T * v0 * sinθ
            
            # Check monotonicity: if mass increases, power should increase
            if P_perturbed > self.P:
                self.log(f"Monotonicity holds: P increases with mass (ΔP = {P_perturbed - self.P:.4f})", True)
            else:
                self.log(f"Monotonicity violation: P does not increase with mass", False)
                success = False
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            self.log(f"Error in monotonicity test: {traceback.format_exc()}", False)
            success = False
            # Ensure mass is restored even on error
            self.m = original_m
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs)."""
        self.logs = []
        overall_success = True
        
        self.log("Starting PhysicsChecker...")
        
        # Run tests
        inv_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        mono_success = self.test_monotonicity()
        
        overall_success = inv_success and trace_success and mono_success
        
        self.log(f"All tests passed: {overall_success}", overall_success)
        
        return overall_success, self.logs