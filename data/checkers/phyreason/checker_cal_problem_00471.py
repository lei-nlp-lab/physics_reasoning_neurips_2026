import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.results = getattr(solution_instance, 'results', {})
        
        # Get all required variables from results with sensible defaults
        self.m = float(self.results.get('m', 5e-3))
        self.g = float(self.results.get('g', 10.0))
        self.E = float(self.results.get('E', 3e4))
        self.q = float(self.results.get('q', 0.0))
        self.F = float(self.results.get('F', 0.0))
        
        # Handle theta: check if it's in degrees or radians
        # First get the value from results
        theta_val = self.results.get('theta')
        if theta_val is None:
            # Default to 60 degrees
            self.theta_deg = 60.0
            self.theta_rad = math.radians(60.0)
        else:
            theta_val = float(theta_val)
            # Determine if it's degrees or radians by checking magnitude
            # If > 2*pi, assume degrees; otherwise assume radians
            if theta_val > 2 * math.pi:
                self.theta_deg = theta_val
                self.theta_rad = math.radians(theta_val)
            else:
                self.theta_rad = theta_val
                self.theta_deg = math.degrees(theta_val)
        
        # Expected values
        self.expected_F = 0.1
        # Expected q = (5√3/3)×10^-6
        self.expected_q = 5 * math.sqrt(3) / 3 * 1e-6
        
        # Logs
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_invariants(self):
        """Check physical invariants: equilibrium of forces, Newton's laws."""
        tests_passed = True
        
        try:
            # Check 1: Horizontal force balance: q*E = F*sin(theta)
            horizontal_force_eq = self.q * self.E - self.F * math.sin(self.theta_rad)
            # Use appropriate tolerance based on expected magnitude
            if not math.isclose(horizontal_force_eq, 0.0, rel_tol=1e-2, abs_tol=1e-6):
                tests_passed &= self._log(
                    f"Horizontal force imbalance: q*E - F*sinθ = {horizontal_force_eq:.2e} (should be 0)", 
                    False
                )
            else:
                tests_passed &= self._log("Horizontal force balance satisfied", True)
            
            # Check 2: Vertical force balance: F*cos(theta) = m*g
            vertical_force_eq = self.F * math.cos(self.theta_rad) - self.m * self.g
            if not math.isclose(vertical_force_eq, 0.0, rel_tol=1e-2, abs_tol=1e-6):
                tests_passed &= self._log(
                    f"Vertical force imbalance: F*cosθ - m*g = {vertical_force_eq:.2e} (should be 0)", 
                    False
                )
            else:
                tests_passed &= self._log("Vertical force balance satisfied", True)
            
            # Check 3: Net force magnitude (should be zero)
            Fx = self.q * self.E - self.F * math.sin(self.theta_rad)
            Fy = self.F * math.cos(self.theta_rad) - self.m * self.g
            F_net = math.sqrt(Fx**2 + Fy**2)
            if not math.isclose(F_net, 0.0, rel_tol=1e-2, abs_tol=1e-6):
                tests_passed &= self._log(
                    f"Net force magnitude = {F_net:.2e} N (should be 0)", 
                    False
                )
            else:
                tests_passed &= self._log("Net force magnitude is zero (equilibrium)", True)
            
            # Check 4: Charge sign (should be positive since field is rightward and ball deflects right)
            if self.q <= 0:
                tests_passed &= self._log(f"Charge q = {self.q:.2e} C should be positive", False)
            else:
                tests_passed &= self._log("Charge sign is positive (consistent with deflection)", True)
                
        except Exception as e:
            tests_passed &= self._log(f"Error in invariant test: {e}", False)
            
        return tests_passed
    
    def test_trace_consistency(self):
        """Check consistency of intermediate results with final answers."""
        tests_passed = True
        
        try:
            # Step 1: Check q = m*g*tan(theta)/E
            q_calc = self.m * self.g * math.tan(self.theta_rad) / self.E
            # Use appropriate tolerance for small q values
            if not math.isclose(self.q, q_calc, rel_tol=1e-2, abs_tol=1e-8):
                tests_passed &= self._log(
                    f"Calculated q ({q_calc:.2e}) does not match provided q ({self.q:.2e})", 
                    False
                )
            else:
                tests_passed &= self._log(f"q consistency with formula: OK", True)
            
            # Step 2: Check F = m*g/cos(theta)
            F_calc = self.m * self.g / math.cos(self.theta_rad)
            if not math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-6):
                tests_passed &= self._log(
                    f"Calculated F ({F_calc:.2e}) does not match provided F ({self.F:.2e})", 
                    False
                )
            else:
                tests_passed &= self._log(f"F consistency with formula: OK", True)
            
            # Check against expected numerical values
            # For q, use appropriate tolerance since it's very small
            if not math.isclose(self.q, self.expected_q, rel_tol=1e-2, abs_tol=1e-8):
                tests_passed &= self._log(
                    f"q ({self.q:.2e}) differs from expected ({self.expected_q:.2e})", 
                    False
                )
            else:
                tests_passed &= self._log(f"q matches expected numerical value", True)
                
            if not math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-6):
                tests_passed &= self._log(
                    f"F ({self.F:.2e}) differs from expected ({self.expected_F:.2e})", 
                    False
                )
            else:
                tests_passed &= self._log(f"F matches expected numerical value", True)
                
        except Exception as e:
            tests_passed &= self._log(f"Error in trace consistency test: {e}", False)
            
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        tests_passed = True
        
        try:
            # Use current values as baseline
            q_base = self.q
            F_base = self.F
            
            # Perturb mass by +10%
            m_pert = self.m * 1.1
            q_pert = m_pert * self.g * math.tan(self.theta_rad) / self.E
            F_pert = m_pert * self.g / math.cos(self.theta_rad)
            
            # q should increase with mass
            if q_pert <= q_base:
                tests_passed &= self._log(
                    f"Monotonicity failure: q did not increase when mass increased", 
                    False
                )
            else:
                tests_passed &= self._log("q increases with mass (as expected)", True)
            
            # F should increase with mass
            if F_pert <= F_base:
                tests_passed &= self._log(
                    f"Monotonicity failure: F did not increase when mass increased", 
                    False
                )
            else:
                tests_passed &= self._log("F increases with mass (as expected)", True)
            
            # Perturb angle (increase to 70°)
            theta_pert_rad = math.radians(70)
            q_pert_theta = self.m * self.g * math.tan(theta_pert_rad) / self.E
            # For fixed m, g, E: q ∝ tan(θ), so increasing θ increases q (since tan is increasing in [0,90))
            if q_pert_theta <= q_base:
                tests_passed &= self._log(
                    f"Monotonicity failure: q did not increase when angle increased", 
                    False
                )
            else:
                tests_passed &= self._log("q increases with angle (as expected)", True)
            
            # F = mg/cos(θ): increasing θ decreases cos(θ), so 1/cos increases.
            F_pert_theta = self.m * self.g / math.cos(theta_pert_rad)
            if F_pert_theta <= F_base:
                tests_passed &= self._log(
                    f"Monotonicity failure: F did not increase when angle increased", 
                    False
                )
            else:
                tests_passed &= self._log("F increases with angle (as expected)", True)
                
        except Exception as e:
            tests_passed &= self._log(f"Error in monotonicity test: {e}", False)
            
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs.clear()
        overall_success = True
        
        try:
            overall_success &= self.test_invariants()
            overall_success &= self.test_trace_consistency()
            overall_success &= self.test_monotonicity()
        except Exception as e:
            self._log(f"Unexpected error in test suite: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs