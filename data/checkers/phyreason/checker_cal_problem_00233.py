import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with safe defaults
        self.inputs['m'] = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', None))
        self.inputs['q'] = getattr(solution_instance, 'q', getattr(solution_instance, 'results', {}).get('q', None))
        self.inputs['v0'] = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', None))
        self.inputs['R'] = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', None))
        self.inputs['N'] = getattr(solution_instance, 'N', getattr(solution_instance, 'results', {}).get('N', None))
        
        # Extract results with safe defaults
        self.results['B'] = getattr(solution_instance, 'B', getattr(solution_instance, 'results', {}).get('B', None))
        self.results['omega'] = getattr(solution_instance, 'omega', getattr(solution_instance, 'results', {}).get('omega', None))
        self.results['F'] = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', None))
        self.results['theta'] = getattr(solution_instance, 'theta', getattr(solution_instance, 'results', {}).get('theta', None))
        self.results['theta_prime'] = getattr(solution_instance, 'theta_prime', getattr(solution_instance, 'results', {}).get('theta_prime', None))
        
        # Intermediate results from steps analysis
        self.intermediate = {}
        keys_to_extract = ['t', 'tprime', 'Rprime', 'v', 'omegat', 'omegaprime', 'omegaprimetprime']
        for key in keys_to_extract:
            self.intermediate[key] = getattr(solution_instance, key, getattr(solution_instance, 'results', {}).get(key, None))
        
        self.logs = []

    def log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def test_invariants(self):
        """Test fundamental physical laws from KB reference and steps analysis."""
        success = True
        
        # 1. Magnetic field from Lorentz force as centripetal force
        if all(x is not None for x in [self.inputs.get('m'), self.inputs.get('v0'), self.inputs.get('q'), self.inputs.get('R'), self.results.get('B')]):
            B_expected = self.inputs['m'] * self.inputs['v0'] / (self.inputs['q'] * self.inputs['R'])
            if not math.isclose(self.results['B'], B_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Magnetic field mismatch: expected {B_expected:.4f}, got {self.results['B']:.4f}", False)
                success = False
            else:
                self.log(f"Magnetic field correct: B = m*v0/(q*R) = {self.results['B']:.4f}", True)
        else:
            self.log("Skipping magnetic field check due to missing data", None)
        
        # 2. Angular velocity for ion exiting along -y axis
        if all(x is not None for x in [self.results.get('omega'), self.inputs.get('v0'), self.inputs.get('R')]):
            # ω = (4k+1)*v0/R, k non-negative integer
            # Check if (ω*R/v0 - 1)/4 is close to an integer
            ratio = self.results['omega'] * self.inputs['R'] / self.inputs['v0']
            k_candidate = (ratio - 1) / 4
            if abs(k_candidate - round(k_candidate)) < 1e-2 and k_candidate >= -1e-2:
                self.log(f"Angular velocity matches ω = (4k+1)*v0/R with k≈{round(k_candidate)}", True)
            else:
                self.log(f"Angular velocity doesn't match expected form: ω*R/v0 = {ratio:.4f}", False)
                success = False
        else:
            self.log("Skipping angular velocity check due to missing data", None)
        
        # 3. Average force from impulse-momentum theorem
        if all(x is not None for x in [self.results.get('F'), self.inputs.get('N'), self.inputs.get('m'), 
                                       self.inputs.get('v0'), self.inputs.get('R'), self.results.get('theta')]):
            theta = self.results['theta']
            # Use purely numeric formula: F = (2nπ + θ) * N * m * v0² * tan(θ/2) / [2π(π-θ) R]
            # Check for integer n = 0,1,2,3,4
            found_match = False
            for n_val in range(0, 5):
                numerator = (2*n_val*math.pi + theta) * self.inputs['N'] * self.inputs['m'] * (self.inputs['v0']**2) * math.tan(theta/2)
                denominator = 2*math.pi * (math.pi - theta) * self.inputs['R']
                F_expected_n = numerator / denominator
                if math.isclose(self.results['F'], F_expected_n, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Average force matches formula with n={n_val}: F = {self.results['F']:.4f}", True)
                    found_match = True
                    break
            
            if not found_match:
                self.log(f"Average force doesn't match formula for n=0..4. Got {self.results['F']:.4f}", False)
                success = False
        else:
            self.log("Skipping average force check due to missing data", None)
        
        # 4. Detection angles θ' should be π/6 or 5π/6
        if self.results.get('theta_prime') is not None:
            theta_prime = self.results['theta_prime']
            expected_vals = [math.pi/6, 5*math.pi/6]
            matches = any(math.isclose(theta_prime, val, rel_tol=1e-2, abs_tol=1e-2) for val in expected_vals)
            if matches:
                self.log(f"Detection angle θ' = {theta_prime:.4f} matches expected values (π/6 or 5π/6)", True)
            else:
                self.log(f"Detection angle θ' = {theta_prime:.4f} not π/6 or 5π/6", False)
                success = False
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check velocity for ion striking at angle θ
        if all(x is not None for x in [self.intermediate.get('v'), self.inputs.get('v0'), self.results.get('theta')]):
            v_expected = self.inputs['v0'] * math.tan(self.results['theta']/2)
            if math.isclose(self.intermediate['v'], v_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Velocity v = v0*tan(θ/2): {self.intermediate['v']:.4f}", True)
            else:
                self.log(f"Velocity mismatch: expected {v_expected:.4f}, got {self.intermediate['v']:.4f}", False)
                success = False
        
        # Check radius R' for circular path
        if all(x is not None for x in [self.intermediate.get('Rprime'), self.inputs.get('R'), self.results.get('theta')]):
            Rprime_expected = self.inputs['R'] * math.tan(self.results['theta']/2)
            if math.isclose(self.intermediate['Rprime'], Rprime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Radius R' = R*tan(θ/2): {self.intermediate['Rprime']:.4f}", True)
            else:
                self.log(f"Radius R' mismatch: expected {Rprime_expected:.4f}, got {self.intermediate['Rprime']:.4f}", False)
                success = False
        
        # Check time t' in magnetic field
        if all(x is not None for x in [self.intermediate.get('tprime'), self.inputs.get('R'), self.inputs.get('v0'), self.results.get('theta')]):
            tprime_expected = (math.pi - self.results['theta']) * self.inputs['R'] / self.inputs['v0']
            if math.isclose(self.intermediate['tprime'], tprime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Time t' = (π-θ)*R/v0: {self.intermediate['tprime']:.4f}", True)
            else:
                self.log(f"Time t' mismatch: expected {tprime_expected:.4f}, got {self.intermediate['tprime']:.4f}", False)
                success = False
        
        # Check ω'*t' = 2nπ + θ for some integer n
        if all(x is not None for x in [self.intermediate.get('omegaprime'), self.intermediate.get('tprime'), self.results.get('theta')]):
            product = self.intermediate['omegaprime'] * self.intermediate['tprime']
            # Find n such that product ≈ 2nπ + θ
            n_candidate = (product - self.results['theta']) / (2 * math.pi)
            if abs(n_candidate - round(n_candidate)) < 1e-2:
                self.log(f"ω'*t' = 2nπ + θ holds with n≈{round(n_candidate)}", True)
            else:
                self.log(f"ω'*t' = {product:.4f} not equal to 2nπ + θ for integer n", False)
                success = False
        
        # Check impulse-momentum theorem: F*(2π/ω') = N*m*v
        if all(x is not None for x in [self.results.get('F'), self.intermediate.get('omegaprime'), 
                                       self.inputs.get('N'), self.inputs.get('m'), self.intermediate.get('v')]):
            left = self.results['F'] * (2 * math.pi / self.intermediate['omegaprime'])
            right = self.inputs['N'] * self.inputs['m'] * self.intermediate['v']
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Impulse-momentum theorem: F*(2π/ω') = N*m*v holds ({left:.4f} ≈ {right:.4f})", True)
            else:
                self.log(f"Impulse-momentum theorem mismatch: {left:.4f} ≠ {right:.4f}", False)
                success = False
        
        return success

    def test_monotonicity(self):
        """Perturb an input and check output trends."""
        success = True
        
        # Perturb v0 and check B = m*v0/(q*R) relationship
        if all(x is not None for x in [self.inputs.get('m'), self.inputs.get('v0'), self.inputs.get('q'), self.inputs.get('R')]):
            original_v0 = self.inputs['v0']
            perturbed_v0 = original_v0 * 1.05  # +5% increase
            
            # Recalculate B
            B_original = self.inputs['m'] * original_v0 / (self.inputs['q'] * self.inputs['R'])
            B_perturbed = self.inputs['m'] * perturbed_v0 / (self.inputs['q'] * self.inputs['R'])
            
            # Check proportional relationship
            ratio_v0 = perturbed_v0 / original_v0
            ratio_B = B_perturbed / B_original
            
            if math.isclose(ratio_B, ratio_v0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"B ∝ v0 holds: v0 increased by {ratio_v0:.2%}, B increased by {ratio_B:.2%}", True)
            else:
                self.log(f"B ∝ v0 violated: v0 ratio {ratio_v0:.4f}, B ratio {ratio_B:.4f}", False)
                success = False
        else:
            self.log("Skipping v0-B monotonicity check due to missing data", None)
        
        # Perturb m and check F from impulse-momentum theorem (F ∝ m if other parameters fixed)
        if all(x is not None for x in [self.inputs.get('m'), self.results.get('F'), self.intermediate.get('v'), 
                                       self.inputs.get('N'), self.intermediate.get('omegaprime')]):
            original_m = self.inputs['m']
            perturbed_m = original_m * 1.05  # +5% increase
            
            # Recalculate F using F = (ω'/(2π)) * N * m * v
            F_original = self.results['F']
            F_perturbed = (self.intermediate['omegaprime']/(2*math.pi)) * self.inputs['N'] * perturbed_m * self.intermediate['v']
            
            ratio_m = perturbed_m / original_m
            ratio_F = F_perturbed / F_original
            
            if math.isclose(ratio_F, ratio_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"F ∝ m holds: m increased by {ratio_m:.2%}, F increased by {ratio_F:.2%}", True)
            else:
                self.log(f"F ∝ m violated: m ratio {ratio_m:.4f}, F ratio {ratio_F:.4f}", False)
                success = False
        else:
            self.log("Skipping m-F monotonicity check due to missing data", None)
        
        return success

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        success = True
        
        try:
            self.log("Starting Physics Audit...")
            
            # Run tests
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            overall_success = inv_success and trace_success and mono_success
            
            self.log(f"Physics Audit Complete: {'PASS' if overall_success else 'FAIL'}", overall_success)
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"Physics audit failed with exception: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg, False)
            return False, self.logs