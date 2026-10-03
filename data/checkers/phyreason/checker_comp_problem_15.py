import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract inputs with defaults
        self.v0 = results_dict.get('v0', 1.0)
        self.z0 = results_dict.get('z0', 1.0)
        self.E = results_dict.get('E', 1.0)
        self.theta0 = results_dict.get('theta0', 0.5)
        self.delta_theta0 = results_dict.get('delta_theta0', 0.01)
        self.x = results_dict.get('x', 1.0)
        
        # Extract all required variables from results dictionary
        # Use .get() with default values to avoid KeyError
        self.A = results_dict.get('A', 0.0)
        self.b = results_dict.get('b', 0.0)
        self.nz = results_dict.get('nz', 0.0)
        self.R = results_dict.get('R', 0.0)
        self.x1theta0 = results_dict.get('x_1theta_0', 0.0)
        
        # Handle dx/dtheta_0 with different possible keys
        dx_keys = ['dx/dtheta_0', 'dx_dtheta_0', 'dx_dtheta0']
        self.dx_dtheta0 = 0.0
        for key in dx_keys:
            if key in results_dict:
                self.dx_dtheta0 = results_dict[key]
                break
        
        # Handle εx with different possible keys
        eps_keys = ['εx', 'eps_x', 'epsilon_x']
        self.eps_x = 0.0
        for key in eps_keys:
            if key in results_dict:
                self.eps_x = results_dict[key]
                break
        
        self.x_plus = results_dict.get('x+', 0.0)
        self.x_minus = results_dict.get('x-', 0.0)
        
        # Handle |x+-x-| with different possible keys
        xdiff_keys = ['|x+-x-|', 'abs_xdiff', 'x_diff']
        self.abs_xdiff = 0.0
        for key in xdiff_keys:
            if key in results_dict:
                self.abs_xdiff = results_dict[key]
                break
        
        self.N = results_dict.get('N', 0.0)
        
        # Handle |dx_1theta_0/dtheta_0| with different possible keys
        dx1_keys = ['|dx_1theta_0/dtheta_0|', 'abs_dx1_dtheta0', 'abs_dx_dtheta0']
        self.abs_dx1_dtheta0 = 0.0
        for key in dx1_keys:
            if key in results_dict:
                self.abs_dx1_dtheta0 = results_dict[key]
                break
        
        self.x_max = results_dict.get('x_max', 0.0)
        
        # Initialize logs
        self.logs = []
        self.success = True
        
    def log(self, message, is_error=False):
        self.logs.append(message)
        if is_error:
            self.success = False
    
    def test_invariants(self):
        """Test Snell's Law and geometric invariants"""
        try:
            # Test 1: Snell's Law for circular arc
            # R = z0/sin(theta0)
            if abs(math.sin(self.theta0)) > 1e-10:  # Avoid division by zero
                R_calc = self.z0 / math.sin(self.theta0)
                if not math.isclose(R_calc, self.R, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Radius mismatch: R_calc={R_calc}, R_given={self.R}", True)
            
            # Test 2: Geometry consistency - x1(theta0) = 2R cos(theta0)
            x1_from_R = 2 * self.R * math.cos(self.theta0)
            if not math.isclose(x1_from_R, self.x1theta0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x1 mismatch: x1_from_R={x1_from_R}, x1_given={self.x1theta0}", True)
            
            # Test 3: Check derivative consistency
            # dx/dtheta0 = -2z0/sin^2(theta0)
            if abs(math.sin(self.theta0)) > 1e-10:  # Avoid division by zero
                dx_dtheta0_calc = -2 * self.z0 / (math.sin(self.theta0) ** 2)
                if not math.isclose(dx_dtheta0_calc, self.dx_dtheta0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Derivative mismatch: calc={dx_dtheta0_calc}, given={self.dx_dtheta0}", True)
            
            # Test 4: Alternative expression for derivative
            # dx/dtheta0 = -(4z0^2 + x^2)/(2z0) where x = x1(theta0)
            if abs(2 * self.z0) > 1e-10:  # Avoid division by zero
                alt_derivative = -(4 * self.z0**2 + self.x1theta0**2) / (2 * self.z0)
                if not math.isclose(alt_derivative, self.dx_dtheta0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Alternative derivative mismatch: alt={alt_derivative}, given={self.dx_dtheta0}", True)
            
            # Test 5: Energy density invariant - FIXED
            # ε(x) = (E/π) * |dθ0/dx| = (E/π) / |dx/dθ0|
            # BUT this is in terms of θ0, not x. We need to compute the θ0 corresponding to x
            # x = 2z0 cot(θ0) → θ0 = arccot(x/(2z0))
            if abs(2 * self.z0) > 1e-10:
                cot_theta0 = self.x / (2 * self.z0)
                if cot_theta0 > 0:
                    theta0_from_x = math.atan(1 / cot_theta0)  # arccot
                    # Compute dx/dθ0 at this θ0
                    if abs(math.sin(theta0_from_x)) > 1e-10:
                        dx_dtheta0_at_x = -2 * self.z0 / (math.sin(theta0_from_x) ** 2)
                        eps_calc = (self.E / math.pi) / abs(dx_dtheta0_at_x)
                        if not math.isclose(eps_calc, self.eps_x, rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"Energy density mismatch via θ0: calc={eps_calc}, given={self.eps_x}", True)
                
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}\n{traceback.format_exc()}", True)
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        try:
            # Test 1: Check A = 2z0
            if not math.isclose(2 * self.z0, self.A, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Constant A mismatch: 2z0={2*self.z0}, A={self.A}", True)
            
            # Test 2: Check b = 1
            if not math.isclose(self.b, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Constant b mismatch: b={self.b}, expected 1", True)
            
            # Test 3: Check x1(θ0) = 2z0 cot(θ0)
            if abs(math.sin(self.theta0)) > 1e-10:  # Avoid division by zero
                x1_expected = 2 * self.z0 * math.cos(self.theta0) / math.sin(self.theta0)
                if not math.isclose(x1_expected, self.x1theta0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"x1 formula mismatch: expected={x1_expected}, given={self.x1theta0}", True)
            
            # Test 4: Check energy density formula
            eps_expected = (2 * self.E * self.z0) / (math.pi * (4 * self.z0**2 + self.x**2))
            if not math.isclose(eps_expected, self.eps_x, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"ε(x) formula mismatch: expected={eps_expected}, given={self.eps_x}", True)
            
            # Test 5: Check x_max formula
            if abs(self.delta_theta0) > 1e-10:  # Avoid division by zero
                x_max_expected = (2 * self.z0 * math.cos(self.theta0)**2) / self.delta_theta0
                if not math.isclose(x_max_expected, self.x_max, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"x_max formula mismatch: expected={x_max_expected}, given={self.x_max}", True)
            
            # Test 6: Check intermediate approximations
            # |x+ - x-| ≈ |dx1/dθ0| * δθ0
            if self.abs_dx1_dtheta0 != 0:
                xdiff_approx = abs(self.abs_dx1_dtheta0) * self.delta_theta0
                if not math.isclose(xdiff_approx, self.abs_xdiff, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"x difference approximation mismatch: approx={xdiff_approx}, given={self.abs_xdiff}", True)
            
            # Test 7: Check N ≈ x1(θ0) / |x+ - x-|
            if self.abs_xdiff > 1e-10:  # Avoid division by zero
                N_approx = self.x1theta0 / self.abs_xdiff
                if not math.isclose(N_approx, self.N, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"N approximation mismatch: approx={N_approx}, given={self.N}", True)
                
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", True)
    
    def test_monotonicity(self):
        """Test that perturbations produce expected trends"""
        try:
            # Save original values
            orig_z0 = self.z0
            orig_delta_theta0 = self.delta_theta0
            orig_theta0 = self.theta0
            orig_x_max = self.x_max
            orig_x1theta0 = self.x1theta0
            
            # Test 1: Increase z0 → x_max should increase (x_max ∝ z0)
            self.z0 *= 1.1  # 10% increase
            if abs(orig_delta_theta0) > 1e-10:
                x_max_new = (2 * self.z0 * math.cos(orig_theta0)**2) / orig_delta_theta0
                if x_max_new <= orig_x_max:
                    self.log(f"Monotonicity fail: x_max should increase with z0, but {x_max_new} <= {orig_x_max}", True)
            
            # Restore z0
            self.z0 = orig_z0
            
            # Test 2: Increase δθ0 → x_max should decrease (x_max ∝ 1/δθ0)
            self.delta_theta0 *= 1.1  # 10% increase
            if abs(self.delta_theta0) > 1e-10:
                x_max_new = (2 * orig_z0 * math.cos(orig_theta0)**2) / self.delta_theta0
                if x_max_new >= orig_x_max:
                    self.log(f"Monotonicity fail: x_max should decrease with δθ0, but {x_max_new} >= {orig_x_max}", True)
            
            # Restore δθ0
            self.delta_theta0 = orig_delta_theta0
            
            # Test 3: Increase θ0 → x1(θ0) should decrease (for 0 < θ0 < π/2)
            if 0 < orig_theta0 < math.pi/2:
                self.theta0 *= 1.05  # 5% increase
                if abs(math.sin(self.theta0)) > 1e-10:
                    x1_new = 2 * orig_z0 * math.cos(self.theta0) / math.sin(self.theta0)
                    if x1_new >= orig_x1theta0:
                        self.log(f"Monotonicity fail: x1 should decrease with θ0, but {x1_new} >= {orig_x1theta0}", True)
            
            # Restore θ0
            self.theta0 = orig_theta0
            
            # Test 4: Energy density ε(x) should decrease with |x| (check at two points)
            eps_x1 = (2 * self.E * self.z0) / (math.pi * (4 * self.z0**2 + self.x**2))
            eps_x2 = (2 * self.E * self.z0) / (math.pi * (4 * self.z0**2 + (2*self.x)**2))
            if eps_x2 >= eps_x1:
                self.log(f"Monotonicity fail: ε(x) should decrease with |x|, but ε({2*self.x}) >= ε({self.x})", True)
                
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}", True)
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        self.logs = []
        self.success = True
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        return self.success, self.logs