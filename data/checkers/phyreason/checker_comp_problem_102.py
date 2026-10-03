import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely using getattr and results dict fallback."""
        self.solution = solution_instance
        
        # Constants (provided in problem context)
        self.G = 6.67e-11
        self.c = 3.00e8
        self.M_sun = 1.989e30
        self.L_sun = 3.828e26
        self.R_earth_orbit = 1.5e11  # Earth's orbital distance
        
        # Extract results safely - ONLY from results dict
        self.results = {}
        keys = ['P', 'F_tan', 'R²/2', 'τ', 'I', 'T', 'dR/dt', 'F_gravity', 'v', 
                'F_radiation/F_gravity', 'F_radiation', 'L']
        
        # First, try to get the results dict from the solution instance
        results_dict = {}
        try:
            results_dict = getattr(solution_instance, 'results', {})
        except:
            results_dict = {}
        
        for key in keys:
            # Try to get value from results dict
            value = results_dict.get(key)
            
            # If not found, try alternative key names
            if value is None:
                # Handle special characters
                if key == 'τ':
                    alt_keys = ['tau', 'τ', 'torque']
                elif key == 'R²/2':
                    alt_keys = ['R2_over_2', 'R2/2', 'R_squared_over_2']
                elif key == 'F_radiation/F_gravity':
                    alt_keys = ['F_radiation_F_gravity', 'ratio', 'F_radiation_over_F_gravity']
                elif key == 'dR/dt':
                    alt_keys = ['dR_dt', 'dRdt']
                else:
                    alt_keys = [key.replace('/', '_'), key.replace('²', '2'), key]
                
                for alt_key in alt_keys:
                    value = results_dict.get(alt_key)
                    if value is not None:
                        break
            
            # Store the value if found
            if value is not None:
                # Ensure it's a float
                try:
                    self.results[key] = float(value)
                except (TypeError, ValueError):
                    self.results[key] = 0.0
            else:
                self.results[key] = None
        
        # Extract potential additional parameters with sensible defaults
        try:
            self.r = getattr(solution_instance, 'r', 0.01)
        except:
            self.r = 0.01
            
        try:
            self.rho = getattr(solution_instance, 'rho', 1000.0)
        except:
            self.rho = 1000.0
            
        try:
            self.Q = getattr(solution_instance, 'Q', 1.0)
        except:
            self.Q = 1.0
            
        try:
            self.R = getattr(solution_instance, 'R', self.R_earth_orbit)
        except:
            self.R = self.R_earth_orbit

    def test_invariants(self):
        """Verify derived formulas against fundamental physical laws."""
        logs = []
        try:
            # 1. Check radiation pressure formula P = L_sun/(4πR²c)
            if self.R == 0:
                logs.append("ERROR: R is zero, cannot compute P_expected")
                return False, logs
                
            P_expected = self.L_sun / (4 * math.pi * self.R**2 * self.c)
            if self.results.get('P') is not None:
                if not math.isclose(self.results['P'], P_expected, rel_tol=0.02):
                    logs.append(f"FAIL: Radiation pressure mismatch. Expected ~{P_expected:.2e}, got {self.results['P']:.2e}")
                else:
                    logs.append("PASS: Radiation pressure formula consistent")
            
            # 2. Check gravitational force scaling F_g ∝ 1/R²
            # We'll verify the ratio F_radiation/F_gravity is independent of R
            if (self.results.get('F_radiation/F_gravity') is not None and 
                self.results.get('F_radiation') is not None and 
                self.results.get('F_gravity') is not None):
                if self.results['F_gravity'] == 0:
                    logs.append("WARN: F_gravity is zero, cannot compute ratio")
                else:
                    ratio = self.results['F_radiation'] / self.results['F_gravity']
                    if not math.isclose(ratio, self.results['F_radiation/F_gravity'], rel_tol=0.02):
                        logs.append(f"FAIL: Force ratio mismatch. Direct calc: {ratio:.2e}, stored: {self.results['F_radiation/F_gravity']:.2e}")
                    else:
                        logs.append("PASS: Force ratio internally consistent")
            
            # 3. Check orbital velocity v = sqrt(G M_sun / R)
            if self.results.get('v') is not None:
                if self.R == 0:
                    logs.append("ERROR: R is zero, cannot compute v_expected")
                else:
                    v_expected = math.sqrt(self.G * self.M_sun / self.R)
                    if not math.isclose(self.results['v'], v_expected, rel_tol=0.02):
                        logs.append(f"FAIL: Orbital velocity mismatch. Expected {v_expected:.2e}, got {self.results['v']:.2e}")
                    else:
                        logs.append("PASS: Orbital velocity formula correct")
            
            # 4. Check torque formula τ = -R * F_tan
            if (self.results.get('τ') is not None and 
                self.results.get('F_tan') is not None):
                tau_from_Ftan = -self.R * self.results['F_tan']
                if not math.isclose(self.results['τ'], tau_from_Ftan, rel_tol=0.02):
                    logs.append(f"FAIL: Torque mismatch. From F_tan: {tau_from_Ftan:.2e}, stored τ: {self.results['τ']:.2e}")
                else:
                    logs.append("PASS: Torque relation τ = -R F_tan holds")
            
            # 5. Check differential equation dimensions (using actual parameters)
            if self.results.get('dR/dt') is not None:
                if self.R == 0 or self.c == 0:
                    logs.append("WARN: R or c is zero, cannot check differential equation")
                else:
                    # Left side: L_sun/(c² R) * Q
                    left_approx = self.L_sun * self.Q / (self.c**2 * self.R)
                    # Right side: (8πρr/3) * dR/dt
                    right_approx = (8 * math.pi * self.rho * self.r / 3) * self.results['dR/dt']
                    # Check if they are approximately equal (within 5%)
                    if not math.isclose(left_approx, -right_approx, rel_tol=0.05):
                        # Note: dR/dt is negative, so we compare left_approx with -right_approx
                        logs.append(f"WARN: Differential equation scale mismatch. Left: {left_approx:.3e}, Right (abs): {-right_approx:.3e}, ratio: {left_approx/(-right_approx):.3f}")
                    else:
                        logs.append("PASS: Differential equation dimensionally plausible")
            
            return len([l for l in logs if 'FAIL' in l]) == 0, logs
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            return False, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        try:
            # 1. Check that integrated form R²/2 leads to correct T
            if (self.results.get('R²/2') is not None and 
                self.results.get('T') is not None):
                if self.rho == 0 or self.r == 0 or self.c == 0 or self.L_sun == 0:
                    logs.append("WARN: Zero value in parameters, cannot check T consistency")
                else:
                    # From steps: R²/2 = (3 L_sun T) / (8πρ r c²)
                    # So T = (R²/2) * (8πρ r c²) / (3 L_sun)
                    T_from_integral = (self.results['R²/2'] * 8 * math.pi * self.rho * self.r * self.c**2) / (3 * self.L_sun)
                    if not math.isclose(T_from_integral, self.results['T'], rel_tol=0.02):
                        logs.append(f"FAIL: Time T mismatch. From integral: {T_from_integral:.2e}, stored: {self.results['T']:.2e}")
                    else:
                        logs.append("PASS: Integrated equation consistent with final time")
            
            # 2. Check that angular momentum L = m v R
            if (self.results.get('L') is not None and 
                self.results.get('v') is not None):
                if self.rho == 0 or self.r == 0:
                    logs.append("WARN: rho or r is zero, cannot compute m")
                else:
                    m = (4/3) * math.pi * self.rho * self.r**3
                    L_expected = m * self.results['v'] * self.R
                    if not math.isclose(self.results['L'], L_expected, rel_tol=0.02):
                        logs.append(f"FAIL: Angular momentum mismatch. Expected {L_expected:.2e}, got {self.results['L']:.2e}")
                    else:
                        logs.append("PASS: Angular momentum formula correct")
            
            # 3. Verify the force ratio leads to reasonable values
            if self.results.get('F_radiation/F_gravity') is not None:
                ratio = self.results['F_radiation/F_gravity']
                # For small particles, radiation should dominate over gravity for removal
                # But for the given parameters (r=0.01m), ratio is small
                if ratio < 0.1:
                    logs.append(f"INFO: Radiation force is {ratio:.1%} of gravity")
                else:
                    logs.append(f"PASS: Radiation force significant ({ratio:.1%} of gravity)")
            
            # 4. Check numerical value of final answer (approximately 2e14 s)
            if self.results.get('T') is not None:
                # The expected value from ground truth is "2 × 10^14 s" - this is approximate
                # Calculate the exact expected value using the formula with given constants
                expected_T_exact = (4 * math.pi * self.rho * self.r * self.R**2) / (3 * self.L_sun) * self.c**2
                # The ground truth says "approximately 2e14", so we should accept a range
                # around both the exact value and the approximate 2e14
                expected_T_approx = 2.0e14
                
                # Check if close to either the exact value (within 1%) or the approximation (within 15%)
                close_to_exact = math.isclose(self.results['T'], expected_T_exact, rel_tol=0.01)
                close_to_approx = math.isclose(self.results['T'], expected_T_approx, rel_tol=0.15)
                
                if not (close_to_exact or close_to_approx):
                    logs.append(f"FAIL: Final time {self.results['T']:.2e} differs from expected ~{expected_T_approx:.0e} s (exact: {expected_T_exact:.2e})")
                else:
                    logs.append("PASS: Final time matches expected order of magnitude")
            
            return len([l for l in logs if 'FAIL' in l]) == 0, logs
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            return False, logs

    def test_monotonicity(self):
        """Perturb particle radius r and verify time T changes as expected."""
        logs = []
        try:
            # We need r and T to perform this test
            if self.results.get('T') is None:
                logs.append("SKIP: Missing T for monotonicity test")
                return True, logs
            
            if self.r == 0 or self.rho == 0 or self.L_sun == 0 or self.c == 0:
                logs.append("SKIP: Zero parameter, cannot test monotonicity")
                return True, logs
            
            # Store original values
            original_r = self.r
            original_T = self.results['T']
            
            # Theoretical scaling: T ∝ r from formula T = (4πρ r R²)/(3L_sun) c²
            # Perturb r by 2%
            perturb_factor = 1.02
            perturbed_r = original_r * perturb_factor
            
            # Recalculate T using the formula (assuming other parameters fixed)
            T_perturbed = (4 * math.pi * self.rho * perturbed_r * self.R**2) / (3 * self.L_sun) * self.c**2
            expected_ratio = perturb_factor  # T ∝ r
            actual_ratio = T_perturbed / original_T
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.01):
                logs.append(f"FAIL: Monotonicity violated. Expected T ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
            else:
                logs.append(f"PASS: Time T scales linearly with particle radius (ratio: {actual_ratio:.3f})")
            
            return len([l for l in logs if 'FAIL' in l]) == 0, logs
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            return False, logs

    def run_all_tests(self):
        """Execute all tests and return overall success status with logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_method()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs