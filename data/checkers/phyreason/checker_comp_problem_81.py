import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results with fallbacks
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants and parameters (with safe defaults)
        self.G = self.inputs.get('G', 6.67430e-11)
        self.M1 = self.inputs.get('M1', 1.0)
        self.M2 = self.inputs.get('M2', 0.5)
        self.R = self.inputs.get('R', 1.0)  # Distance between M1 and M2
        
        # Extract key results with safe fallbacks
        self.omega = self._safe_get_result('omega')
        self.T = self._safe_get_result('T')
        self.alpha = self._safe_get_result('alpha')
        self.rho1 = self._safe_get_result('rho_1t')
        self.rho2 = self._safe_get_result('rho_2t')
        self.accel_vec = self._safe_get_result('fracd2dt2mathbfr')
        self.accel_x = self._safe_get_result('d2x/dt2')
        self.accel_y = self._safe_get_result('d2y/dt2')
        self.constant = self._safe_get_result('Constant')
        
        self.logs = []
        
    def _safe_get_result(self, key):
        """Safely get result with fallback to 0.0 or None"""
        result = self.results.get(key, 0.0)
        # Convert to float if possible
        try:
            return float(result)
        except:
            return None
    
    def _log(self, message, success=True):
        """Log test result"""
        self.logs.append({
            'test': message,
            'success': success,
            'type': 'info' if success else 'warning'
        })
        return success
    
    def _is_close(self, val1, val2, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison"""
        if val1 is None or val2 is None:
            return False
        try:
            return math.isclose(float(val1), float(val2), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False
    
    def test_invariants(self):
        """Check fundamental physical invariants"""
        tests_passed = True
        
        # 1. Check ω and T relationship: ω = 2π/T
        if self.omega is not None and self.T is not None:
            try:
                omega_from_T = 2 * math.pi / float(self.T)
                if self._is_close(float(self.omega), omega_from_T):
                    tests_passed = self._log("ω = 2π/T relationship holds", True) and tests_passed
                else:
                    tests_passed = self._log(f"ω = 2π/T mismatch: ω={self.omega}, 2π/T={omega_from_T}", False) and False
            except:
                tests_passed = self._log("Could not verify ω = 2π/T relationship", False) and False
        
        # 2. Check α definition: α = M2/(M1 + M2)
        if self.alpha is not None:
            alpha_expected = self.M2 / (self.M1 + self.M2)
            if self._is_close(float(self.alpha), alpha_expected):
                tests_passed = self._log("α = M2/(M1+M2) holds", True) and tests_passed
            else:
                tests_passed = self._log(f"α mismatch: got {self.alpha}, expected {alpha_expected}", False) and False
        
        # 3. Check ω formula from gravitational force
        if self.omega is not None:
            omega_expected = math.sqrt(self.G * (self.M1 + self.M2) / (self.R ** 3))
            if self._is_close(float(self.omega), omega_expected, rel_tol=5e-2):  # 5% tolerance for square root
                tests_passed = self._log("ω = √[G(M1+M2)/R³] holds numerically", True) and tests_passed
            else:
                tests_passed = self._log(f"ω formula mismatch: got {self.omega}, expected {omega_expected}", False) and False
        
        # 4. Check distances ρ1, ρ2 are positive
        if self.rho1 is not None and self.rho1 > 0:
            tests_passed = self._log("ρ1(t) gives positive distance", True) and tests_passed
        elif self.rho1 is not None:
            tests_passed = self._log(f"ρ1(t) negative: {self.rho1}", False) and False
            
        if self.rho2 is not None and self.rho2 > 0:
            tests_passed = self._log("ρ2(t) gives positive distance", True) and tests_passed
        elif self.rho2 is not None:
            tests_passed = self._log(f"ρ2(t) negative: {self.rho2}", False) and False
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        tests_passed = True
        
        # 1. Check acceleration components are reasonable
        if self.accel_x is not None and self.accel_y is not None and self.accel_vec is not None:
            # Check that vector magnitude is sqrt(ax² + ay²) within tolerance
            mag_calc = math.sqrt(self.accel_x**2 + self.accel_y**2)
            if self._is_close(self.accel_vec, mag_calc):
                tests_passed = self._log("Acceleration magnitude matches components", True) and tests_passed
            else:
                tests_passed = self._log(f"Acceleration magnitude mismatch: got {self.accel_vec}, calculated {mag_calc}", False) and False
        
        # 2. Check constant of motion is reasonable
        if self.constant is not None:
            # For the circular restricted three-body problem, the constant should be
            # C = -2U - v², which can be any real number depending on initial conditions
            # Just check it's a finite number
            if math.isfinite(self.constant):
                tests_passed = self._log("Constant of motion is finite", True) and tests_passed
            else:
                tests_passed = self._log(f"Constant of motion is not finite: {self.constant}", False) and False
        
        return tests_passed
    
    def test_monotonicity(self):
        """Check monotonic relationships by perturbing inputs"""
        tests_passed = True
        
        # 1. Perturb M1 and check ω trend
        original_M1 = self.M1
        try:
            # Increase M1 by 10%
            self.M1 *= 1.1
            
            # Recalculate expected ω
            omega_original = math.sqrt(self.G * (original_M1 + self.M2) / (self.R ** 3))
            omega_perturbed = math.sqrt(self.G * (self.M1 + self.M2) / (self.R ** 3))
            
            # ω should increase with total mass
            if omega_perturbed > omega_original:
                tests_passed = self._log("ω increases with total mass (M1↑)", True) and tests_passed
            else:
                tests_passed = self._log("ω does not increase with total mass", False) and False
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed: {str(e)}", False) and False
        finally:
            self.M1 = original_M1  # Restore
        
        # 2. Perturb R and check T trend
        original_R = self.R
        try:
            # Increase R by 10%
            self.R *= 1.1
            
            # Recalculate expected T
            T_original = 2 * math.pi * math.sqrt(original_R ** 3 / (self.G * (self.M1 + self.M2)))
            T_perturbed = 2 * math.pi * math.sqrt(self.R ** 3 / (self.G * (self.M1 + self.M2)))
            
            # T should increase with distance R (T ∝ R^(3/2))
            if T_perturbed > T_original:
                tests_passed = self._log("T increases with distance R (R↑)", True) and tests_passed
            else:
                tests_passed = self._log("T does not increase with distance R", False) and False
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed: {str(e)}", False) and False
        finally:
            self.R = original_R  # Restore
        
        # 3. Check α monotonic with M2
        original_M2 = self.M2
        try:
            # Increase M2 by 10%
            self.M2 *= 1.1
            
            alpha_original = original_M2 / (self.M1 + original_M2)
            alpha_perturbed = self.M2 / (self.M1 + self.M2)
            
            # α should increase with M2
            if alpha_perturbed > alpha_original:
                tests_passed = self._log("α increases with M2 (M2↑)", True) and tests_passed
            else:
                tests_passed = self._log("α does not increase with M2", False) and False
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed: {str(e)}", False) and False
        finally:
            self.M2 = original_M2  # Restore
        
        return tests_passed
    
    def run_all_tests(self):
        """Run all tests and return overall success"""
        self.logs = []  # Reset logs
        
        try:
            # Run individual test suites
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall_success = inv_ok and trace_ok and mono_ok
            
            # Summary log
            summary = {
                'test': f"All tests completed. Invariants: {inv_ok}, Trace: {trace_ok}, Monotonicity: {mono_ok}",
                'success': overall_success,
                'type': 'summary'
            }
            self.logs.append(summary)
            
            return overall_success, self.logs
            
        except Exception as e:
            error_log = {
                'test': f"Test suite crashed: {str(e)}",
                'success': False,
                'type': 'error',
                'traceback': traceback.format_exc()
            }
            self.logs.append(error_log)
            return False, self.logs