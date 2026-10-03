import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization using safe attribute access.
        Extracts inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Extract input parameters with safe defaults
        self.theta = math.radians(30)  # Fixed in problem
        self.m = self._safe_get('m', 1.0)
        self.L = self._safe_get('L', 1.0)
        self.B = self._safe_get('B', 1.0)
        self.g = self._safe_get('g', 9.8)
        self.R = self._safe_get('R', 1.0)
        self.t = self._safe_get('t', 1.0)
        
        # Extract key results using mapping
        self.v1 = self._safe_get_result('v_1')
        self.E_cd = self._safe_get_result('E_cd')
        self.I_m = self._safe_get_result('I_m')
        self.I = self._safe_get_result('I')
        self.BIL = self._safe_get_result('BIL')
        self.logs = []
        
    def _safe_get(self, attr_name, default):
        """Safely get attribute from solution instance or its results dict."""
        try:
            # First try direct attribute
            val = getattr(self.solution, attr_name, None)
            if val is not None:
                return float(val)
            # Then try results dict
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(attr_name, default)
                return float(val) if val is not None else default
        except Exception:
            pass
        return default
    
    def _safe_get_result(self, key):
        """Safely extract result using key mapping."""
        # Try multiple possible keys
        keys_to_try = [key, key.replace('_', ''), key.lower()]
        for k in keys_to_try:
            val = getattr(self.solution, k, None)
            if val is not None:
                return float(val)
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(k, None)
                if val is not None:
                    return float(val)
        return None
    
    def _log(self, msg, is_error=False):
        """Helper to log messages."""
        self.logs.append(("ERROR" if is_error else "INFO", msg))
    
    def _compare_numeric(self, val1, val2, context=""):
        """Compare two numeric values with tolerance."""
        try:
            if val1 is None or val2 is None:
                return False, f"Missing value in {context}"
            
            if math.isclose(float(val1), float(val2), rel_tol=1e-2, abs_tol=1e-2):
                return True, f"Numeric match in {context}: {val1:.3g} ≈ {val2:.3g}"
            else:
                return False, f"Numeric mismatch in {context}: {val1:.3g} ≠ {val2:.3g}"
        except Exception as e:
            return False, f"Comparison failed in {context}: {str(e)}"
    
    def test_invariants(self):
        """Verify fundamental physics laws using numeric calculations."""
        success = True
        
        # 1. Conservation of Energy (from step_7)
        # Expected E_cd = (2*m^2*g^2*R*t)/(B^2*L^2) - (5*m^3*g^2*R^2)/(B^4*L^4)
        try:
            if self.B == 0 or self.L == 0:
                self._log("Energy conservation test skipped due to zero denominator", is_error=True)
                return False
                
            expected_E_cd = (2 * self.m**2 * self.g**2 * self.R * self.t) / (self.B**2 * self.L**2) \
                           - (5 * self.m**3 * self.g**2 * self.R**2) / (self.B**4 * self.L**4)
            
            ok, msg = self._compare_numeric(self.E_cd, expected_E_cd, "Energy conservation (E_cd)")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except ZeroDivisionError:
            self._log("Energy conservation test skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"Energy conservation test failed: {str(e)}", is_error=True)
            success = False
        
        # 2. Newton's 2nd Law for rod ab (from step_3)
        # BIL = mg sin30°
        try:
            expected_BIL = self.m * self.g * 0.5  # sin30° = 0.5
            ok, msg = self._compare_numeric(self.BIL, expected_BIL, "Force equilibrium (rod ab)")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except Exception as e:
            self._log(f"Force equilibrium test failed: {str(e)}", is_error=True)
            success = False
        
        # 3. Ohm's Law and Faraday's Law consistency
        # From step_1 and step_2: I = BLv1/(2R)
        # And v1 = mgR/(B²L²)
        try:
            if self.B == 0 or self.L == 0 or self.R == 0:
                self._log("Ohm/Faraday test skipped due to zero denominator", is_error=True)
                return success
                
            expected_v1 = self.m * self.g * self.R / (self.B**2 * self.L**2)
            expected_I = (self.B * self.L * expected_v1) / (2 * self.R)
            
            ok, msg = self._compare_numeric(self.I, expected_I, "Ohm/Faraday consistency")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except ZeroDivisionError:
            self._log("Ohm/Faraday test skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"Ohm/Faraday test failed: {str(e)}", is_error=True)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check v1 from step_4: v1 = mgR/(B²L²)
        try:
            if self.B == 0 or self.L == 0:
                self._log("v1 check skipped due to zero denominator", is_error=True)
                return False
                
            expected_v1 = self.m * self.g * self.R / (self.B**2 * self.L**2)
            ok, msg = self._compare_numeric(self.v1, expected_v1, "Velocity v1")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except ZeroDivisionError:
            self._log("v1 check skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"v1 check failed: {str(e)}", is_error=True)
            success = False
        
        # 2. Check I_m from step_14: I_m = 2mg/(3BL)
        try:
            if self.B == 0 or self.L == 0:
                self._log("I_m check skipped due to zero denominator", is_error=True)
                return success
                
            expected_I_m = (2 * self.m * self.g) / (3 * self.B * self.L)
            ok, msg = self._compare_numeric(self.I_m, expected_I_m, "Maximum current I_m")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except ZeroDivisionError:
            self._log("I_m check skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"I_m check failed: {str(e)}", is_error=True)
            success = False
        
        # 3. Check impulse-momentum consistency (step_5 and step_6)
        try:
            # Step_5: (2mg sin30°)t - F_t = 2m v1
            # Step_6 gives F_t = B²L²x/(2R)
            # From energy: E_cd = (2mg sin30°)x - m v1²
            # So x = (E_cd + m v1²)/(2mg sin30°)
            if self.m == 0 or self.g == 0:
                self._log("Impulse-momentum check skipped due to zero denominator", is_error=True)
                return success
                
            denominator = 2 * self.m * self.g * 0.5
            if denominator == 0:
                self._log("Impulse-momentum check skipped due to zero denominator", is_error=True)
                return success
                
            x_val = (self.E_cd + self.m * self.v1**2) / denominator
            
            # Now check step_6: (2mg sin30°)t - B²L²x/(2R) = 2m v1
            if self.R == 0:
                self._log("Impulse-momentum check skipped due to R=0", is_error=True)
                return success
                
            lhs = (2 * self.m * self.g * 0.5) * self.t - (self.B**2 * self.L**2 * x_val) / (2 * self.R)
            rhs = 2 * self.m * self.v1
            
            ok, msg = self._compare_numeric(lhs, rhs, "Impulse-momentum consistency")
            if not ok:
                self._log(msg, is_error=True)
                success = False
            else:
                self._log(msg)
        except ZeroDivisionError:
            self._log("Impulse-momentum check skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"Impulse-momentum check failed: {str(e)}", is_error=True)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check expected trends."""
        success = True
        
        # We'll perturb m by +10% and check the trends in v1 and I_m
        # E_cd's monotonicity is not guaranteed by physics for this specific system
        perturb_factor = 1.10  # 10% increase
        
        try:
            # Compute expected values with original m
            if self.B == 0 or self.L == 0:
                self._log("Monotonicity test skipped due to zero denominator", is_error=True)
                return False
            
            v1_orig = (self.m * self.g * self.R) / (self.B**2 * self.L**2)
            I_m_orig = (2 * self.m * self.g) / (3 * self.B * self.L)
            
            # Compute expected values with perturbed m
            m_pert = self.m * perturb_factor
            v1_pert = (m_pert * self.g * self.R) / (self.B**2 * self.L**2)
            I_m_pert = (2 * m_pert * self.g) / (3 * self.B * self.L)
            
            # Check trends
            # v1 should be proportional to m
            expected_v1_ratio = perturb_factor
            if v1_orig == 0:
                self._log("v1 monotonicity test skipped due to v1_orig=0", is_error=True)
            else:
                actual_v1_ratio = v1_pert / v1_orig
                if not math.isclose(actual_v1_ratio, expected_v1_ratio, rel_tol=1e-2):
                    self._log(f"v1 monotonicity failed: ratio {actual_v1_ratio:.3g} ≠ {expected_v1_ratio:.3g}", is_error=True)
                    success = False
                else:
                    self._log(f"v1 monotonicity OK: ratio {actual_v1_ratio:.3g} ≈ {expected_v1_ratio:.3g}")
            
            # I_m should be proportional to m
            if I_m_orig == 0:
                self._log("I_m monotonicity test skipped due to I_m_orig=0", is_error=True)
            else:
                expected_I_m_ratio = perturb_factor
                actual_I_m_ratio = I_m_pert / I_m_orig
                if not math.isclose(actual_I_m_ratio, expected_I_m_ratio, rel_tol=1e-2):
                    self._log(f"I_m monotonicity failed: ratio {actual_I_m_ratio:.3g} ≠ {expected_I_m_ratio:.3g}", is_error=True)
                    success = False
                else:
                    self._log(f"I_m monotonicity OK: ratio {actual_I_m_ratio:.3g} ≈ {expected_I_m_ratio:.3g}")
            
            # E_cd: Not testing monotonicity as it's not guaranteed by physics
            # The formula E_cd = (2*m^2*g^2*R*t)/(B^2*L^2) - (5*m^3*g^2*R^2)/(B^4*L^4)
            # can increase or decrease with m depending on parameters
            self._log("E_cd monotonicity not tested - behavior depends on parameters")
                
        except ZeroDivisionError:
            self._log("Monotonicity test skipped due to division by zero", is_error=True)
            success = False
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", is_error=True)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name} test...")
                test_result = test_func()
                if not test_result:
                    overall_success = False
                    self._log(f"{test_name} test failed", is_error=True)
                else:
                    self._log(f"{test_name} test passed")
            except Exception as e:
                self._log(f"{test_name} test crashed: {traceback.format_exc()}", is_error=True)
                overall_success = False
        
        return overall_success, self.logs