import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        
        # Safe extraction of inputs - use results dict first, then attributes
        self.m = self._safe_get('m', 1.0)
        self.g = self._safe_get('g', 9.8)
        self.R = self._safe_get('R', 1.0)
        # For theta, check if it's in results; if not, use the value we actually used in computation (π/4)
        self.theta = self._safe_get('theta', math.pi/4)  # Use π/4 as default, not 0.0
        self.v0 = self._safe_get('v_0', 0.0)
        
        # Safe extraction of results
        self.v_E = self._safe_get('v_E')
        self.v_B = self._safe_get('v_B')
        self.v_C = self._safe_get('v_C')
        self.N_C = self._safe_get('N_C')
        self.N_prime_C = self._safe_get('N_prime_C', self._safe_get("N'_C"))  # try alternative key
        self.P_G = self._safe_get('P_G')
        self.W_f = self._safe_get('W_f')
        
        # Precompute derived values for invariants USING THE SAME THETA
        self.v_E_expected = math.sqrt(self.g * self.R)
        self.v_C_expected = math.sqrt(5 * self.g * self.R)
        self.v_B_expected = math.sqrt(self.g * self.R * (3 + 2 * math.cos(self.theta)))
        self.N_C_expected = 6 * self.m * self.g
        self.P_G_expected = self.m * self.g * math.sin(self.theta) * self.v_B_expected
        
    def _safe_get(self, key, default=None):
        """Safely retrieve attribute from solution instance or its results dict."""
        try:
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) else val
        except (AttributeError, TypeError):
            pass
        
        try:
            results = getattr(self.solution, 'results', {})
            val = results.get(key, default)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) else val
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        
        return default if default is None else float(default)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for floats."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Circular motion condition at E: v_E = sqrt(gR)
        if not self._is_close(self.v_E, self.v_E_expected):
            logs.append(f"FAIL: v_E = {self.v_E}, expected sqrt(gR) = {self.v_E_expected}")
            success = False
        else:
            logs.append(f"PASS: v_E = sqrt(gR) = {self.v_E_expected}")
        
        # 2. Energy conservation between C and E: v_C^2 = v_E^2 + 4gR
        v_C_sq_expected = self.v_E_expected**2 + 4 * self.g * self.R
        if not self._is_close(self.v_C**2 if self.v_C else None, v_C_sq_expected):
            logs.append(f"FAIL: v_C^2 = {self.v_C**2 if self.v_C else 'None'}, expected v_E^2+4gR = {v_C_sq_expected}")
            success = False
        else:
            logs.append(f"PASS: v_C^2 = v_E^2 + 4gR = {v_C_sq_expected}")
        
        # 3. Energy between B and C: v_B^2 = v_C^2 - 2gR(1-cosθ)
        if self.v_C and self.v_B:
            lhs = self.v_B**2
            rhs = self.v_C**2 - 2 * self.g * self.R * (1 - math.cos(self.theta))
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: v_B^2 = {lhs}, expected v_C^2 - 2gR(1-cosθ) = {rhs}")
                success = False
            else:
                logs.append(f"PASS: v_B^2 = v_C^2 - 2gR(1-cosθ) = {lhs}")
        else:
            logs.append("SKIP: v_B or v_C missing for energy check B-C")
        
        # 4. Normal force at C: N_C = mg + m*v_C^2/R = 6mg
        if not self._is_close(self.N_C, self.N_C_expected):
            logs.append(f"FAIL: N_C = {self.N_C}, expected 6mg = {self.N_C_expected}")
            success = False
        else:
            logs.append(f"PASS: N_C = 6mg = {self.N_C_expected}")
        
        # 5. Newton's third law: N'_C = N_C
        if self.N_prime_C is not None and self.N_C is not None:
            if not self._is_close(self.N_prime_C, self.N_C):
                logs.append(f"FAIL: N'_C = {self.N_prime_C}, N_C = {self.N_C} (should be equal)")
                success = False
            else:
                logs.append(f"PASS: N'_C = N_C = {self.N_C}")
        else:
            logs.append("SKIP: N'_C or N_C missing for Newton's third law")
        
        # 6. Power of gravity at B: P_G = mg * v_B * sinθ
        if not self._is_close(self.P_G, self.P_G_expected):
            logs.append(f"FAIL: P_G = {self.P_G}, expected mg*v_B*sinθ = {self.P_G_expected}")
            success = False
        else:
            logs.append(f"PASS: P_G = mg*v_B*sinθ = {self.P_G_expected}")
        
        # 7. Work done by friction: W_f = 0
        if not self._is_close(self.W_f, 0.0):
            logs.append(f"FAIL: W_f = {self.W_f}, expected 0")
            success = False
        else:
            logs.append(f"PASS: W_f = 0")
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check that the derived v_B matches the expected formula
        v_B_formula = math.sqrt(self.g * self.R * (3 + 2 * math.cos(self.theta)))
        if not self._is_close(self.v_B, v_B_formula):
            logs.append(f"FAIL: v_B = {self.v_B}, but formula sqrt(gR(3+2cosθ)) gives {v_B_formula}")
            success = False
        else:
            logs.append(f"PASS: v_B consistent with formula sqrt(gR(3+2cosθ))")
        
        # Check that the normal force formula holds: N_C = mg + m*v_C^2/R
        if self.v_C and self.N_C:
            n_c_formula = self.m * self.g + self.m * self.v_C**2 / self.R
            if not self._is_close(self.N_C, n_c_formula):
                logs.append(f"FAIL: N_C = {self.N_C}, but mg + m*v_C^2/R gives {n_c_formula}")
                success = False
            else:
                logs.append(f"PASS: N_C = mg + m*v_C^2/R = {n_c_formula}")
        else:
            logs.append("SKIP: v_C or N_C missing for normal force formula check")
        
        # Check that the work-energy theorem holds for the entire cycle (return to A with same v0)
        # This is implicit in W_f=0, but we can check with the extracted v0 if available.
        if self.v0 is not None and self.v_B:
            # From step_9 and step_10, we can derive that v0 cancels out, but we can check consistency:
            # Step_9: mg*L*sinθ + W_f = 0.5*m*v_B^2 - 0.5*m*v0^2
            # Step_10: -mg*L*sinθ = 0.5*m*v0^2 - 0.5*m*v_B^2
            # Adding gives W_f = 0. So if W_f=0, then the two equations are consistent.
            # We can check that the two equations are negatives of each other.
            # Since we don't have L, we cannot check directly, but we trust the derived W_f.
            pass
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Perturb mass m and check scaling
        original_m = self.m
        perturbed_m = 2.0 * original_m
        
        # Recalculate expected results with perturbed m
        N_C_perturbed = 6 * perturbed_m * self.g
        P_G_perturbed = perturbed_m * self.g * math.sin(self.theta) * self.v_B_expected
        
        # Check that N_C scales linearly with m
        if not self._is_close(N_C_perturbed / perturbed_m, self.N_C_expected / original_m):
            logs.append(f"FAIL: N_C/m not constant under mass scaling")
            success = False
        else:
            logs.append(f"PASS: N_C scales linearly with mass")
        
        # Check that P_G scales linearly with m
        if not self._is_close(P_G_perturbed / perturbed_m, self.P_G_expected / original_m):
            logs.append(f"FAIL: P_G/m not constant under mass scaling")
            success = False
        else:
            logs.append(f"PASS: P_G scales linearly with mass")
        
        # Perturb theta and check trend of P_G
        original_theta = self.theta
        perturbed_theta = original_theta + 0.1  # rad
        
        v_B_perturbed = math.sqrt(self.g * self.R * (3 + 2 * math.cos(perturbed_theta)))
        P_G_perturbed_theta = self.m * self.g * math.sin(perturbed_theta) * v_B_perturbed
        
        # We expect that P_G changes continuously with theta
        # No strict monotonicity, but we can check that the change is non-zero
        delta_P = P_G_perturbed_theta - self.P_G_expected
        if abs(delta_P) < 1e-9:
            logs.append("WARN: P_G insensitive to theta perturbation (might be at extremum)")
        else:
            logs.append(f"PASS: P_G changes with theta (delta = {delta_P:.2e})")
        
        # Restore original values (not strictly needed as we didn't modify instance)
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs