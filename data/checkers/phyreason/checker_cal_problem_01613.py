import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # Input parameters (with safe access)
        # First try to get from results, then from attributes
        self.m = self._safe_get(solution_instance, 'm')
        self.q = self._safe_get(solution_instance, 'q')
        self.E = self._safe_get(solution_instance, 'E')
        self.R = self._safe_get(solution_instance, 'R')
        self.B = self._safe_get(solution_instance, 'B')  # magnetic field in velocity selector
        
        # If any are None, try to get from results with different keys
        if self.m is None:
            self.m = self._safe_get(solution_instance, 'm_val')
        if self.q is None:
            self.q = self._safe_get(solution_instance, 'q_val')
        if self.E is None:
            self.E = self._safe_get(solution_instance, 'E_val')
        if self.R is None:
            self.R = self._safe_get(solution_instance, 'R_val')
        if self.B is None:
            self.B = self._safe_get(solution_instance, 'B_val')
        
        # Results (using key mapping)
        self.v0 = self._safe_get(solution_instance, 'v_0')
        self.B0_min = self._safe_get(solution_instance, 'B_0min')
        self.B0_max = self._safe_get(solution_instance, 'B_0max')
        self.t = self._safe_get(solution_instance, 't')
        
        # Direction is optional - not required to be in results
        self.direction = self._safe_get(solution_instance, 'direction', default="")
        
        # Geometric bounds for radius (from steps analysis)
        self.r_min = 5 * self.R / 4 if self.R is not None else None
        self.r_max = 7 * self.R / 4 if self.R is not None else None
        
    def _safe_get(self, obj, key, default=None):
        """
        Safely retrieve attribute from solution_instance.
        Priority: direct attribute -> results dict -> default.
        """
        value = getattr(obj, key, None)
        if value is None and hasattr(obj, 'results'):
            value = obj.results.get(key, default)
        return value if value is not None else default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # Handle non-numeric values (e.g., strings)
            return False
    
    def test_invariants(self):
        """Check fundamental physical laws."""
        logs = []
        success = True
        
        # 1. Velocity selector condition: v0 = E/B
        if None not in (self.v0, self.E, self.B) and self.B != 0:
            expected_v0 = self.E / self.B
            if not self._is_close(self.v0, expected_v0):
                logs.append(f"FAIL: v0 = {self.v0} but expected E/B = {expected_v0}")
                success = False
            else:
                logs.append("PASS: Velocity selector condition v0 = E/B holds.")
        else:
            logs.append("SKIP: Missing data for velocity selector check.")
            success = False  # Treat missing data as failure for robustness
        
        # 2. Lorentz force as centripetal force (numerical check with r bounds)
        if None not in (self.v0, self.B0_min, self.B0_max, self.m, self.q, self.r_min, self.r_max):
            # For B0_min, compute corresponding radius: r = m*v0/(q*B0_min)
            if self.q != 0 and self.B0_min != 0:
                r_from_B0_min = self.m * self.v0 / (self.q * self.B0_min)
                if not self._is_close(r_from_B0_min, self.r_max):
                    logs.append(f"FAIL: With B0_min, r = {r_from_B0_min} but expected r_max = {self.r_max}")
                    success = False
                else:
                    logs.append("PASS: B0_min corresponds to r_max.")
            else:
                logs.append("SKIP: Division by zero in B0_min radius calculation.")
                success = False
            
            # For B0_max, compute corresponding radius: r = m*v0/(q*B0_max)
            if self.q != 0 and self.B0_max != 0:
                r_from_B0_max = self.m * self.v0 / (self.q * self.B0_max)
                if not self._is_close(r_from_B0_max, self.r_min):
                    logs.append(f"FAIL: With B0_max, r = {r_from_B0_max} but expected r_min = {self.r_min}")
                    success = False
                else:
                    logs.append("PASS: B0_max corresponds to r_min.")
            else:
                logs.append("SKIP: Division by zero in B0_max radius calculation.")
                success = False
        else:
            logs.append("SKIP: Missing data for Lorentz force centripetal check.")
            success = False
        
        # 3. Magnetic force does no work (speed constant)
        # Since motion is in magnetic field only, speed should remain v0.
        # We can check that the centripetal acceleration formula uses v0, implying constant speed.
        if None not in (self.v0, self.B0_min, self.m, self.q, self.r_max, self.r_min):
            # Verify Lorentz = centripetal for both bounds
            for B0, r in [(self.B0_min, self.r_max), (self.B0_max, self.r_min)]:
                if r != 0:
                    left = self.q * self.v0 * B0
                    right = self.m * self.v0**2 / r
                    if not self._is_close(left, right):
                        logs.append(f"FAIL: Lorentz != centripetal for B0={B0}, r={r}. Left={left}, Right={right}")
                        success = False
            if success:
                logs.append("PASS: Magnetic force provides centripetal force (no work done, speed constant).")
        else:
            logs.append("SKIP: Missing data for work-energy check.")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final expressions."""
        logs = []
        success = True
        
        # Check that B0_min and B0_max formulas match the derived range
        if None not in (self.m, self.E, self.q, self.B, self.R, self.B0_min, self.B0_max):
            if self.q != 0 and self.B != 0 and self.R != 0:
                expected_B0_min = 4 * self.m * self.E / (7 * self.q * self.B * self.R)
                expected_B0_max = 4 * self.m * self.E / (5 * self.q * self.B * self.R)
                
                if not self._is_close(self.B0_min, expected_B0_min):
                    logs.append(f"FAIL: B0_min = {self.B0_min} but expected 4mE/(7qBR) = {expected_B0_min}")
                    success = False
                else:
                    logs.append("PASS: B0_min matches formula 4mE/(7qBR).")
                
                if not self._is_close(self.B0_max, expected_B0_max):
                    logs.append(f"FAIL: B0_max = {self.B0_max} but expected 4mE/(5qBR) = {expected_B0_max}")
                    success = False
                else:
                    logs.append("PASS: B0_max matches formula 4mE/(5qBR).")
            else:
                logs.append("SKIP: Division by zero in B0 formula calculation.")
                success = False
        else:
            logs.append("SKIP: Missing data for B0_min/B0_max formula check.")
            success = False
        
        # Check time t = π*m/(q*B0_min) = 7πBR/(4E)
        if None not in (self.t, self.m, self.q, self.B0_min, self.B, self.R, self.E):
            if self.q != 0 and self.B0_min != 0 and self.E != 0:
                t_from_B0_min = math.pi * self.m / (self.q * self.B0_min)
                if not self._is_close(self.t, t_from_B0_min):
                    logs.append(f"FAIL: t = {self.t} but π*m/(q*B0_min) = {t_from_B0_min}")
                    success = False
                else:
                    logs.append("PASS: t matches π*m/(q*B0_min).")
                
                t_alternative = 7 * math.pi * self.B * self.R / (4 * self.E)
                if not self._is_close(self.t, t_alternative):
                    logs.append(f"FAIL: t = {self.t} but 7πBR/(4E) = {t_alternative}")
                    success = False
                else:
                    logs.append("PASS: t matches 7πBR/(4E).")
            else:
                logs.append("SKIP: Division by zero in time calculation.")
                success = False
        else:
            logs.append("SKIP: Missing data for time consistency check.")
            success = False
        
        # Check direction string (case-insensitive) - handle separately without float conversion
        # Direction is optional, so don't fail if it's missing
        expected_direction_phrases = ["perpendicular", "outward"]
        if self.direction:
            direction_lower = str(self.direction).lower()
            if all(phrase in direction_lower for phrase in expected_direction_phrases):
                logs.append("PASS: Direction description includes 'perpendicular' and 'outward'.")
            else:
                logs.append(f"WARN: Direction '{self.direction}' might not match expected description.")
                # Not a failure, just a warning
        else:
            logs.append("INFO: No direction string provided (this is OK per contract).")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input and verify expected trends.
        We'll perturb E (electric field) and see how B0_min and B0_max change.
        """
        logs = []
        success = True
        
        if None not in (self.m, self.q, self.E, self.B, self.R):
            if self.q != 0 and self.B != 0 and self.R != 0:
                # Original values
                E_original = self.E
                # Perturbation factor
                factor = 1.1  # 10% increase
                E_perturbed = E_original * factor
                
                # Recalculate B0_min and B0_max using formulas (numerical)
                B0_min_original = 4 * self.m * E_original / (7 * self.q * self.B * self.R)
                B0_max_original = 4 * self.m * E_original / (5 * self.q * self.B * self.R)
                B0_min_perturbed = 4 * self.m * E_perturbed / (7 * self.q * self.B * self.R)
                B0_max_perturbed = 4 * self.m * E_perturbed / (5 * self.q * self.B * self.R)
                
                # Check monotonicity: increasing E should increase both B0_min and B0_max
                if B0_min_perturbed > B0_min_original and B0_max_perturbed > B0_max_original:
                    logs.append("PASS: Increasing E increases both B0_min and B0_max (as expected).")
                else:
                    logs.append(f"FAIL: Monotonicity w.r.t. E violated. B0_min: {B0_min_original} -> {B0_min_perturbed}, B0_max: {B0_max_original} -> {B0_max_perturbed}")
                    success = False
                
                # Restore original values (not strictly needed, but good practice)
                self.E = E_original
            else:
                logs.append("SKIP: Division by zero in monotonicity calculation.")
                success = False
        else:
            logs.append("SKIP: Missing data for monotonicity test.")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} Test ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs