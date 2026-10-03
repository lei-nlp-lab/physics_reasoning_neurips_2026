import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Get results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with sensible defaults
        self.m = self._get_value('m', 1.0)
        self.g = self._get_value('g', 9.8)
        self.d = self._get_value('d', 0.1)
        self.R1 = self._get_value('R1', 100.0)
        self.R2 = self._get_value('R2', 200.0)
        self.q = self._get_value('q', 1e-6)
        self.v = self._get_value('v', 5.0)
        self.E0 = self._get_value('E_0', 1470.0)
        
        # Extract results
        self.r = self._get_value('r', 0.2)
        self.U2 = self._get_value('U_2', 980.0)
        self.B = self._get_value('B', 0.025)
        self.E_prime = self._get_value('E_prime', 4900000.0)
        self.qE = self._get_value('qE', 9.8)
        
        # Store for restoration in monotonicity test
        self._original_m = self.m
        
    def _get_value(self, key, default):
        """Get value from results dict with fallback."""
        value = self.results.get(key, default)
        # Ensure it's a float
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Check qE = mg (vertical force balance)
            if not math.isclose(self.qE, self.m * self.g, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: qE ({self.qE}) ≠ mg ({self.m*self.g})")
                success = False
            else:
                logs.append("PASS: Vertical force balance qE = mg")
            
            # 2. Check centripetal force: qvB = mv²/r
            if self.r > 0 and self.B > 0:
                lhs = self.q * self.v * self.B
                rhs = self.m * self.v**2 / self.r
                if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: qvB ({lhs}) ≠ mv²/r ({rhs})")
                    success = False
                else:
                    logs.append("PASS: Magnetic force provides centripetal force")
            
            # 3. Check geometry: r = 2d
            if not math.isclose(self.r, 2*self.d, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Radius r ({self.r}) ≠ 2d ({2*self.d})")
                success = False
            else:
                logs.append("PASS: Geometric constraint r = 2d")
            
            # 4. Check Ohm's Law consistency: U2 = (E0 * R2)/(R1 + R2)
            if self.R1 + self.R2 > 0:
                expected_U2 = (self.E0 * self.R2) / (self.R1 + self.R2)
                if not math.isclose(self.U2, expected_U2, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: U2 ({self.U2}) ≠ (E0*R2)/(R1+R2) ({expected_U2})")
                    success = False
                else:
                    logs.append("PASS: Ohm's Law for voltage divider")
            
            # 5. Check E-field relations
            # From step2: U2 = E*d, so E = U2/d
            if self.d > 0:
                E_from_U2 = self.U2 / self.d
                # From step1: E = mg/q
                E_from_force = self.m * self.g / self.q if self.q != 0 else 0
                if not math.isclose(E_from_U2, E_from_force, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: E from U2/d ({E_from_U2}) ≠ E from mg/q ({E_from_force})")
                    success = False
                else:
                    logs.append("PASS: Electric field consistency")
            
            # 6. Check E' = mg/(2q)
            if self.q != 0:
                expected_E_prime = self.m * self.g / (2 * self.q)
                if not math.isclose(self.E_prime, expected_E_prime, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: E' ({self.E_prime}) ≠ mg/(2q) ({expected_E_prime})")
                    success = False
                else:
                    logs.append("PASS: Minimum electric field E' = mg/(2q)")
        
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Expected formulas from STEPS ANALYSIS (using numeric calculations)
            E0_expected = self.m * self.g * self.d * (self.R1 + self.R2) / (self.q * self.R2)
            B_expected = self.m * self.v / (2 * self.d * self.q)
            E_prime_expected = self.m * self.g / (2 * self.q)
            
            # Compare with extracted numerical values
            if not math.isclose(self.E0, E0_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: E0 ({self.E0}) doesn't match expected formula ({E0_expected})")
                success = False
            else:
                logs.append("PASS: E0 matches analytic expression")
            
            if not math.isclose(self.B, B_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: B ({self.B}) doesn't match expected formula ({B_expected})")
                success = False
            else:
                logs.append("PASS: B matches analytic expression")
            
            if not math.isclose(self.E_prime, E_prime_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: E' ({self.E_prime}) doesn't match expected formula ({E_prime_expected})")
                success = False
            else:
                logs.append("PASS: E' matches analytic expression")
            
            # Check that derived r gives correct circular motion period
            if self.r > 0 and self.v > 0:
                # Centripetal acceleration: a_c = v²/r
                a_c = self.v**2 / self.r
                # Magnetic force: F_B = qvB
                F_B = self.q * self.v * self.B
                # Should equal centripetal force
                if not math.isclose(F_B, self.m * a_c, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: F_B ({F_B}) ≠ m*a_c ({self.m*a_c}) in trace check")
                    success = False
                else:
                    logs.append("PASS: Circular motion dynamics consistent")
        
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """Perturb mass m and check output trends."""
        logs = []
        success = True
        
        try:
            # Store original
            original_m = self.m
            
            # Perturb by 5% increase
            perturbed_m = original_m * 1.05
            
            # Recalculate expected values using formulas
            E0_new = (perturbed_m * self.g * self.d * (self.R1 + self.R2)) / (self.q * self.R2)
            B_new = (perturbed_m * self.v) / (2 * self.d * self.q)
            E_prime_new = (perturbed_m * self.g) / (2 * self.q)
            
            # Check monotonic relationships
            if not (E0_new > self.E0 * 1.03):  # Should increase by ~5%
                logs.append(f"WARN: E0 not monotonic with m. New: {E0_new}, Original: {self.E0}")
                success = False
            else:
                logs.append("PASS: E0 increases with mass")
            
            if not (B_new > self.B * 1.03):
                logs.append(f"WARN: B not monotonic with m. New: {B_new}, Original: {self.B}")
                success = False
            else:
                logs.append("PASS: B increases with mass")
            
            if not (E_prime_new > self.E_prime * 1.03):
                logs.append(f"WARN: E' not monotonic with m. New: {E_prime_new}, Original: {self.E_prime}")
                success = False
            else:
                logs.append("PASS: E' increases with mass")
            
            # Check linear proportionality
            ratio_m = 1.05  # perturbation ratio
            if not math.isclose(E0_new/self.E0, ratio_m, rel_tol=0.02):
                logs.append(f"WARN: E0 not linearly proportional to m (ratio: {E0_new/self.E0})")
            if not math.isclose(B_new/self.B, ratio_m, rel_tol=0.02):
                logs.append(f"WARN: B not linearly proportional to m (ratio: {B_new/self.B})")
            if not math.isclose(E_prime_new/self.E_prime, ratio_m, rel_tol=0.02):
                logs.append(f"WARN: E' not linearly proportional to m (ratio: {E_prime_new/self.E_prime})")
        
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
        
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs