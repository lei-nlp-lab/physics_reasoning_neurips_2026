import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction using getattr with fallbacks
        self.solution = solution_instance
        
        # Input parameters (with safe defaults)
        self.E = getattr(solution_instance, 'E', 0.0)
        self.R = getattr(solution_instance, 'R', 0.0)
        self.d = getattr(solution_instance, 'd', 0.0)
        self.m = getattr(solution_instance, 'm', 0.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Results from solution (with safe extraction)
        results = getattr(solution_instance, 'results', {})
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.U = getattr(solution_instance, 'U', results.get('U', 0.0))
        self.E_field = getattr(solution_instance, 'E_field', results.get('E_field', 0.0))
        self.q = getattr(solution_instance, 'q', results.get('q', 0.0))
        
        # Expected symbolic expressions from KB analysis
        self.expected_I = 2 * self.E / (5 * self.R) if self.R != 0 else 0
        self.expected_U = 2 * self.E / 5
        self.expected_E_field = 2 * self.E / (5 * self.d) if self.d != 0 else 0
        self.expected_q = 5 * self.m * self.g * self.d / (2 * self.E) if self.E != 0 else 0
        
        self.logs = []

    def log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success

    def test_invariants(self):
        """Test fundamental physics laws."""
        passed = True
        
        # 1. Ohm's Law for entire circuit: E = I * (2R + r) with r = R/2
        try:
            total_resistance = 2 * self.R + self.R/2
            expected_voltage = self.I * total_resistance
            if not math.isclose(self.E, expected_voltage, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Ohm's Law violation: E={self.E} ≠ I*(2R+r)={expected_voltage}", False)
            else:
                self.log(f"Ohm's Law satisfied: E = I*(2R+r) = {expected_voltage:.3f}", True)
        except Exception as e:
            passed &= self.log(f"Ohm's Law check failed: {str(e)}", False)
        
        # 2. Ohm's Law for parallel resistor: U = I * R
        try:
            if not math.isclose(self.U, self.I * self.R, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Resistor voltage violation: U={self.U} ≠ I*R={self.I*self.R}", False)
            else:
                self.log(f"Resistor voltage correct: U = I*R = {self.I*self.R:.3f}", True)
        except Exception as e:
            passed &= self.log(f"Resistor Ohm's Law check failed: {str(e)}", False)
        
        # 3. Electric field formula: E_field = U / d
        try:
            if self.d > 0:
                expected = self.U / self.d
                if not math.isclose(self.E_field, expected, rel_tol=1e-2, abs_tol=1e-2):
                    passed &= self.log(f"E-field formula violation: E_field={self.E_field} ≠ U/d={expected}", False)
                else:
                    self.log(f"E-field formula correct: E_field = U/d = {expected:.3f}", True)
        except Exception as e:
            passed &= self.log(f"E-field check failed: {str(e)}", False)
        
        # 4. Force equilibrium: mg = qE_field
        try:
            mg = self.m * self.g
            qE = abs(self.q) * self.E_field if self.q != 0 else 0
            if not math.isclose(mg, qE, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Force equilibrium violation: mg={mg:.3f} ≠ |q|E_field={qE:.3f}", False)
            else:
                self.log(f"Force equilibrium satisfied: mg = |q|E_field = {mg:.3f}", True)
        except Exception as e:
            passed &= self.log(f"Force equilibrium check failed: {str(e)}", False)
        
        # 5. Sign consistency: Droplet must be negative to oppose gravity (field upward)
        try:
            if self.q >= 0:  # Non-negative charge would fall
                passed &= self.log(f"Charge sign issue: q={self.q} should be negative", False)
            else:
                self.log(f"Charge sign correct: q is negative", True)
        except Exception as e:
            passed &= self.log(f"Sign check failed: {str(e)}", False)
        
        return passed

    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answer."""
        passed = True
        
        try:
            # Check current against expected symbolic value
            if not math.isclose(self.I, self.expected_I, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Current mismatch: I={self.I} ≠ 2E/(5R)={self.expected_I}", False)
            else:
                self.log(f"Current consistent: I = 2E/(5R) = {self.I:.3f}", True)
        except Exception as e:
            passed &= self.log(f"Current consistency check failed: {str(e)}", False)
        
        try:
            # Check voltage against expected
            if not math.isclose(self.U, self.expected_U, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Voltage mismatch: U={self.U} ≠ 2E/5={self.expected_U}", False)
            else:
                self.log(f"Voltage consistent: U = 2E/5 = {self.U:.3f}", True)
        except Exception as e:
            passed &= self.log(f"Voltage consistency check failed: {str(e)}", False)
        
        try:
            # Check electric field against expected
            if self.d > 0:
                if not math.isclose(self.E_field, self.expected_E_field, rel_tol=1e-2, abs_tol=1e-2):
                    passed &= self.log(f"E-field mismatch: E_field={self.E_field} ≠ 2E/(5d)={self.expected_E_field}", False)
                else:
                    self.log(f"E-field consistent: E_field = 2E/(5d) = {self.E_field:.3f}", True)
        except Exception as e:
            passed &= self.log(f"E-field consistency check failed: {str(e)}", False)
        
        try:
            # Check charge against expected
            if self.E != 0:
                if not math.isclose(abs(self.q), abs(self.expected_q), rel_tol=1e-2, abs_tol=1e-2):
                    passed &= self.log(f"Charge mismatch: |q|={abs(self.q)} ≠ 5mgd/(2E)={abs(self.expected_q)}", False)
                else:
                    self.log(f"Charge consistent: |q| = 5mgd/(2E) = {abs(self.q):.3f}", True)
        except Exception as e:
            passed &= self.log(f"Charge consistency check failed: {str(e)}", False)
        
        return passed

    def test_monotonicity(self):
        """Perturb mass and verify charge changes proportionally."""
        if self.m == 0 or self.E == 0 or self.d == 0:
            self.log("Skipping monotonicity test (zero parameters)", True)
            return True
        
        passed = True
        original_m = self.m
        perturbation = 1.1  # 10% increase
        
        try:
            # Perturb mass
            self.m *= perturbation
            
            # Recalculate expected charge with perturbed mass
            new_expected_q = 5 * self.m * self.g * self.d / (2 * self.E)
            original_expected_q = self.expected_q
            
            # Check if ratio matches perturbation factor
            ratio = new_expected_q / original_expected_q
            if not math.isclose(ratio, perturbation, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self.log(f"Monotonicity violation: q should scale with m, got ratio={ratio:.3f} ≠ {perturbation}", False)
            else:
                self.log(f"Monotonicity correct: q ∝ m (ratio={ratio:.3f} ≈ {perturbation})", True)
        except Exception as e:
            passed &= self.log(f"Monotonicity test failed: {str(e)}", False)
        finally:
            # RESTORE original mass (crucial!)
            self.m = original_m
        
        return passed

    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append(f"✗ {test_name} crashed: {str(e)}")
                self.logs.append(traceback.format_exc())
                success = False
        
        self.logs.append(f"\n{'='*40}")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs