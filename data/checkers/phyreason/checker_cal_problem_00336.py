import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Uses getattr with fallback to .results dictionary to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Input parameters
        self.E = getattr(solution_instance, 'E', 40.0)  # N/C
        self.m = getattr(solution_instance, 'm', 2e-9)  # kg
        self.q = getattr(solution_instance, 'q', -2e-9)  # C
        self.W_AB = getattr(solution_instance, 'W_AB', 1.5e-7)  # J (work by field)
        
        # Results from solution
        self.U_AB = getattr(solution_instance, 'U_AB', 
                           getattr(solution_instance, 'results', {}).get('U_AB', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', 
                          getattr(solution_instance, 'results', {}).get('v_B', 0.0))
        
        # Expected symbolic/float values for comparison
        self.target_U_AB = -75.0  # V
        self.target_v_B = 5.0 * math.sqrt(10)  # m/s (~15.811388)
        
        # Logs storage
        self.logs = []
    
    def _log(self, message, success):
        """Helper to append test results with consistent formatting."""
        self.logs.append((success, message))
    
    def test_invariants(self):
        """Verify fundamental physics laws: energy conservation and potential-work relation."""
        success = True
        
        # 1. Check electric potential difference: U_AB = W_AB / q (from KB definition)
        # Note: W_AB is work done BY the electric field on the charge.
        # For negative q, positive work implies U_AB negative (as in target).
        try:
            U_calculated = self.W_AB / self.q
            if not math.isclose(self.U_AB, U_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Potential difference mismatch: U_AB = {self.U_AB:.3f}, expected {U_calculated:.3f}", False)
                success = False
            else:
                self._log(f"✓ Potential difference consistent: U_AB = {self.U_AB:.3f} V", True)
        except Exception as e:
            self._log(f"Error in potential difference check: {e}", False)
            success = False
        
        # 2. Work-energy theorem: net work = change in kinetic energy.
        # Since only electric force does work (from problem statement), W_AB = ΔK.
        # We don't have v_A, but we can check that ΔK is positive (since W_AB > 0).
        # Use symbolic check with v_A as unknown.
        try:
            v_A = sp.symbols('v_A', real=True)
            delta_K = 0.5 * self.m * (self.v_B**2 - v_A**2)
            # Net work should equal delta_K for some real v_A.
            # Solve for v_A^2: v_A^2 = v_B^2 - (2 * W_AB)/m
            v_A_sq = self.v_B**2 - (2 * self.W_AB) / self.m
            if v_A_sq >= 0:
                self._log(f"✓ Work-energy theorem satisfied with v_A = ±{math.sqrt(v_A_sq):.3f} m/s", True)
            else:
                self._log(f"Work-energy theorem violated: v_A^2 = {v_A_sq:.3f} (negative)", False)
                success = False
        except Exception as e:
            self._log(f"Error in work-energy check: {e}", False)
            success = False
        
        # 3. Check relationship between electric field and potential difference.
        # For uniform field along displacement: |ΔV| = E * |d|, but sign depends on direction.
        # Since work by field is positive and q is negative, the particle moves opposite to field direction.
        # Therefore, ΔV = V_B - V_A should be negative, and magnitude |U_AB| = E * d.
        # We can compute d from work: W_AB = |q| * E * d (since force magnitude is |q|E, and work positive).
        try:
            d = self.W_AB / (abs(self.q) * self.E)  # distance from work formula
            U_mag_from_E = self.E * d  # magnitude of potential difference
            if not math.isclose(abs(self.U_AB), U_mag_from_E, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Field-potential consistency issue: |U_AB|={abs(self.U_AB):.3f}, E*d={U_mag_from_E:.3f}", False)
                success = False
            else:
                self._log(f"✓ Field-potential magnitude consistent: d = {d:.3f} m", True)
        except Exception as e:
            self._log(f"Error in field-potential check: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate steps from the STEPS ANALYSIS with final results.
        """
        success = True
        
        # Step 1: U_AB = W_AB / q
        try:
            U_step1 = self.W_AB / self.q
            if not math.isclose(self.U_AB, U_step1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 mismatch: U_AB = {self.U_AB:.3f}, step1 calc = {U_step1:.3f}", False)
                success = False
            else:
                self._log("✓ Step 1 (U_AB = W_AB/q) consistent with final result", True)
        except Exception as e:
            self._log(f"Error in step 1 consistency: {e}", False)
            success = False
        
        # Step 2: v_B = sqrt(2*W/m + v_A^2). We lack v_A, but can check if v_B matches target.
        # Since v_A is unknown, we check if using v_A=0 gives at least the correct order.
        # Alternatively, verify that v_B is within physical bounds.
        try:
            # Lower bound: v_B >= sqrt(2*W/m) if v_A=0.
            v_min = math.sqrt(2 * self.W_AB / self.m)
            if self.v_B < v_min - 1e-2:  # Allow small tolerance
                self._log(f"v_B {self.v_B:.3f} is less than minimum {v_min:.3f} (v_A=0)", False)
                success = False
            else:
                self._log(f"✓ v_B = {self.v_B:.3f} m/s is physically plausible (≥ {v_min:.3f})", True)
            
            # Compare with target value from steps analysis.
            if not math.isclose(self.v_B, self.target_v_B, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_B {self.v_B:.3f} does not match target {self.target_v_B:.3f}", False)
                success = False
            else:
                self._log(f"✓ v_B matches target value {self.target_v_B:.3f} m/s", True)
        except Exception as e:
            self._log(f"Error in step 2 consistency: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (e.g., charge magnitude) and verify the expected trend.
        Restore original values after test.
        """
        original_q = self.q
        success = True
        
        try:
            # Increase magnitude of negative charge (more negative).
            perturbed_q = self.q * 1.1  # 10% more negative
            # Recalculate U_AB using formula: U = W / q (since W fixed).
            U_perturbed = self.W_AB / perturbed_q
            
            # Since q is negative, making it more negative (larger magnitude) makes U_AB less negative.
            # For example: original q = -2e-9 → U = -75; perturbed q = -2.2e-9 → U ≈ -68.18.
            # So U should increase (become less negative).
            if self.U_AB < U_perturbed:
                self._log(f"✓ Monotonicity: U_AB increases when |q| increases (as expected)", True)
            else:
                self._log(f"Monotonicity violated: U_AB did not increase with |q|", False)
                success = False
        except Exception as e:
            self._log(f"Error in monotonicity test: {e}", False)
            success = False
        finally:
            self.q = original_q  # Restore original value
        
        return success
    
    def run_all_tests(self):
        """
        Execute all checks and return (overall_success, logs).
        """
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...", None)
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs