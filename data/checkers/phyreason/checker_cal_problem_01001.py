import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results safely - ONLY from results dict
        self.T1 = self.results.get('T1', 300.0)
        self.T2 = self.results.get('T2', 282.0)
        self.p2 = self.results.get('p2', None)
        self.delta_V = self.results.get('delta_V', None)
        self.n = self.results.get('n', None)
        
        # Extract inputs from results or use defaults
        self.S = self.results.get('S', 3.0e-4)  # m^2
        self.l = self.results.get('l', 1.6e-1)  # m
        self.p_target = self.results.get('p', 1.3e5)  # Pa
        self.V2 = self.results.get('V2', 1.2e-3)  # m^3, initial volume
        self.p1 = self.results.get('p1', 1.0e5)  # Pa
        self.p0 = self.results.get('p0', 1.0e5)  # Pa
        
        # Calculate derived quantities
        self.V_pump = self.S * self.l  # Pump stroke volume

    def test_invariants(self):
        logs = []
        try:
            # 1. Ideal Gas Law (isochoric step): p1/T1 = p2/T2
            if self.p2 is not None:
                lhs = self.p1 / self.T1
                rhs = self.p2 / self.T2
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Isochoric process obeys p1/T1 = p2/T2")
                else:
                    logs.append(f"FAIL: p1/T1 = {lhs}, p2/T2 = {rhs}")
            else:
                logs.append("SKIP: p2 not available for isochoric check")

            # 2. Boyle's Law (conversion of initial gas): p2 * V2 = p_target * delta_V
            if self.p2 is not None and self.delta_V is not None:
                lhs = self.p2 * self.V2
                rhs = self.p_target * self.delta_V
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Boyle's law holds for initial gas conversion")
                else:
                    logs.append(f"FAIL: p2*V2 = {lhs}, p_target*delta_V = {rhs}")
            else:
                logs.append("SKIP: Missing p2 or delta_V for Boyle's law")

            # 3. Correct pumping equation: p_target * V2 = p2 * V2 + n * p0 * V_pump
            if self.n is not None and self.p2 is not None:
                lhs = self.p_target * self.V2
                rhs = self.p2 * self.V2 + self.n * self.p0 * self.V_pump
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Pumping equation holds")
                else:
                    logs.append(f"FAIL: p_target*V2 = {lhs}, p2*V2 + n*p0*V_pump = {rhs}")
            else:
                logs.append("SKIP: Missing n or p2 for pumping equation")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Recalculate key steps and compare with stored results
            # Step 5: p2 calculation
            p2_calc = self.p1 * self.T2 / self.T1
            if self.p2 is not None and math.isclose(p2_calc, self.p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: p2 calculation correct")
            else:
                logs.append(f"FAIL: p2 calc {p2_calc} vs stored {self.p2}")

            # Step 10: ΔV calculation
            if self.p2 is not None:
                delta_V_calc = self.p2 * self.V2 / self.p_target
                if self.delta_V is not None and math.isclose(delta_V_calc, self.delta_V, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: ΔV calculation correct")
                else:
                    logs.append(f"FAIL: ΔV calc {delta_V_calc} vs stored {self.delta_V}")

            # Step 12: n calculation (using derived formula from KB)
            if self.p2 is not None:
                # Check for division by zero
                if abs(self.p0 * self.V_pump) < 1e-15:
                    logs.append("ERROR: Division by zero in n calculation")
                else:
                    n_calc = self.V2 * (self.p_target - self.p2) / (self.p0 * self.V_pump)
                    if self.n is not None and math.isclose(n_calc, self.n, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append("PASS: n calculation correct")
                    else:
                        logs.append(f"FAIL: n calc {n_calc} vs stored {self.n}")
            else:
                logs.append("SKIP: Missing p2 for n calculation")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        original_p1 = self.p1
        try:
            # Perturb initial pressure p1 upward by 5%
            self.p1 *= 1.05
            # Recalculate p2 and n
            p2_perturbed = self.p1 * self.T2 / self.T1
            # Check for division by zero
            if abs(self.p0 * self.V_pump) < 1e-15:
                logs.append("ERROR: Division by zero in monotonicity test")
            else:
                n_perturbed = self.V2 * (self.p_target - p2_perturbed) / (self.p0 * self.V_pump)
                # Check monotonicity: higher p1 -> higher p2 -> lower n (since p_target fixed)
                if self.p2 is not None and self.n is not None:
                    if p2_perturbed > self.p2 and n_perturbed < self.n:
                        logs.append("PASS: Monotonicity holds for p1 increase")
                    else:
                        logs.append(f"FAIL: p1↑ {p2_perturbed} vs {self.p2}, n {n_perturbed} vs {self.n}")
                else:
                    logs.append("SKIP: Missing p2 or n for monotonicity")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original p1
            self.p1 = original_p1
        return logs

    def run_all_tests(self):
        logs = []
        try:
            logs.append("=== PhysicsChecker Start ===")
            logs.extend(self.test_invariants())
            logs.extend(self.test_trace_consistency())
            logs.extend(self.test_monotonicity())
            logs.append("=== PhysicsChecker End ===")
            # Determine success: no failures or errors
            success = not any("FAIL" in log or "ERROR" in log for log in logs)
            return success, logs
        except Exception as e:
            logs.append(f"CRITICAL ERROR: {traceback.format_exc()}")
            return False, logs