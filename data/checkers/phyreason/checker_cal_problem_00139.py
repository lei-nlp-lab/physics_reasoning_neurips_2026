import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract all inputs and results from results dictionary."""
        self.solution = solution_instance
        results = solution_instance.results
        
        # Extract constants from results (they should be stored by the solution)
        self.e = results.get('e', 1.0)
        self.m = results.get('m', 1.0)
        self.v0 = results.get('v0', 5.0)
        self.E = results.get('E', 1.0)
        self.d = results.get('d', 1.0)
        self.L = results.get('L', 1.0)
        self.n = results.get('n', 1.0)
        
        # Guard against zero values
        if self.e == 0:
            self.e = 1.0
        if self.m == 0:
            self.m = 1.0
        if self.L == 0:
            self.L = 1.0
        
        # Results: use exact keys from contract
        self.B0 = results.get('B_0', 0.0)
        self.F_prime = results.get("F'", 0.0)
        self.costheta = results.get('costheta', 0.0)
        self.R3 = results.get('R_3', 0.0)
        self.vs = results.get('v_s', results.get('vs', 0.0))
        
        # Log storage
        self.logs = []

    def test_invariants(self):
        """Test fundamental physics laws with safe division."""
        try:
            # 1. Work-energy theorem for acceleration region (S to O)
            q = 2 * self.e  # Ion charge is 2e
            W = q * self.E * self.d
            delta_K = 0.5 * self.m * (self.v0**2 - self.vs**2)
            if not math.isclose(W, delta_K, rel_tol=1e-2, abs_tol=1e-12):
                self.logs.append(f"Work-energy violation: W={W:.3e}, ΔK={delta_K:.3e}")

            # 2. Lorentz force as centripetal force for circular motion
            # For the given B0, the effective field is sqrt(2)*B0
            if self.B0 != 0 and self.v0 != 0 and self.e != 0:
                B_eff = math.sqrt(2) * self.B0
                q = 2 * self.e
                r_theory = self.m * self.v0 / (q * B_eff)
                if not math.isclose(self.R3, r_theory, rel_tol=1e-2, abs_tol=1e-12):
                    self.logs.append(f"Lorentz force radius mismatch: R3={self.R3:.3e}, theory={r_theory:.3e}")
            else:
                self.logs.append("Skipping Lorentz force check due to zero B0, v0, or e")

            # 3. Momentum-impulse theorem for force calculation
            F_expected = self.n * self.m * self.v0 * self.costheta
            if not math.isclose(self.F_prime, F_expected, rel_tol=1e-2, abs_tol=1e-12):
                self.logs.append(f"Momentum-impulse mismatch: F'={self.F_prime:.3e}, expected={F_expected:.3e}")

        except Exception as e:
            self.logs.append(f"Invariant test error: {traceback.format_exc()}")

    def test_trace_consistency(self):
        """Cross-check intermediate results with safe division."""
        try:
            # Step 9: R3 = (5/4)L
            expected_R3 = 1.25 * self.L
            if not math.isclose(self.R3, expected_R3, rel_tol=1e-2, abs_tol=1e-12):
                self.logs.append(f"R3 inconsistent with L: R3={self.R3:.3e}, 1.25*L={expected_R3:.3e}")

            # Step 10: cosθ = 3/5 = 0.6
            if not math.isclose(self.costheta, 0.6, rel_tol=1e-2, abs_tol=1e-12):
                self.logs.append(f"cosθ inconsistent: {self.costheta:.3e} vs 0.6")

            # Step 4 and 7: B0 range: 0 to m*v0/(3*e*L)
            if self.e != 0 and self.L != 0:
                B0_max = (self.m * self.v0) / (3 * self.e * self.L)
                # The B0 in results is the GIVEN B0 for Q3, not the range maximum
                # So we shouldn't check it's in the range - that's for a different B0
                # Instead, verify the given B0 equals (√2*m*v0)/(5*e*L)
                B0_expected = (math.sqrt(2) * self.m * self.v0) / (5 * self.e * self.L)
                if not math.isclose(self.B0, B0_expected, rel_tol=1e-2, abs_tol=1e-12):
                    self.logs.append(f"B0 doesn't match given formula: {self.B0:.3e} vs {B0_expected:.3e}")
            else:
                self.logs.append("Skipping B0 check due to zero e or L")

            # Step 12: F' = (3/5)*n*m*v0 = 0.6*n*m*v0
            F_expected = 0.6 * self.n * self.m * self.v0
            if not math.isclose(self.F_prime, F_expected, rel_tol=1e-2, abs_tol=1e-12):
                self.logs.append(f"F' inconsistent with formula: {self.F_prime:.3e} vs {F_expected:.3e}")

        except Exception as e:
            self.logs.append(f"Trace consistency error: {traceback.format_exc()}")

    def test_monotonicity(self):
        """Perturb mass m and check trends with safe division."""
        original_m = self.m
        original_vs = self.vs
        original_B0 = self.B0
        original_F_prime = self.F_prime
        
        try:
            # Increase mass by 10%
            self.m *= 1.1

            # Recalculate expected values using formulas
            q = 2 * self.e
            # v_s^2 = v0^2 - (4eE d)/m = v0^2 - (q*E*d)/m
            if self.m != 0:
                vs2 = self.v0**2 - (q * self.E * self.d) / self.m
                vs_new = math.sqrt(max(vs2, 0))
            else:
                vs_new = 0

            # B0_given = (√2*m*v0)/(5*e*L)
            if self.e != 0 and self.L != 0:
                B0_given_new = (math.sqrt(2) * self.m * self.v0) / (5 * self.e * self.L)
            else:
                B0_given_new = 0

            # F' = 0.6*n*m*v0
            F_prime_new = 0.6 * self.n * self.m * self.v0

            # CORRECTED: v_s should INCREASE with increased m
            # From v_s^2 = v0^2 - (4eEd)/m, as m increases, (4eEd)/m decreases, so v_s^2 increases
            if not (vs_new > original_vs or math.isclose(vs_new, original_vs, rel_tol=1e-2)):
                self.logs.append(f"Monotonicity: v_s should increase with increased m (old {original_vs:.3e}, new {vs_new:.3e})")
            
            if not (B0_given_new > original_B0 or math.isclose(B0_given_new, original_B0, rel_tol=1e-2)):
                self.logs.append(f"Monotonicity: B0 should increase with m (old {original_B0:.3e}, new {B0_given_new:.3e})")
            
            if not (F_prime_new > original_F_prime or math.isclose(F_prime_new, original_F_prime, rel_tol=1e-2)):
                self.logs.append(f"Monotonicity: F' should increase with m (old {original_F_prime:.3e}, new {F_prime_new:.3e})")

        except Exception as e:
            self.logs.append(f"Monotonicity test error: {traceback.format_exc()}")
        finally:
            # Restore original values
            self.m = original_m

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs.clear()
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        success = len(self.logs) == 0
        return success, self.logs.copy()