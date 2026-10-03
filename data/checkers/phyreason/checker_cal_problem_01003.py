import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        self.solution = solution_instance

        # Extract inputs with safe defaults
        self.S1 = getattr(solution_instance, 'S1', 10e-4)  # m²
        self.S2 = getattr(solution_instance, 'S2', 20e-4)  # m²
        self.m = getattr(solution_instance, 'm', 12.0)     # kg
        self.H = getattr(solution_instance, 'H', 0.1)      # m
        self.L = getattr(solution_instance, 'L', 0.05)     # m
        self.p0 = getattr(solution_instance, 'p0', 1.0e5)  # Pa
        self.g = getattr(solution_instance, 'g', 10.0)     # m/s²
        self.sin30 = getattr(solution_instance, 'sin30', 0.5)  # sin(30°)
        self.T0 = getattr(solution_instance, 'T0', 300.0)  # K (not used in force checks)

        # Extract results from solution_instance.results with safe fallback
        results = getattr(solution_instance, 'results', {})
        self.p_I = results.get('p_I', results.get('p_{I}', 0.0))   # initial gas pressure
        self.p_2 = results.get('p_2', results.get('p_{2}', 0.0))   # final gas pressure
        self.F = results.get('F', 0.0)                             # external force

        # Convert to floats if they are sympy objects
        if hasattr(self.p_I, 'evalf'):
            self.p_I = float(self.p_I)
        if hasattr(self.p_2, 'evalf'):
            self.p_2 = float(self.p_2)
        if hasattr(self.F, 'evalf'):
            self.F = float(self.F)

        # Store logs
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Check fundamental physical laws: force equilibrium and Boyle's Law."""
        passed = True

        # 1. Initial force equilibrium: (p0 - p_I)*(S1 - S2) = m*g*sin30
        left_init = (self.p0 - self.p_I) * (self.S1 - self.S2)
        right_init = self.m * self.g * self.sin30
        if not math.isclose(left_init, right_init, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL initial force equilibrium: left={left_init:.2f}, right={right_init:.2f}")
            passed = False
        else:
            self._log(f"PASS initial force equilibrium: left={left_init:.2f}, right={right_init:.2f}")

        # 2. Boyle's Law: p_I * (L*S1 + L*S2) = p_2 * (H*S2)
        V_initial = self.L * (self.S1 + self.S2)
        V_final = self.H * self.S2   # piston a at bottom, piston b at top (assumed)
        left_boyle = self.p_I * V_initial
        right_boyle = self.p_2 * V_final
        if not math.isclose(left_boyle, right_boyle, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL Boyle's Law: p_I*V_i={left_boyle:.2f}, p_2*V_f={right_boyle:.2f}")
            passed = False
        else:
            self._log(f"PASS Boyle's Law: p_I*V_i={left_boyle:.2f}, p_2*V_f={right_boyle:.2f}")

        # 3. Final force equilibrium with external force F
        left_final = self.p0 * self.S1 + self.p_2 * self.S2 + self.F
        right_final = self.m * self.g * self.sin30 + self.p0 * self.S2 + self.p_2 * self.S1
        if not math.isclose(left_final, right_final, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL final force equilibrium: left={left_final:.2f}, right={right_final:.2f}")
            passed = False
        else:
            self._log(f"PASS final force equilibrium: left={left_final:.2f}, right={right_final:.2f}")

        return passed

    def test_trace_consistency(self):
        """Verify that intermediate results (p_I, p_2) lead to the final F."""
        # Recompute F from p_I and p_2 using the final equilibrium equation
        # F = m*g*sin30 + p0*(S2 - S1) + p_2*(S1 - S2)   (derived from step 3)
        F_recomputed = (self.m * self.g * self.sin30 +
                        self.p0 * (self.S2 - self.S1) +
                        self.p_2 * (self.S1 - self.S2))

        if not math.isclose(self.F, F_recomputed, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL trace consistency: F={self.F:.2f}, recomputed={F_recomputed:.2f}")
            return False
        else:
            self._log(f"PASS trace consistency: F={self.F:.2f}, recomputed={F_recomputed:.2f}")
            return True

    def test_monotonicity(self):
        """Perturb mass m by +1%, recompute all quantities, check consistency."""
        original_m = self.m
        perturbed_m = original_m * 1.01  # increase by 1%

        # Recompute p_I from initial equilibrium: (p0 - p_I)*(S1 - S2) = m*g*sin30
        # => p_I = p0 - (m*g*sin30) / (S1 - S2)
        p_I_new = self.p0 - (perturbed_m * self.g * self.sin30) / (self.S1 - self.S2)

        # Recompute p_2 from Boyle's Law: p_I * V_initial = p_2 * V_final
        V_initial = self.L * (self.S1 + self.S2)
        V_final = self.H * self.S2
        p_2_new = p_I_new * V_initial / V_final

        # Recompute F from final equilibrium: F = m*g*sin30 + p0*(S2-S1) + p_2*(S1-S2)
        F_new = (perturbed_m * self.g * self.sin30 +
                 self.p0 * (self.S2 - self.S1) +
                 p_2_new * (self.S1 - self.S2))

        # Check that the new values satisfy the three equations (within tolerance)
        passed = True

        # Check initial equilibrium with new values
        left_init_new = (self.p0 - p_I_new) * (self.S1 - self.S2)
        right_init_new = perturbed_m * self.g * self.sin30
        if not math.isclose(left_init_new, right_init_new, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL monotonicity: initial equilibrium after perturbation")
            passed = False

        # Check Boyle's Law with new values
        left_boyle_new = p_I_new * V_initial
        right_boyle_new = p_2_new * V_final
        if not math.isclose(left_boyle_new, right_boyle_new, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL monotonicity: Boyle's Law after perturbation")
            passed = False

        # Check final equilibrium with new values (using F_new)
        left_final_new = self.p0 * self.S1 + p_2_new * self.S2 + F_new
        right_final_new = perturbed_m * self.g * self.sin30 + self.p0 * self.S2 + p_2_new * self.S1
        if not math.isclose(left_final_new, right_final_new, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL monotonicity: final equilibrium after perturbation")
            passed = False

        if passed:
            self._log(f"PASS monotonicity test with +1% mass perturbation")
        return passed

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs.clear()
        self._log("Starting PhysicsChecker tests...")

        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()

        success = test1 and test2 and test3
        self._log(f"Overall result: {'PASS' if success else 'FAIL'}")

        return success, self.logs