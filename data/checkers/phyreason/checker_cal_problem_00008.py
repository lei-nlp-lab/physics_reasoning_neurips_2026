import sympy as sp
import math
import traceback
import random

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses getattr to avoid AttributeError in the sandbox.
        """
        # Inputs (symbolic or numeric)
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        self.omega1 = getattr(solution_instance, 'omega1', solution_instance.results.get('omega1', None))
        self.r1 = getattr(solution_instance, 'r1', solution_instance.results.get('r1', None))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', None))
        self.beta = getattr(solution_instance, 'beta', solution_instance.results.get('beta', None))
        self.r2 = getattr(solution_instance, 'r2', solution_instance.results.get('r2', None))

        # Results (symbolic or numeric)
        self.Tcosalpha = getattr(solution_instance, 'Tcosalpha', solution_instance.results.get('Tcosalpha', None))
        self.Tsinalpha = getattr(solution_instance, 'Tsinalpha', solution_instance.results.get('Tsinalpha', None))
        self.T1 = getattr(solution_instance, 'T1', solution_instance.results.get('T1', None))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', None))
        self.T2 = getattr(solution_instance, 'T2', solution_instance.results.get('T2', None))
        self.Fn = getattr(solution_instance, 'Fn', solution_instance.results.get('Fn', None))
        self.T1cosbeta = getattr(solution_instance, 'T1cosbeta', solution_instance.results.get('T1cosbeta', None))
        self.tanalpha = getattr(solution_instance, 'tanalpha', solution_instance.results.get('tanalpha', None))
        self.omega2 = getattr(solution_instance, 'omega2', solution_instance.results.get('omega2', None))

        # Store the instance for reference
        self.solution = solution_instance
        self.logs = []

    def _log(self, msg, success):
        """Append a test result to logs."""
        self.logs.append((msg, success))

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Robust numerical comparison for floats or sympy expressions.
        If inputs are symbolic, substitute random values and compare.
        """
        # If both are numbers, use math.isclose
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

        # If one or both are sympy expressions, try to convert to float if possible
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # Symbolic case: generate random values for all symbols and evaluate
            symbols = set()
            for expr in (a, b):
                if isinstance(expr, sp.Expr):
                    symbols.update(expr.free_symbols)
            if not symbols:
                # No symbols, should have been caught above
                return False

            # Create a random substitution dictionary
            subs = {}
            for sym in symbols:
                # Avoid zero to prevent division by zero
                subs[sym] = random.uniform(0.5, 5.0)

            try:
                a_sub = float(a.subs(subs)) if isinstance(a, sp.Expr) else a
                b_sub = float(b.subs(subs)) if isinstance(b, sp.Expr) else b
                return math.isclose(a_sub, b_sub, rel_tol=rel_tol, abs_tol=abs_tol)
            except Exception:
                return False

    def test_invariants(self):
        """Check physical laws based on the KB and steps analysis."""
        success = True

        # Part 1: First configuration (steps 1-3)
        if all(v is not None for v in [self.m, self.mu, self.g, self.omega1, self.r1,
                                        self.Tcosalpha, self.Tsinalpha, self.tanalpha]):
            # 1. Tangential equilibrium: μ m g = T sinα
            expected_Tsinalpha = self.mu * self.m * self.g
            if not self._is_close(self.Tsinalpha, expected_Tsinalpha):
                self._log(f"Tangential equilibrium failed: T sinα = {self.Tsinalpha}, expected μ m g = {expected_Tsinalpha}", False)
                success = False
            else:
                self._log("Tangential equilibrium holds.", True)

            # 2. Radial equilibrium: T cosα = m ω₁² r₁
            expected_Tcosalpha = self.m * self.omega1**2 * self.r1
            if not self._is_close(self.Tcosalpha, expected_Tcosalpha):
                self._log(f"Radial equilibrium failed: T cosα = {self.Tcosalpha}, expected m ω₁² r₁ = {expected_Tcosalpha}", False)
                success = False
            else:
                self._log("Radial equilibrium holds.", True)

            # 3. Definition of tanα: tanα = T sinα / T cosα
            if not self._is_close(self.tanalpha, self.Tsinalpha / self.Tcosalpha):
                self._log(f"tanα definition failed: tanα = {self.tanalpha}, T sinα / T cosα = {self.Tsinalpha / self.Tcosalpha}", False)
                success = False
            else:
                self._log("tanα definition holds.", True)

            # 4. Formula for tanα: tanα = μ g / (ω₁² r₁)
            expected_tanalpha = self.mu * self.g / (self.omega1**2 * self.r1)
            if not self._is_close(self.tanalpha, expected_tanalpha):
                self._log(f"tanα formula failed: tanα = {self.tanalpha}, expected μ g/(ω₁² r₁) = {expected_tanalpha}", False)
                success = False
            else:
                self._log("tanα formula holds.", True)
        else:
            self._log("Skipping part 1 checks due to missing data.", True)  # Not a failure, just skip

        # Part 2: Second configuration (steps 4-9)
        if all(v is not None for v in [self.m, self.mu, self.g, self.theta, self.beta, self.r2,
                                        self.T1, self.f, self.T2, self.Fn, self.T1cosbeta, self.omega2]):
            # 5. T₁ and T₂ relationship: T₁ / T₂ = tanθ
            if not self._is_close(self.T1 / self.T2, sp.tan(self.theta)):
                self._log(f"T₁/T₂ = tanθ failed: T₁/T₂ = {self.T1/self.T2}, tanθ = {sp.tan(self.theta)}", False)
                success = False
            else:
                self._log("T₁/T₂ = tanθ holds.", True)

            # 6. Centripetal force: T₁ cosβ = m ω₂² r₂
            expected_T1cosbeta = self.m * self.omega2**2 * self.r2
            if not self._is_close(self.T1cosbeta, expected_T1cosbeta):
                self._log(f"Centripetal force failed: T₁ cosβ = {self.T1cosbeta}, expected m ω₂² r₂ = {expected_T1cosbeta}", False)
                success = False
            else:
                self._log("Centripetal force holds.", True)

            # 7. Friction force: f = μ Fₙ
            if not self._is_close(self.f, self.mu * self.Fn):
                self._log(f"Friction relation failed: f = {self.f}, μ Fₙ = {self.mu * self.Fn}", False)
                success = False
            else:
                self._log("Friction relation holds.", True)

            # 8. Tangential force balance: f = T₁ sinβ
            if not self._is_close(self.f, self.T1 * sp.sin(self.beta)):
                self._log(f"Tangential force balance failed: f = {self.f}, T₁ sinβ = {self.T1 * sp.sin(self.beta)}", False)
                success = False
            else:
                self._log("Tangential force balance holds.", True)

            # 9. Vertical equilibrium: Fₙ + T₂ = m g
            if not self._is_close(self.Fn + self.T2, self.m * self.g):
                self._log(f"Vertical equilibrium failed: Fₙ + T₂ = {self.Fn + self.T2}, m g = {self.m * self.g}", False)
                success = False
            else:
                self._log("Vertical equilibrium holds.", True)

            # 10. Formula for ω₂: ω₂ = sqrt(μ g sinθ cosβ / ((sinθ sinβ + μ cosθ) r₂))
            numerator = self.mu * self.g * sp.sin(self.theta) * sp.cos(self.beta)
            denominator = (sp.sin(self.theta) * sp.sin(self.beta) + self.mu * sp.cos(self.theta)) * self.r2
            expected_omega2 = sp.sqrt(numerator / denominator)
            if not self._is_close(self.omega2, expected_omega2):
                self._log(f"ω₂ formula failed: ω₂ = {self.omega2}, expected sqrt(...) = {expected_omega2}", False)
                success = False
            else:
                self._log("ω₂ formula holds.", True)
        else:
            self._log("Skipping part 2 checks due to missing data.", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True

        # Part 1: tanα should equal Tsinalpha / Tcosalpha
        if all(v is not None for v in [self.tanalpha, self.Tsinalpha, self.Tcosalpha]):
            if not self._is_close(self.tanalpha, self.Tsinalpha / self.Tcosalpha):
                self._log(f"tanα inconsistent: tanα = {self.tanalpha}, T sinα / T cosα = {self.Tsinalpha / self.Tcosalpha}", False)
                success = False
            else:
                self._log("tanα consistent with T components.", True)

        # Part 2: Check that T1cosbeta / (m * r2) equals omega2**2
        if all(v is not None for v in [self.T1cosbeta, self.m, self.r2, self.omega2]):
            if not self._is_close(self.T1cosbeta, self.m * self.r2 * self.omega2**2):
                self._log(f"ω₂ inconsistent: T₁ cosβ = {self.T1cosbeta}, m r₂ ω₂² = {self.m * self.r2 * self.omega2**2}", False)
                success = False
            else:
                self._log("ω₂ consistent with centripetal force.", True)

        # Check that f from step 7 equals μ (mg - T2) from step 8
        if all(v is not None for v in [self.f, self.mu, self.m, self.g, self.T2]):
            if not self._is_close(self.f, self.mu * (self.m * self.g - self.T2)):
                self._log(f"Friction inconsistent: f = {self.f}, μ(mg - T₂) = {self.mu * (self.m * self.g - self.T2)}", False)
                success = False
            else:
                self._log("Friction consistent with vertical equilibrium.", True)

        return success

    def test_monotonicity(self):
        """
        Perturb an input (e.g., μ) and check that outputs change as expected.
        Restores the original value after testing.
        """
        success = True

        # Only run if we have numerical values or can evaluate symbols
        try:
            # Store original mu
            mu_orig = self.mu
            if mu_orig is None:
                self._log("Skipping monotonicity test: μ not provided.", True)
                return True

            # Perturb: increase μ by 10%
            if isinstance(mu_orig, (int, float)):
                mu_pert = mu_orig * 1.1
            elif isinstance(mu_orig, sp.Symbol):
                # If symbolic, skip numeric test
                self._log("Skipping monotonicity test: μ is symbolic.", True)
                return True
            else:
                # Try to convert to float
                mu_pert = float(mu_orig) * 1.1

            # Compute expected tanα and ω₂ with perturbed μ
            # tanα = μ g / (ω₁² r₁) -> increases with μ
            if all(v is not None for v in [self.g, self.omega1, self.r1]):
                tanalpha_orig = self.tanalpha
                tanalpha_pert = mu_pert * self.g / (self.omega1**2 * self.r1)
                if not tanalpha_pert > tanalpha_orig:
                    self._log(f"tanα monotonicity failed: tanα(μ={mu_pert}) = {tanalpha_pert} not > tanα(μ={mu_orig}) = {tanalpha_orig}", False)
                    success = False
                else:
                    self._log("tanα increases with μ as expected.", True)

            # ω₂ = sqrt(μ g sinθ cosβ / ((sinθ sinβ + μ cosθ) r₂))
            # We check that ω₂ increases with μ? Not necessarily monotonic; skip if derivative unclear.
            # Instead, we check consistency: recompute ω₂ with perturbed μ using the formula and compare.
            if all(v is not None for v in [self.g, self.theta, self.beta, self.r2]):
                numerator = mu_pert * self.g * sp.sin(self.theta) * sp.cos(self.beta)
                denominator = (sp.sin(self.theta) * sp.sin(self.beta) + mu_pert * sp.cos(self.theta)) * self.r2
                omega2_pert = sp.sqrt(numerator / denominator)
                # We don't assert monotonicity, just that the formula holds for perturbed value.
                # We can compute numerically if all are numbers.
                try:
                    omega2_orig_val = float(self.omega2) if not isinstance(self.omega2, (int, float)) else self.omega2
                    omega2_pert_val = float(omega2_pert) if not isinstance(omega2_pert, (int, float)) else omega2_pert
                    # Just log the change, no hard check
                    self._log(f"ω₂ changes from {omega2_orig_val} to {omega2_pert_val} when μ increases by 10%.", True)
                except Exception:
                    self._log("Could not numerically evaluate ω₂ for monotonicity check.", True)

        except Exception as e:
            self._log(f"Monotonicity test error: {e}", False)
            success = False

        return success

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True

        try:
            if not self.test_invariants():
                overall_success = False
            if not self.test_trace_consistency():
                overall_success = False
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self._log(f"Unexpected error during testing: {traceback.format_exc()}", False)
            overall_success = False

        return overall_success, self.logs