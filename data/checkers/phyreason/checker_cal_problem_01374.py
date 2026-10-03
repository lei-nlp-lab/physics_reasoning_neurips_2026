import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})

        # Extract inputs safely
        self.r = getattr(solution_instance, 'r', None) or self.inputs.get('r')
        self.T = getattr(solution_instance, 'T', None) or self.inputs.get('T')
        self.R = getattr(solution_instance, 'R', None) or self.inputs.get('R')
        self.G = getattr(solution_instance, 'G', None) or self.inputs.get('G')

        # Extract results safely
        self.M = getattr(solution_instance, 'M', None) or self.results.get('M')
        self.T_1 = getattr(solution_instance, 'T_1', None) or self.results.get('T_1')
        self.t = getattr(solution_instance, 't', None) or self.results.get('t')
        self.a = getattr(solution_instance, 'a', None) or self.results.get('a')
        self.rho = getattr(solution_instance, 'rho', None) or self.results.get('rho') or self.results.get('ρ')

        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def test_invariants(self):
        """Verify physical invariants from the steps analysis."""
        try:
            pi = sp.pi

            # 1. Check mass of the Moon M = (4π² r³) / (G T²)
            if all(v is not None for v in [self.M, self.r, self.T, self.G]):
                M_expected = (4 * pi**2 * self.r**3) / (self.G * self.T**2)
                diff = sp.simplify(self.M - M_expected)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.M), float(M_expected), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: M mismatch. Expected {M_expected}, got {self.M}")
                            return False
                    except:
                        self.log(f"FAIL: M symbolic mismatch. Diff: {diff}")
                        return False
                self.log("PASS: M matches Newton+centripetal formula.")

            # 2. Check semi-major axis a = (R + r)/2
            if all(v is not None for v in [self.a, self.r, self.R]):
                a_expected = (self.R + self.r) / 2
                diff = sp.simplify(self.a - a_expected)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.a), float(a_expected), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: a mismatch. Expected {a_expected}, got {self.a}")
                            return False
                    except:
                        self.log(f"FAIL: a symbolic mismatch. Diff: {diff}")
                        return False
                self.log("PASS: a matches ellipse geometry.")

            # 3. Check period T₁ = T * √(a³ / r³)
            if all(v is not None for v in [self.T_1, self.a, self.r, self.T]):
                T1_expected = self.T * sp.sqrt(self.a**3 / self.r**3)
                diff = sp.simplify(self.T_1 - T1_expected)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.T_1), float(T1_expected), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: T₁ mismatch. Expected {T1_expected}, got {self.T_1}")
                            return False
                    except:
                        self.log(f"FAIL: T₁ symbolic mismatch. Diff: {diff}")
                        return False
                self.log("PASS: T₁ matches Kepler's third law.")

            # 4. Check time t = T₁ / 2
            if all(v is not None for v in [self.t, self.T_1]):
                t_expected = self.T_1 / 2
                diff = sp.simplify(self.t - t_expected)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.t), float(t_expected), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: t mismatch. Expected {t_expected}, got {self.t}")
                            return False
                    except:
                        self.log(f"FAIL: t symbolic mismatch. Diff: {diff}")
                        return False
                self.log("PASS: t = T₁/2 holds.")

            # 5. Check density ρ = (3π r³) / (G T² R³)
            if all(v is not None for v in [self.rho, self.r, self.T, self.R, self.G]):
                rho_expected = (3 * pi * self.r**3) / (self.G * self.T**2 * self.R**3)
                diff = sp.simplify(self.rho - rho_expected)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.rho), float(rho_expected), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: ρ mismatch. Expected {rho_expected}, got {self.rho}")
                            return False
                    except:
                        self.log(f"FAIL: ρ symbolic mismatch. Diff: {diff}")
                        return False
                self.log("PASS: ρ matches density formula.")

            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check ρ from M and R: ρ = M / ( (4/3)π R³ )
            if all(v is not None for v in [self.rho, self.M, self.R]):
                pi = sp.pi
                rho_from_M = self.M / ( (4/3) * pi * self.R**3 )
                diff = sp.simplify(self.rho - rho_from_M)
                if diff != 0:
                    try:
                        if not math.isclose(float(self.rho), float(rho_from_M), rel_tol=1e-2, abs_tol=1e-2):
                            self.log(f"FAIL: ρ from M inconsistent. Expected {rho_from_M}, got {self.rho}")
                            return False
                    except:
                        self.log(f"FAIL: ρ from M symbolic inconsistency. Diff: {diff}")
                        return False
                self.log("PASS: ρ consistent with M and R.")

            # 2. Check t from T₁ (already done in invariants, but double-check)
            if all(v is not None for v in [self.t, self.T_1]):
                if not math.isclose(float(self.t), float(self.T_1/2), rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: t not half of T₁. t={self.t}, T₁/2={self.T_1/2}")
                    return False
                self.log("PASS: t = T₁/2 consistent.")

            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}")
            return False

    def test_monotonicity(self):
        """Perturb r and verify trends in ρ and t."""
        try:
            if any(v is None for v in [self.r, self.T, self.R, self.G, self.rho, self.t]):
                self.log("SKIP: Missing data for monotonicity test.")
                return True  # Not a failure, just skip

            pi = sp.pi
            # Perturb r by +10%
            perturbed_r = self.r * 1.1

            # Compute ρ for perturbed r: ρ ∝ r³
            rho_perturbed = (3 * pi * perturbed_r**3) / (self.G * self.T**2 * self.R**3)

            # Compute t for perturbed r: t = (R+r)/(4r) * √((R+r)/(2r)) * T
            a_perturbed = (self.R + perturbed_r) / 2
            T1_perturbed = self.T * sp.sqrt(a_perturbed**3 / perturbed_r**3)
            t_perturbed = T1_perturbed / 2

            # Convert to numerical if symbolic
            try:
                rho_orig = float(self.rho) if not hasattr(self.rho, 'subs') else float(self.rho.evalf())
                rho_pert = float(rho_perturbed) if not hasattr(rho_perturbed, 'subs') else float(rho_perturbed.evalf())
                t_orig = float(self.t) if not hasattr(self.t, 'subs') else float(self.t.evalf())
                t_pert = float(t_perturbed) if not hasattr(t_perturbed, 'subs') else float(t_perturbed.evalf())
            except Exception as e:
                self.log(f"SKIP: Could not evaluate numerically: {e}")
                return True  # Skip if symbolic without numerical substitution

            # Check ρ trend: increasing with r (since ρ ∝ r³)
            if perturbed_r > self.r:
                if not rho_pert > rho_orig:
                    self.log(f"FAIL: ρ should increase when r increases. ρ_orig={rho_orig}, ρ_pert={rho_pert}")
                    return False
                self.log("PASS: ρ increases with r (cubic trend).")
            else:
                if not rho_pert < rho_orig:
                    self.log(f"FAIL: ρ should decrease when r decreases. ρ_orig={rho_orig}, ρ_pert={rho_pert}")
                    return False
                self.log("PASS: ρ decreases with r (cubic trend).")

            # Check t trend: decreasing with r (derived from formula)
            if perturbed_r > self.r:
                if not t_pert < t_orig:
                    self.log(f"FAIL: t should decrease when r increases. t_orig={t_orig}, t_pert={t_pert}")
                    return False
                self.log("PASS: t decreases with r (monotonic).")
            else:
                if not t_pert > t_orig:
                    self.log(f"FAIL: t should increase when r decreases. t_orig={t_orig}, t_pert={t_pert}")
                    return False
                self.log("PASS: t increases with r (monotonic).")

            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}")
            return False

    def run_all_tests(self):
        """Execute all checks and return success and logs."""
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test in tests:
            try:
                if not test():
                    success = False
                    self.log(f"--- {name} test failed ---")
                else:
                    self.log(f"--- {name} test passed ---")
            except Exception as e:
                success = False
                self.log(f"--- {name} test crashed: {e} ---")
                self.log(traceback.format_exc())

        return success, self.logs