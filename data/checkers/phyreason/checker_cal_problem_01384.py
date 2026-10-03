import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Inputs (with defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.M = getattr(solution_instance, 'M', None)
        self.L = getattr(solution_instance, 'L', None)
        self.G = getattr(solution_instance, 'G', None)

        # Results (with safe fallbacks)
        self.r = self.results.get('r', getattr(solution_instance, 'r', None))
        self.R = self.results.get('R', getattr(solution_instance, 'R', None))
        self.T = self.results.get('T', getattr(solution_instance, 'T', None))
        self.T1 = self.results.get('T₁', self.results.get('T1', getattr(solution_instance, 'T₁', getattr(solution_instance, 'T1', None))))
        self.T2 = self.results.get('T₂', self.results.get('T2', getattr(solution_instance, 'T₂', getattr(solution_instance, 'T2', None))))
        self.ratio_sq = self.results.get('(T₂ / T₁)²', self.results.get('T₂/T₁²', getattr(solution_instance, 'ratio_sq', None)))

        # Symbolic variables for invariant checks
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'M': sp.symbols('M', positive=True),
            'L': sp.symbols('L', positive=True),
            'G': sp.symbols('G', positive=True),
            'r': sp.symbols('r', positive=True),
            'R': sp.symbols('R', positive=True),
            'T': sp.symbols('T', positive=True),
            'T1': sp.symbols('T1', positive=True),
            'T2': sp.symbols('T2', positive=True),
            'ω': sp.symbols('ω', positive=True)
        }

        self.logs = []

    def _log(self, msg, success):
        self.logs.append((success, msg))

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for numbers or sympy expressions."""
        if a is None or b is None:
            return False
        try:
            # If both are numeric (float/int)
            num_a = float(a) if not isinstance(a, (sp.Expr, sp.Symbol)) else float(a.evalf())
            num_b = float(b) if not isinstance(b, (sp.Expr, sp.Symbol)) else float(b.evalf())
            return math.isclose(num_a, num_b, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # Symbolic equality check (simplify difference)
            if isinstance(a, (sp.Expr, sp.Symbol)) and isinstance(b, (sp.Expr, sp.Symbol)):
                return sp.simplify(a - b) == 0
            return False

    def test_invariants(self):
        """Check physical laws: centripetal force balance, radius sum, period formula."""
        success = True

        # 1. Centripetal force balance: m * r = M * R (from equal gravitational force)
        if None not in (self.m, self.M, self.r, self.R):
            lhs = self.m * self.r
            rhs = self.M * self.R
            if not self._is_close(lhs, rhs):
                self._log(f"Centripetal force balance failed: m*r = {lhs}, M*R = {rhs}", False)
                success = False
            else:
                self._log(f"Centripetal force balance OK: m*r ≈ M*R", True)
        else:
            self._log("Skipping centripetal force balance (missing data)", True)

        # 2. Geometric constraint: r + R = L
        if None not in (self.r, self.R, self.L):
            if not self._is_close(self.r + self.R, self.L):
                self._log(f"Radius sum failed: r + R = {self.r + self.R}, L = {self.L}", False)
                success = False
            else:
                self._log(f"Radius sum OK: r + R ≈ L", True)
        else:
            self._log("Skipping radius sum (missing data)", True)

        # 3. Period formula consistency (using gravitational force as centripetal force)
        # From F_grav = m * ω² * r, with ω = 2π/T
        # => G * M * m / L² = m * (2π/T)² * r  => T = 2π √(r * L² / (G * M))
        if None not in (self.G, self.M, self.m, self.L, self.r, self.T):
            try:
                # Compute expected T from the derived formula (using r = M*L/(M+m) if available, else direct)
                if None not in (self.M, self.m, self.L):
                    r_expected = (self.M * self.L) / (self.M + self.m) if self.r is None else self.r
                else:
                    r_expected = self.r
                T_expected = 2 * math.pi * math.sqrt(r_expected * self.L**2 / (self.G * self.M))
                if not self._is_close(self.T, T_expected):
                    self._log(f"Period formula mismatch: T = {self.T}, expected ≈ {T_expected}", False)
                    success = False
                else:
                    self._log(f"Period formula OK: T ≈ 2π √(r L²/(G M))", True)
            except Exception as e:
                self._log(f"Period formula check error: {e}", False)
                success = False
        else:
            self._log("Skipping period formula (missing data)", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True

        # Check that r and R from step_3 match the derived formulas
        if None not in (self.m, self.M, self.L, self.r, self.R):
            r_expected = (self.M * self.L) / (self.M + self.m)
            R_expected = (self.m * self.L) / (self.M + self.m)
            if not (self._is_close(self.r, r_expected) and self._is_close(self.R, R_expected)):
                self._log(f"Orbital radii mismatch: r = {self.r} (expected {r_expected}), R = {self.R} (expected {R_expected})", False)
                success = False
            else:
                self._log(f"Orbital radii consistent with formulas", True)
        else:
            self._log("Skipping orbital radii consistency (missing data)", True)

        # Check that T matches the formula from step_4
        if None not in (self.G, self.M, self.m, self.L, self.T):
            T_expected = 2 * math.pi * math.sqrt(self.L**3 / (self.G * (self.M + self.m)))
            if not self._is_close(self.T, T_expected):
                self._log(f"Period T mismatch: T = {self.T}, expected {T_expected}", False)
                success = False
            else:
                self._log(f"Period T consistent with formula", True)
        else:
            self._log("Skipping period T consistency (missing data)", True)

        # Check T1 and T2 if available
        if None not in (self.T1, self.G, self.M, self.m, self.L):
            T1_expected = 2 * math.pi * math.sqrt(self.L**3 / (self.G * (self.M + self.m)))
            if not self._is_close(self.T1, T1_expected):
                self._log(f"T1 mismatch: T1 = {self.T1}, expected {T1_expected}", False)
                success = False
            else:
                self._log(f"T1 consistent with binary period", True)

        if None not in (self.T2, self.G, self.M, self.L):
            T2_expected = 2 * math.pi * math.sqrt(self.L**3 / (self.G * self.M))
            if not self._is_close(self.T2, T2_expected):
                self._log(f"T2 mismatch: T2 = {self.T2}, expected {T2_expected}", False)
                success = False
            else:
                self._log(f"T2 consistent with fixed-center period", True)

        # Check ratio (T2/T1)^2 = (M+m)/M
        if None not in (self.ratio_sq, self.M, self.m):
            ratio_expected = (self.M + self.m) / self.M
            if not self._is_close(self.ratio_sq, ratio_expected):
                self._log(f"Ratio mismatch: (T2/T1)^2 = {self.ratio_sq}, expected {ratio_expected}", False)
                success = False
            else:
                self._log(f"Ratio (T2/T1)^2 consistent with (M+m)/M", True)

        return success

    def test_monotonicity(self):
        """Perturb mass m and verify expected trends in period and ratio."""
        success = True

        if None in (self.m, self.M, self.L, self.G):
            self._log("Skipping monotonicity (missing input data)", True)
            return True

        # Perturbation factor (10% increase)
        original_m = self.m
        perturbed_m = original_m * 1.1

        # 1. Period T = 2π √(L³/(G(M+m))) should decrease when m increases
        T_original = 2 * math.pi * math.sqrt(self.L**3 / (self.G * (self.M + original_m)))
        T_perturbed = 2 * math.pi * math.sqrt(self.L**3 / (self.G * (self.M + perturbed_m)))
        if T_perturbed >= T_original:
            self._log(f"Monotonicity failed: T increased when m increased (T_orig={T_original}, T_pert={T_perturbed})", False)
            success = False
        else:
            self._log(f"Monotonicity OK: T decreases when m increases", True)

        # 2. Ratio (T2/T1)^2 = (M+m)/M should increase when m increases
        ratio_original = (self.M + original_m) / self.M
        ratio_perturbed = (self.M + perturbed_m) / self.M
        if ratio_perturbed <= ratio_original:
            self._log(f"Monotonicity failed: ratio did not increase when m increased", False)
            success = False
        else:
            self._log(f"Monotonicity OK: ratio (T2/T1)^2 increases when m increases", True)

        # Restore original m (important for sandbox safety)
        self.m = original_m
        return success

    def run_all_tests(self):
        """Execute all checks and return (overall_success, logs)."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{name} test crashed: {traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs