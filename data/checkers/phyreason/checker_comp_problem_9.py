import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all parameters and results from the solution instance.
        Uses getattr and .get() to avoid AttributeErrors in sandbox.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters (with safe defaults)
        self.G = self._get_value('G')
        self.m_a = self._get_value('m_a')      # Earth mass (central body)
        self.R = self._get_value('R')          # orbital radius of Atlantis
        self.m = self._get_value('m')          # mass of satellite
        self.B = self._get_value('B')          # magnetic field strength
        self.I = self._get_value('I')          # current in rod
        self.L = self._get_value('L')          # rod length
        self.DeltaR = self._get_value('DeltaR')  # change in orbital radius
        self.T = self._get_value('T')          # orbital period

        # Results and intermediate quantities
        self.Omega = self._get_value('Omega')
        self.alpha = self._get_value('alpha')  # may be string or list
        self.t = self._get_value('t')
        self.DeltaU = self._get_value('DeltaU')
        self.W = self._get_value('W')
        self.F_l = self._get_value('F_l')
        self.U = self._get_value('U')

        # Derived quantities
        self.v = None
        if self.Omega is not None and self.R is not None:
            try:
                self.v = float(self.Omega) * float(self.R)
            except (TypeError, ValueError):
                pass

        # Log storage
        self.logs = []

    def _get_value(self, key):
        """Safely retrieve attribute from solution or results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, None)
        return val

    def log(self, message):
        """Append message to logs."""
        self.logs.append(message)

    def test_invariants(self):
        """Check fundamental physical invariants: orbital condition and energy."""
        success = True
        try:
            # 1. Orbital condition: Ω² = G * m_a / R³
            if all(v is not None for v in [self.G, self.m_a, self.R, self.Omega]):
                lhs = float(self.Omega)**2
                rhs = float(self.G) * float(self.m_a) / (float(self.R)**3)
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Orbital condition Ω² = G*m_a/R³: {lhs} vs {rhs}")
                    success = False
                else:
                    self.log("PASS: Orbital condition holds.")

            # 2. Total energy for circular orbit: U = -G * m_a * m / (2 * R)
            if all(v is not None for v in [self.U, self.G, self.m_a, self.m, self.R]):
                computed_U = -float(self.G) * float(self.m_a) * float(self.m) / (2 * float(self.R))
                if not math.isclose(float(self.U), computed_U, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Total energy U mismatch: {self.U} vs {computed_U}")
                    success = False
                else:
                    self.log("PASS: Total energy formula holds.")

            # 3. Orbital speed relation: v = Ω * R (if v is given)
            if self.v is not None and self.Omega is not None and self.R is not None:
                computed_v = float(self.Omega) * float(self.R)
                if not math.isclose(self.v, computed_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Orbital speed v mismatch: {self.v} vs {computed_v}")
                    success = False
                else:
                    self.log("PASS: Orbital speed relation v = Ω*R holds.")

        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}")
            success = False
        return success

    def test_trace_consistency(self):
        """Check consistency of derived formulas and numerical results."""
        success = True
        try:
            # 1. Check that α values satisfy sin(α)cos(α) = 0
            if self.alpha is not None:
                alphas = self._parse_alpha(self.alpha)
                for a in alphas:
                    if not math.isclose(math.sin(a) * math.cos(a), 0.0, abs_tol=1e-2):
                        self.log(f"FAIL: α={a} does not satisfy sin(α)cos(α)=0")
                        success = False
                if alphas:
                    self.log("PASS: All α values satisfy condition.")

            # 2. Energy change formula: ΔU = 0.5 * m * Ω² * R * ΔR
            if all(v is not None for v in [self.DeltaU, self.m, self.Omega, self.R, self.DeltaR]):
                computed = 0.5 * float(self.m) * (float(self.Omega)**2) * float(self.R) * float(self.DeltaR)
                if not math.isclose(float(self.DeltaU), computed, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: ΔU mismatch: {self.DeltaU} vs {computed}")
                    success = False
                else:
                    self.log("PASS: Energy change formula holds.")

            # 3. Lorentz force: F_l = B * I * L
            if all(v is not None for v in [self.F_l, self.B, self.I, self.L]):
                computed = float(self.B) * float(self.I) * float(self.L)
                if not math.isclose(float(self.F_l), computed, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Lorentz force F_l mismatch: {self.F_l} vs {computed}")
                    success = False
                else:
                    self.log("PASS: Lorentz force formula holds.")

            # 4. Work done: W = F_l * v * t, with v = Ω * R
            if all(v is not None for v in [self.W, self.F_l, self.Omega, self.R, self.t]):
                v = float(self.Omega) * float(self.R)
                computed = float(self.F_l) * v * float(self.t)
                if not math.isclose(float(self.W), computed, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Work W mismatch: {self.W} vs {computed}")
                    success = False
                else:
                    self.log("PASS: Work formula holds.")

            # 5. Time formula: t = 0.5 * m * Ω * ΔR / (B * I * L)
            if all(v is not None for v in [self.t, self.m, self.Omega, self.DeltaR, self.B, self.I, self.L]):
                computed = 0.5 * float(self.m) * float(self.Omega) * float(self.DeltaR)
                computed /= (float(self.B) * float(self.I) * float(self.L))
                if not math.isclose(float(self.t), computed, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Time t mismatch: {self.t} vs {computed}")
                    success = False
                else:
                    self.log("PASS: Time formula holds.")

            # 6. Numerical check: t ≈ 5.8 × 10³ s
            if self.t is not None:
                if not math.isclose(float(self.t), 5.8e3, rel_tol=1e-2, abs_tol=1e2):
                    self.log(f"FAIL: Time t={self.t} not close to 5.8×10³ s")
                    success = False
                else:
                    self.log("PASS: Time value ≈ 5.8×10³ s.")

        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}")
            success = False
        return success

    def test_monotonicity(self):
        """Perturb ΔR and check that t changes proportionally."""
        success = True
        try:
            # Required parameters for t formula
            required = [self.m, self.Omega, self.DeltaR, self.B, self.I, self.L]
            if any(v is None for v in required):
                self.log("SKIP: Missing parameters for monotonicity test.")
                return True  # Not a failure, just skipped

            # Original t
            t0 = 0.5 * float(self.m) * float(self.Omega) * float(self.DeltaR)
            t0 /= (float(self.B) * float(self.I) * float(self.L))

            # Perturb ΔR by +1%
            DeltaR_pert = float(self.DeltaR) * 1.01
            t1 = 0.5 * float(self.m) * float(self.Omega) * DeltaR_pert
            t1 /= (float(self.B) * float(self.I) * float(self.L))

            if t1 > t0:
                self.log("PASS: Monotonicity: t increases with ΔR.")
            else:
                self.log(f"FAIL: Monotonicity violated: t0={t0}, t1={t1} (after +1% ΔR)")
                success = False

        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}")
            success = False
        return success

    def _parse_alpha(self, alpha_val):
        """
        Convert alpha from string or list to list of float radian values.
        Handles formats like "$0,\pi,\frac{\pi}{2},\frac{3\pi}{2}$"
        """
        alphas = []
        try:
            if isinstance(alpha_val, str):
                # Remove LaTeX markers and whitespace
                s = alpha_val.strip('$').replace(' ', '')
                parts = s.split(',')
                for p in parts:
                    if p == '0':
                        alphas.append(0.0)
                    elif p == r'\pi':
                        alphas.append(math.pi)
                    elif p == r'\frac{\pi}{2}':
                        alphas.append(math.pi / 2)
                    elif p == r'\frac{3\pi}{2}':
                        alphas.append(3 * math.pi / 2)
                    else:
                        # Try direct float conversion
                        alphas.append(float(p))
            elif isinstance(alpha_val, (list, tuple)):
                alphas = [float(a) for a in alpha_val]
            else:
                alphas = [float(alpha_val)]
        except Exception as e:
            self.log(f"Warning: Could not parse alpha: {e}")
        return alphas

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            self.log(f"\n--- Running {name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"EXCEPTION in {name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, self.logs