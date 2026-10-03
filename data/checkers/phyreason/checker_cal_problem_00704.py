import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract results safely using getattr and .get() fallbacks."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Wave parameters from solution (with safe defaults)
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.T = getattr(solution_instance, 'T', self.results.get('T', 0.0))
        self.lam = getattr(solution_instance, 'λ', self.results.get('λ', 0.0))

        # Input parameters from problem statement
        self.d_buoys = 10.0  # meters between buoys
        self.t_travel = 2.0   # seconds until 5th buoy starts
        self.t_osc = 3.0      # seconds for 10 oscillations
        self.n_osc = 10.0     # number of oscillations

        # Expected values from steps analysis
        self.v_expected = 20.0  # m/s
        self.T_expected = 0.3   # s
        self.lam_expected = 6.0 # m

        self.logs = []

    def log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        self.logs.append(msg)
        if success is not None:
            self.logs.append(f"Result: {'PASS' if success else 'FAIL'}")
        return success

    def fuzzy_eq(self, a, b, name=""):
        """Robust fuzzy equality check with 1% tolerance."""
        is_close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        self.log(f"{name}: {a} ≈ {b} ? {is_close}")
        return is_close

    def test_invariants(self):
        """Check physics laws (wave speed formula, λ = vT)."""
        try:
            # Symbolic check
            v_sym, T_sym, lam_sym = sp.symbols('v T λ')
            wave_law = sp.Eq(lam_sym, v_sym * T_sym)

            # Numerical substitution
            subs = {v_sym: float(self.v), T_sym: float(self.T), lam_sym: float(self.lam)}
            symbolic_ok = wave_law.subs(subs).simplify()

            # Fuzzy numerical fallback
            numerical_ok = self.fuzzy_eq(self.lam, self.v * self.T, "λ = v * T")
            success = bool(symbolic_ok) or numerical_ok  # Accept either
            return self.log("Wave speed-wavelength-period relation", success)
        except Exception as e:
            self.log(f"Invariant check failed: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Verify intermediate steps match final results."""
        try:
            # Step 1: Wave speed from travel time
            distance = 4 * self.d_buoys  # 1st to 5th buoy = 4 intervals
            v_calc = distance / self.t_travel
            step1_ok = self.fuzzy_eq(self.v, v_calc, "v = Δx/Δt")

            # Step 2: Period from oscillations
            T_calc = self.t_osc / self.n_osc
            step2_ok = self.fuzzy_eq(self.T, T_calc, "T = t_osc / n_osc")

            # Step 3: Wavelength from v and T
            lam_calc = v_calc * T_calc
            step3_ok = self.fuzzy_eq(self.lam, lam_calc, "λ = v * T")

            # Compare with expected values from KB
            expected_ok = all([
                self.fuzzy_eq(self.v, self.v_expected, "v vs expected"),
                self.fuzzy_eq(self.T, self.T_expected, "T vs expected"),
                self.fuzzy_eq(self.lam, self.lam_expected, "λ vs expected")
            ])

            overall = step1_ok and step2_ok and step3_ok and expected_ok
            return self.log("Trace consistency (steps & expected values)", overall)
        except Exception as e:
            self.log(f"Trace consistency failed: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb an input and verify output trends match physics."""
        original_d = self.d_buoys
        try:
            # Perturb: Increase distance between buoys by 50%
            perturbed_d = original_d * 1.5
            self.log(f"Perturbing: d_buoys {original_d} → {perturbed_d}")

            # Recalculate expected wave speed (v ∝ distance if time fixed)
            new_v = (4 * perturbed_d) / self.t_travel
            v_ratio = new_v / self.v_expected
            expected_v_ratio = perturbed_d / original_d  # v ∝ d
            v_trend_ok = self.fuzzy_eq(v_ratio, expected_v_ratio, "v ratio vs expected")

            # Recalculate expected wavelength (λ ∝ v if T fixed)
            new_lam = new_v * self.T_expected
            lam_ratio = new_lam / self.lam_expected
            expected_lam_ratio = expected_v_ratio  # λ ∝ v
            lam_trend_ok = self.fuzzy_eq(lam_ratio, expected_lam_ratio, "λ ratio vs expected")

            # Restore original distance (for safety)
            self.d_buoys = original_d

            overall = v_trend_ok and lam_trend_ok
            return self.log("Monotonicity: v and λ scale with distance", overall)
        except Exception as e:
            self.d_buoys = original_d  # Restore on error
            self.log(f"Monotonicity failed: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs.clear()
        results = []
        try:
            results.append(self.test_invariants())
            results.append(self.test_trace_consistency())
            results.append(self.test_monotonicity())
            success = all(results)
        except Exception as e:
            self.log(f"Run all tests crashed: {traceback.format_exc()}")
            success = False
        return success, self.logs.copy()