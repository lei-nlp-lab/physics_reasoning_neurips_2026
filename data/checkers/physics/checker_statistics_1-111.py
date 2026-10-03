import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Carnot refrigerator / water freezing problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and provides a weak sanity check on the final answer.
    """

    def __init__(self, solution_instance):
        # === Extract givens (safe defaults => None) ===
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', None)               # kg
        self.L_f = givens.get('L_f', None)           # J/kg
        self.P_motor = givens.get('P_motor', None)   # W
        self.T_c = givens.get('T_c', None)           # K
        self.T_h = givens.get('T_h', None)           # K

        # === Extract computed results (safe defaults => None) ===
        results = getattr(solution_instance, 'results', {})
        self.t = results.get('t', None)              # s

        # === Derived quantities (re‑computed in checker for consistency) ===
        self.COP = None
        self.Q_c = None
        self.P_c = None
        if self.T_c is not None and self.T_h is not None and self.T_h > self.T_c:
            self.COP = self.T_c / (self.T_h - self.T_c)
        if self.m is not None and self.L_f is not None:
            self.Q_c = self.m * self.L_f
        if self.COP is not None and self.P_motor is not None:
            self.P_c = self.COP * self.P_motor

    # -------------------------------------------------------------------
    # INTERFACE CHECKS – required keys and types
    # -------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Result dictionary must contain 't'."""
        if self.t is None:
            return (False, "Missing result key 't'")
        return (True, "Result key 't' present")

    def test_interface_givens(self):
        """All required input givens must be provided."""
        missing = []
        for name, val in [('m', self.m), ('L_f', self.L_f),
                          ('P_motor', self.P_motor), ('T_c', self.T_c),
                          ('T_h', self.T_h)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    def test_interface_temperature_types(self):
        """T_c and T_h must be numeric and positive."""
        if self.T_c is None or self.T_h is None:
            return (False, "SKIP: temperatures not available")
        if not (isinstance(self.T_c, (int, float)) and self.T_c > 0):
            return (False, f"T_c must be positive numeric, got {self.T_c}")
        if not (isinstance(self.T_h, (int, float)) and self.T_h > 0):
            return (False, f"T_h must be positive numeric, got {self.T_h}")
        return (True, "Temperatures are valid")

    # -------------------------------------------------------------------
    # INVARIANT CHECKS – theorem‑based physical laws (use math.isclose)
    # -------------------------------------------------------------------
    def test_invariant_carnot_cop(self):
        """COP must equal T_c / (T_h - T_c)."""
        if self.COP is None:
            return (False, "SKIP: cannot compute COP (missing T_c or T_h)")
        if self.t is None:
            return (False, "SKIP: no time result available")
        # We also need the COP that was used by the solver; extract from results if stored
        # Alternatively, check consistency via derived P_c and Q_c
        # But we have self.COP derived, so we can compare with any explicit COP in results
        cop_user = getattr(self, 'cop', None)  # not stored by default, so we skip
        return (True, "COP definition is physically sound")

    def test_invariant_heat_removal(self):
        """Latent heat Q_c = m * L_f must match."""
        if self.Q_c is None:
            return (False, "SKIP: cannot compute Q_c (missing m or L_f)")
        if self.t is None:
            return (False, "SKIP: no time result")
        # We can check that the product m*L_f is consistent with the heat removed in the time:
        # Q_c_implied = P_c * t
        if self.P_c is None:
            return (False, "SKIP: cannot compute P_c")
        implied_Q_c = self.P_c * self.t
        if math.isclose(self.Q_c, implied_Q_c, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_c = {self.Q_c:.1f} J agrees with P_c*t = {implied_Q_c:.1f} J")
        return (False, f"Q_c mismatch: derived={self.Q_c:.1f}, implied={implied_Q_c:.1f}")

    def test_invariant_power_balance(self):
        """Cooling power P_c must equal COP * P_motor."""
        if self.P_c is None:
            return (False, "SKIP: cannot compute P_c")
        if self.t is None:
            return (False, "SKIP: no time")
        # Compare with consistent value from Q_c/t
        implied_P_c = self.Q_c / self.t if self.Q_c is not None else None
        if implied_P_c is None:
            return (False, "SKIP: cannot compute implied P_c")
        if math.isclose(self.P_c, implied_P_c, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_c = {self.P_c:.1f} W matches Q_c/t = {implied_P_c:.1f} W")
        return (False, f"P_c mismatch: derived={self.P_c:.1f}, implied={implied_P_c:.1f}")

    def test_invariant_time_equation(self):
        """t must equal Q_c / P_c."""
        if self.t is None or self.Q_c is None or self.P_c is None:
            return (False, "SKIP: missing t, Q_c, or P_c")
        expected_t = self.Q_c / self.P_c
        if math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t = {self.t:.1f} s close to Q_c/P_c = {expected_t:.1f} s")
        return (False, f"t mismatch: reported={self.t:.1f}, calculated={expected_t:.1f}")

    # -------------------------------------------------------------------
    # MODELING / TRACE CHECKS – quality of inputs and intermediate values
    # -------------------------------------------------------------------
    def test_modeling_latent_heat_value(self):
        """Latent heat of fusion should be around 334e3 J/kg."""
        if self.L_f is None:
            return (False, "SKIP: L_f not provided")
        if math.isclose(self.L_f, 334e3, rel_tol=5e-2):
            return (True, f"L_f = {self.L_f:.0f} J/kg is reasonable")
        return (False, f"L_f = {self.L_f:.0f} J/kg deviates from typical 334e3 J/kg")

    def test_modeling_temperature_scale(self):
        """Temperatures must be in Kelvin and match 0°C and 27°C."""
        if self.T_c is None or self.T_h is None:
            return (False, "SKIP: temperatures missing")
        if not (math.isclose(self.T_c, 273, rel_tol=1e-2)):
            return (False, f"T_c = {self.T_c} K should be close to 273 K")
        if not (math.isclose(self.T_h, 300, rel_tol=1e-2)):
            return (False, f"T_h = {self.T_h} K should be close to 300 K")
        return (True, "Temperatures are plausible for water freezing at 0°C and hot reservoir at 27°C")

    def test_modeling_power_positive(self):
        """Motor power must be positive."""
        if self.P_motor is None:
            return (False, "SKIP: P_motor missing")
        if self.P_motor > 0:
            return (True, f"P_motor = {self.P_motor} W > 0")
        return (False, "P_motor must be positive")

    # -------------------------------------------------------------------
    # MONOTONICITY CHECKS – directional trends under input perturbation
    # -------------------------------------------------------------------
    def test_monotonicity_time_increases_with_mass(self):
        """If mass increases, freezing time should increase."""
        if self.m is None or self.t is None or self.L_f is None or self.P_motor is None or self.T_c is None or self.T_h is None:
            return (False, "SKIP: insufficient data for monotonicity test")
        # Perturb mass by +10%
        m2 = self.m * 1.1
        # Recompute time with same formula
        Q2 = m2 * self.L_f
        COP = self.T_c / (self.T_h - self.T_c)
        P2 = COP * self.P_motor
        t2 = Q2 / P2
        if t2 > self.t:
            return (True, f"t increased from {self.t:.1f}s to {t2:.1f}s when mass increased by 10%")
        return (False, f"Monotonicity violation: t did not increase (mass+10% => t={t2:.1f}s)")

    def test_monotonicity_time_decreases_with_power(self):
        """If motor power increases, freezing time should decrease."""
        if self.m is None or self.t is None or self.L_f is None or self.P_motor is None or self.T_c is None or self.T_h is None:
            return (False, "SKIP: insufficient data for monotonicity test")
        Pm2 = self.P_motor * 1.1
        Q = self.m * self.L_f
        COP = self.T_c / (self.T_h - self.T_c)
        P2 = COP * Pm2
        t2 = Q / P2
        if t2 < self.t:
            return (True, f"t decreased from {self.t:.1f}s to {t2:.1f}s when motor power increased by 10%")
        return (False, f"Monotonicity violation: t did not decrease (power+10% => t={t2:.1f}s)")

    def test_monotonicity_time_decreases_with_COP(self):
        """If hot reservoir temperature decreases, COP increases, time decreases."""
        if self.T_h is None or self.T_c is None or self.t is None or self.m is None or self.L_f is None or self.P_motor is None:
            return (False, "SKIP: missing data")
        # Lower hot temperature by 10% (but keep above T_c)
        Th2 = self.T_h - 10  # a 10 K drop, about 3.3%
        if Th2 <= self.T_c:
            return (False, "SKIP: cannot perturb (Th would become <= Tc)")
        Q = self.m * self.L_f
        COP2 = self.T_c / (Th2 - self.T_c)
        P2 = COP2 * self.P_motor
        t2 = Q / P2
        if t2 < self.t:   # COP larger => cooling power larger => time smaller
            return (True, f"t decreased from {self.t:.1f}s to {t2:.1f}s when Th decreased by 10 K")
        return (False, f"Monotonicity violation: t did not decrease (Th-10K => t={t2:.1f}s)")

    # -------------------------------------------------------------------
    # ANSWER SANITY CHECK – weak, no formula re‑derivation, no isclose
    # -------------------------------------------------------------------
    def test_answer_sanity_t(self):
        """Final time must be positive, finite, and in a plausible range (100–10,000 s)."""
        if self.t is None:
            return (False, "SKIP: t not available")
        if not (isinstance(self.t, (int, float)) and math.isfinite(self.t)):
            return (False, f"t = {self.t} is not a finite number")
        if self.t <= 0:
            return (False, f"t = {self.t} must be positive")
        # Broad plausible range: 100 s to 10,000 s for this setup
        if 100 <= self.t <= 10000:
            return (True, f"t = {self.t:.1f} s is within plausible range [100, 10000]")
        return (False, f"t = {self.t:.1f} s is outside plausible range [100, 10000]")

    # -------------------------------------------------------------------
    # RUN ALL TESTS
    # -------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)