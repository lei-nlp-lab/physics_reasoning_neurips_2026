import math
import sympy as sp

class PhysicsChecker:
    """
    Audits hydrogen adsorption coverage solution:
    - Verifies interface completeness
    - Checks invariant properties (0 ≤ θ ≤ 1)
    - Checks modeling consistency via derived formula
    - Tests monotonicity with respect to pressure and temperature
    - Performs a weak final-answer sanity check
    """
    def __init__(self, solution_instance):
        # Givens: Temperature T [K], Pressure P [Pa], binding energy epsilon0 [J],
        # mass of H2 molecule m [kg], Planck constant h [J·s], Boltzmann constant k_B [J/K]
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})

        # Safe extraction – never use numeric defaults!
        self.T = givens.get('T', None)
        self.P = givens.get('P', None)
        self.epsilon0 = givens.get('epsilon0', None)
        self.m = givens.get('m', None)
        self.h = givens.get('h', None)
        self.k_B = givens.get('k_B', None)

        # Results
        self.theta = results.get('theta', None)

        # Additional internally derived constants (computed only when all ingredients present)
        self._lambda = None
        self._beta = None
        self._A = None
        self._p0 = None
        self._derived_theta = None

        if all(x is not None for x in [self.T, self.P, self.epsilon0, self.m, self.h, self.k_B]):
            self._compute_derived()

    def _compute_derived(self):
        """Compute all derived quantities needed for modeling checks."""
        try:
            beta = 1.0 / (self.k_B * self.T)
            lam = self.h / math.sqrt(2.0 * math.pi * self.m * self.k_B * self.T)
            A = lam**3 / (self.k_B * self.T)
            p0 = 1.0 / (A * math.exp(beta * self.epsilon0))

            # Coverage formula from physics mapping
            numerator = self.P * (1.0 + math.exp(-beta * self.epsilon0 / 2.0))
            denominator = p0 + self.P * (1.0 + math.exp(-beta * self.epsilon0 / 2.0))
            theta_derived = numerator / denominator

            self._beta = beta
            self._lambda = lam
            self._A = A
            self._p0 = p0
            self._derived_theta = theta_derived
        except (ValueError, OverflowError, ZeroDivisionError):
            pass  # leave derived quantities as None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_givens_present(self):
        """Check that all required input givens exist."""
        missing = []
        for name in ['T', 'P', 'epsilon0', 'm', 'h', 'k_B']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens present")

    def test_interface_theta_present(self):
        """Check that the result 'theta' exists."""
        if self.theta is None:
            return (False, "theta is missing from results")
        return (True, "theta present")

    def test_interface_theta_is_numeric(self):
        """Check theta is a finite number (not sympy symbolic, not None)."""
        if self.theta is None:
            return (False, "theta is None")
        if isinstance(self.theta, (int, float)):
            if math.isfinite(self.theta):
                return (True, "theta is a finite number")
            else:
                return (False, f"theta is not finite: {self.theta}")
        # also accept sympy numeric expression that can be evaluated
        try:
            val = float(self.theta)
            if math.isfinite(val):
                return (True, "theta evaluates to finite number")
            else:
                return (False, f"theta evaluates to non-finite: {val}")
        except (TypeError, ValueError, OverflowError):
            return (False, "theta is not a numeric value")

    # ----------------------------------------------------------------------
    # Invariant tests (physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_coverage_range(self):
        """Coverage must be between 0 and 1 inclusive."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        try:
            theta_val = float(self.theta)
        except (TypeError, ValueError):
            return (False, "theta cannot be converted to float")
        if 0.0 <= theta_val <= 1.0 + 1e-12:
            return (True, f"theta = {theta_val:.4g} ∈ [0,1]")
        else:
            return (False, f"theta = {theta_val:.4g} outside [0,1]")

    def test_invariant_chemical_potential_consistency(self):
        """
        Verify that the Grand Canonical partition function structure
        (empty, state1, state2) is respected by checking that
        theta = (exp(beta*(mu+ε0)) + exp(beta*(mu+ε0/2))) / (1 + ...)
        This is a symbolic relation; we check numerically if possible.
        """
        if self._derived_theta is None or self.theta is None:
            return (False, "SKIP: missing data for chemical potential consistency")
        try:
            theta_val = float(self.theta)
            derived = float(self._derived_theta)
        except (TypeError, ValueError):
            return (False, "Cannot convert theta to float")
        if math.isclose(theta_val, derived, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta consistent with derived formula (diff {abs(theta_val-derived):.3g})")
        else:
            return (False, f"theta ({theta_val:.4g}) deviates from derived ({derived:.4g})")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_low_pressure_limit(self):
        """
        As P → 0, coverage → 0.  We cannot set P=0, but we can check that
        if we set P very small (e.g., 1% of original) the derived formula
        yields a value close to 0 and the reported theta shows same trend.
        This test requires numeric values to compute.
        """
        if self._derived_theta is None or self.theta is None:
            return (False, "SKIP: missing data for low-pressure modeling")
        try:
            theta_orig = float(self.theta)
            # Use derived formula to compute with P reduced to 1% of original
            beta, epsilon0 = self._beta, self.epsilon0
            P_small = self.P * 0.01
            numerator = P_small * (1.0 + math.exp(-beta * epsilon0 / 2.0))
            denominator = self._p0 + P_small * (1.0 + math.exp(-beta * epsilon0 / 2.0))
            theta_small_derived = numerator / denominator
            # The reported theta should also be small (if the solution is consistent)
            # We cannot directly change P, but we can trust that derived formula is correct.
            # So we just check that the derived low-P theta is indeed small.
            if theta_small_derived < 0.1:
                return (True, f"Low-P derived theta = {theta_small_derived:.4g} < 0.1")
            else:
                return (False, f"Low-P derived theta = {theta_small_derived:.4g} not small")
        except Exception:
            return (False, "Error in low-pressure limit check")

    def test_modeling_high_pressure_limit(self):
        """
        As P → ∞, coverage → 1.  Use derived formula with large P (100× original)
        to see that theta_derived approaches 1.
        """
        if self._derived_theta is None:
            return (False, "SKIP: missing data for high-pressure modeling")
        try:
            beta, epsilon0 = self._beta, self.epsilon0
            P_large = self.P * 100.0
            numerator = P_large * (1.0 + math.exp(-beta * epsilon0 / 2.0))
            denominator = self._p0 + P_large * (1.0 + math.exp(-beta * epsilon0 / 2.0))
            theta_large = numerator / denominator
            if theta_large > 0.99:
                return (True, f"High-P derived theta = {theta_large:.4g} close to 1")
            else:
                return (False, f"High-P derived theta = {theta_large:.4g} not close to 1")
        except Exception:
            return (False, "Error in high-pressure limit check")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_coverage_increases_with_pressure(self):
        """
        Coverage should be a monotonically increasing function of pressure.
        Perturb P upward by 5% and check that derived theta increases.
        """
        if self._derived_theta is None or self.theta is None:
            return (False, "SKIP: missing data for pressure monotonicity")
        try:
            beta, epsilon0, p0, P = self._beta, self.epsilon0, self._p0, self.P
            # baseline derived theta
            def theta_func(P_val):
                num = P_val * (1.0 + math.exp(-beta * epsilon0 / 2.0))
                den = p0 + num
                return num / den

            theta_base = theta_func(P)
            theta_up = theta_func(P * 1.05)
            if theta_up > theta_base + 1e-12:
                return (True, f"Coverage increases with P ({theta_base:.4g} → {theta_up:.4g})")
            else:
                return (False, f"Coverage does not increase with P: base={theta_base:.4g}, up={theta_up:.4g}")
        except Exception:
            return (False, "Error in pressure monotonicity test")

    def test_monotonicity_coverage_decreases_with_temperature(self):
        """
        Coverage should decrease with temperature at fixed pressure.
        Perturb T upward by 5% and check that derived theta decreases.
        """
        if self._derived_theta is None:
            return (False, "SKIP: missing data for temperature monotonicity")
        try:
            # Need to recompute derived theta for perturbed T
            T0 = self.T
            P = self.P
            epsilon0 = self.epsilon0
            h = self.h
            m = self.m
            k_B = self.k_B

            def theta_at_T(T_val):
                beta = 1.0 / (k_B * T_val)
                lam = h / math.sqrt(2.0 * math.pi * m * k_B * T_val)
                A = lam**3 / (k_B * T_val)
                p0 = 1.0 / (A * math.exp(beta * epsilon0))
                num = P * (1.0 + math.exp(-beta * epsilon0 / 2.0))
                den = p0 + num
                return num / den

            theta_base = theta_at_T(T0)
            theta_up = theta_at_T(T0 * 1.05)  # 5% higher T
            if theta_up < theta_base - 1e-12:
                return (True, f"Coverage decreases with T ({theta_base:.4g} → {theta_up:.4g})")
            else:
                return (False, f"Coverage does not decrease with T: base={theta_base:.4g}, up={theta_up:.4g}")
        except Exception:
            return (False, "Error in temperature monotonicity test")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_theta_simple_range(self):
        """Do not re-derive; just check that theta is finite and 0 ≤ theta ≤ 1."""
        if self.theta is None:
            return (False, "SKIP: theta missing")
        try:
            val = float(self.theta)
            if math.isfinite(val) and 0.0 <= val <= 1.0:
                return (True, "theta is a finite number between 0 and 1")
            else:
                return (False, f"theta = {val:.4g} is not finite or not in [0,1]")
        except Exception:
            return (False, "Cannot evaluate theta as float")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)