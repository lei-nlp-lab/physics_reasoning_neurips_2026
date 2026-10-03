import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for capacitor problem: three concentric spherical shells.
    Checks interface, charge conservation, equipotential, Gauss law,
    modeling formulas, and monotonicity constraints.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.a = givens.get('a', None)
        self.b = givens.get('b', None)
        self.d = givens.get('d', None)
        self.Q_B = givens.get('Q_B', None)
        self.epsilon0 = givens.get('epsilon0', None)  # may be None or symbolic

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.Q1 = results.get('Q1', None)
        self.C = results.get('C', None)
        self.Q_mi = results.get('Q_mi', None)
        self.Q_mo = results.get('Q_mo', None)
        self.Q_B_result = results.get('Q_B', None)  # could be redundant

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist."""
        required = ['Q1', 'C', 'Q_mi', 'Q_mo']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing keys in results: {missing}")
        return (True, "All required keys present")

    # ------------------------------------------------------------------
    # Invariant checks (physics laws)
    # ------------------------------------------------------------------
    def test_invariant_charge_conservation_middle(self):
        """Check Q_B = Q_mi + Q_mo"""
        if any(v is None for v in [self.Q_B, self.Q_mi, self.Q_mo]):
            return (False, "SKIP: missing charge data")
        if math.isclose(self.Q_B, self.Q_mi + self.Q_mo, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Charge conservation on middle shell holds")
        else:
            return (False, f"Q_B={self.Q_B} != Q_mi+Q_mo={self.Q_mi+self.Q_mo}")

    def test_invariant_gauss_inner_surface(self):
        """Check that inner surface charge of middle shell = -Q1 (field zero inside middle)"""
        if any(v is None for v in [self.Q1, self.Q_mi]):
            return (False, "SKIP: missing Q1 or Q_mi")
        if math.isclose(-self.Q1, self.Q_mi, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Gauss law: Q_mi = -Q1 holds")
        else:
            return (False, f"Q_mi={self.Q_mi} != -Q1={-self.Q1}")

    def test_invariant_equipotential_inner_outer(self):
        """Check V(a) = V(d) via Q1*(1/a - 1/b) = -Q_mo*(1/b - 1/d)"""
        if any(v is None for v in [self.a, self.b, self.d, self.Q1, self.Q_mo]):
            return (False, "SKIP: missing givens or Q variables")
        if self.a <= 0 or self.b <= 0 or self.d <= 0:
            return (False, "Radii must be positive")
        lhs = self.Q1 * (1/self.a - 1/self.b)
        rhs = -self.Q_mo * (1/self.b - 1/self.d)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Equipotential condition V(a)=V(d) satisfied")
        else:
            return (False, f"LHS={lhs} != RHS={rhs}")

    def test_invariant_charge_relation_mo(self):
        """Check that Q_mo = Q_B + Q1 (from conservation and Gauss)"""
        if any(v is None for v in [self.Q_B, self.Q1, self.Q_mo]):
            return (False, "SKIP: missing charges")
        expected = self.Q_B + self.Q1
        if math.isclose(self.Q_mo, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Q_mo = Q_B + Q1 holds")
        else:
            return (False, f"Q_mo={self.Q_mo} != Q_B+Q1={expected}")

    # ------------------------------------------------------------------
    # Modeling checks (derived formulas from physics laws)
    # ------------------------------------------------------------------
    def test_modeling_capacitance_formula(self):
        """Check that C matches 4πϵ0 a d/(d-a) (re-derived from givens)."""
        if self.C is None:
            return (False, "SKIP: no C in results")
        if any(v is None for v in [self.a, self.d, self.epsilon0]):
            return (False, "SKIP: missing a, d, or epsilon0 for numeric check")
        if not isinstance(self.epsilon0, (int, float)):
            return (False, "SKIP: epsilon0 is not numeric")
        expected = 4 * math.pi * self.epsilon0 * self.a * self.d / (self.d - self.a)
        if math.isclose(self.C, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Capacitance matches formula: C ≈ {expected}")
        else:
            return (False, f"C={self.C} != expected={expected}")

    def test_modeling_inner_charge_expression(self):
        """Check that Q1 = -a*(d-b)/(b*(d-a))*Q_B (from equipotential derivation)."""
        if any(v is None for v in [self.a, self.b, self.d, self.Q_B, self.Q1]):
            return (False, "SKIP: missing data for Q1 check")
        # Derive expected from the equipotential condition
        expected = -self.a * (self.d - self.b) / (self.b * (self.d - self.a)) * self.Q_B
        if math.isclose(self.Q1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Q1 expression matches derived formula")
        else:
            return (False, f"Q1={self.Q1} != derived={expected}")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_capacitance_increases_with_a(self):
        """Check that increasing a (with fixed d) increases C."""
        if any(v is None for v in [self.a, self.d, self.C, self.epsilon0]):
            return (False, "SKIP: missing parameters")
        if not isinstance(self.epsilon0, (int, float)):
            return (False, "SKIP: epsilon0 not numeric")
        a_new = self.a * 1.01
        if a_new >= self.d:
            return (False, "Perturbed a exceeds d")
        C_new = 4 * math.pi * self.epsilon0 * a_new * self.d / (self.d - a_new)
        if C_new > self.C:
            return (True, "Capacitance increases with increasing a (monotonicity holds)")
        else:
            return (False, "Capacitance did not increase when a increased")

    def test_monotonicity_capacitance_increases_with_d(self):
        """Check that increasing d (with fixed a) increases C."""
        if any(v is None for v in [self.a, self.d, self.C, self.epsilon0]):
            return (False, "SKIP: missing parameters")
        if not isinstance(self.epsilon0, (int, float)):
            return (False, "SKIP: epsilon0 not numeric")
        d_new = self.d * 1.01
        if d_new <= self.a:
            return (False, "Perturbed d not larger than a")
        C_new = 4 * math.pi * self.epsilon0 * self.a * d_new / (d_new - self.a)
        if C_new > self.C:
            return (True, "Capacitance increases with increasing d")
        else:
            return (False, "Capacitance did not increase when d increased")

    # ------------------------------------------------------------------
    # Single weak answer sanity check (no hardcoded target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_general(self):
        """Check that capacitance is positive/finite and charges have plausible signs."""
        issues = []
        if self.C is not None:
            if not math.isfinite(self.C) or self.C <= 0:
                issues.append(f"C is non-positive or non-finite: {self.C}")
        if all(v is not None for v in [self.Q_B, self.Q_mi, self.Q_mo]):
            if self.Q_B != 0:
                # Q_mi should have opposite sign to Q_B (from formula)
                if (self.Q_mi / self.Q_B) > 0:
                    issues.append("Q_mi should have opposite sign to Q_B")
                # Q_mo should have same sign as Q_B
                if (self.Q_mo / self.Q_B) < 0:
                    issues.append("Q_mo should have same sign as Q_B")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All general sanity checks passed")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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