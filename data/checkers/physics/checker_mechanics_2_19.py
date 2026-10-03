import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the damped harmonic oscillator Lagrangian problem.
    Expected results keys: eq1 .. eq9 (see problem KB).
    """

    def __init__(self, solution_instance):
        # Extract all result fields safely, defaulting to None
        self.results = getattr(solution_instance, 'results', {})
        self.eq1 = self.results.get('eq1', None)
        self.eq2 = self.results.get('eq2', None)
        self.eq3 = self.results.get('eq3', None)
        self.eq4 = self.results.get('eq4', None)
        self.eq5 = self.results.get('eq5', None)
        self.eq6 = self.results.get('eq6', None)
        self.eq7 = self.results.get('eq7', None)
        self.eq8 = self.results.get('eq8', None)
        self.eq9 = self.results.get('eq9', None)

        # Also extract givens (if present) for assistance
        self.givens = getattr(solution_instance, 'givens', {})
        self.m = self.givens.get('m', None)
        self.k = self.givens.get('k', None)
        self.gamma = self.givens.get('gamma', None)

    # ----------------------------------------------------------------
    # Interface tests – check required keys exist and have expected types
    # ----------------------------------------------------------------
    def test_interface_eq1_exists(self):
        if self.eq1 is None:
            return (False, "eq1 (Lagrange equation for q) is missing")
        if not isinstance(self.eq1, str):
            return (False, "eq1 must be a string")
        return (True, "eq1 present and is string")

    def test_interface_eq2_exists(self):
        if self.eq2 is None:
            return (False, "eq2 (constants of motion for q) is missing")
        if not isinstance(self.eq2, str):
            return (False, "eq2 must be a string")
        return (True, "eq2 present and is string")

    def test_interface_eq3_exists(self):
        if self.eq3 is None:
            return (False, "eq3 (underdamped solution for q) is missing")
        if not isinstance(self.eq3, str):
            return (False, "eq3 must be a string")
        return (True, "eq3 present")

    def test_interface_eq4_exists(self):
        if self.eq4 is None:
            return (False, "eq4 (critically damped solution) is missing")
        if not isinstance(self.eq4, str):
            return (False, "eq4 must be a string")
        return (True, "eq4 present")

    def test_interface_eq5_exists(self):
        if self.eq5 is None:
            return (False, "eq5 (overdamped solution) is missing")
        if not isinstance(self.eq5, str):
            return (False, "eq5 must be a string")
        return (True, "eq5 present")

    def test_interface_eq6_exists(self):
        if self.eq6 is None:
            return (False, "eq6 (Lagrangian in S) is missing")
        if not isinstance(self.eq6, str):
            return (False, "eq6 must be a string")
        return (True, "eq6 present")

    def test_interface_eq7_exists(self):
        if self.eq7 is None:
            return (False, "eq7 (Lagrange equation for S) is missing")
        if not isinstance(self.eq7, str):
            return (False, "eq7 must be a string")
        return (True, "eq7 present")

    def test_interface_eq8_exists(self):
        if self.eq8 is None:
            return (False, "eq8 (constant of motion for S) is missing")
        if not isinstance(self.eq8, str):
            return (False, "eq8 must be a string")
        return (True, "eq8 present")

    def test_interface_eq9_exists(self):
        if self.eq9 is None:
            return (False, "eq9 (relationship between solutions) is missing")
        if not isinstance(self.eq9, str):
            return (False, "eq9 must be a string")
        return (True, "eq9 present")

    # ----------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_euler_lagrange_q(self):
        """
        The Euler-Lagrange equation for q must contain a damping term 
        proportional to gamma and qdot. Check that the string eq1 
        contains 'gamma' near 'qdot' (or 'dq').
        """
        if self.eq1 is None:
            return (False, "SKIP: eq1 not available")
        eq1_lower = self.eq1.lower()
        has_gamma = 'gamma' in eq1_lower
        has_qdot = 'qdot' in eq1_lower or 'dq' in eq1_lower
        if has_gamma and has_qdot:
            return (True, "eq1 contains gamma and a qdot term")
        else:
            return (False, "eq1 does not contain explicit gamma and qdot damping")

    def test_invariant_euler_lagrange_s(self):
        """
        The Lagrange equation for S (eq7) should be a simple harmonic 
        oscillator: no first derivative term, and the coefficient of S 
        is (k - m*gamma^2/4).
        """
        if self.eq7 is None:
            return (False, "SKIP: eq7 not available")
        eq7_lower = self.eq7.lower()
        # No damping term: 'sdot' or 'dq' should not appear as separate terms
        has_first_derivative = ('sdot' in eq7_lower) or ('dq' in eq7_lower)
        if has_first_derivative:
            return (False, "eq7 contains a first‑derivative term; should be absent")
        # Should contain a term with 'sddot' (second derivative) and a term with S
        has_sddot = 'sddot' in eq7_lower or 'ddot' in eq7_lower
        has_s = (' s ' in eq7_lower) or ('*s' in eq7_lower) or (' s' in eq7_lower) or ('s' == eq7_lower.split()[-1])
        if not has_sddot:
            return (False, "eq7 missing second derivative term")
        if not has_s:
            return (False, "eq7 missing S term")
        return (True, "eq7 appears to be a second‑order ODE for S without damping")

    def test_invariant_lagrangian_s_time_independent(self):
        """
        After the point transformation and dropping a total derivative,
        the Lagrangian for S should contain no explicit time dependence.
        Check that the string eq6 does not contain an explicit 'exp(gamma*t)' factor.
        """
        if self.eq6 is None:
            return (False, "SKIP: eq6 not available")
        eq6_lower = self.eq6.lower()
        if 'exp(gamma*t)' in eq6_lower or 'exp(gamma' in eq6_lower:
            return (False, "eq6 still contains an explicit exponential time factor")
        # The Lagrangian should be quadratic in Sdot and S
        if 'sdot' in eq6_lower and ' s' in eq6_lower:
            return (True, "eq6 appears time‑independent and quadratic in Sdot, S")
        return (True, "eq6 passes basic time‑independence check")

    def test_invariant_constant_of_motion_s(self):
        """
        For the S system, the conserved quantity should be of the form
        Sdot^2 + omega^2 S^2 = constant (eq8). Check that the string
        contains 'sdot^2' (or similar) and 's^2'.
        """
        if self.eq8 is None:
            return (False, "SKIP: eq8 not available")
        eq8_lower = self.eq8.lower()
        has_sdot2 = 'sdot^2' in eq8_lower or 'sdot**2' in eq8_lower
        has_s2 = 's^2' in eq8_lower or 's**2' in eq8_lower
        if has_sdot2 and has_s2:
            return (True, "eq8 contains quadratic terms in Sdot and S")
        else:
            return (False, "eq8 does not appear as a sum of squares")

    # ----------------------------------------------------------------
    # Modeling / trace checks – consistency with explanation and transformation
    # ----------------------------------------------------------------
    def test_trace_transformation_q_to_s(self):
        """
        Verify that the Lagrangian in S (eq6) correctly results from
        the substitution q = exp(-gamma t/2) S. Check that eq6 contains
        a term '- gamma S/2' inside the kinetic part.
        """
        if self.eq6 is None:
            return (False, "SKIP: eq6 not available")
        eq6_lower = self.eq6.lower()
        if 'gamma' in eq6_lower and 's/2' in eq6_lower:
            return (True, "eq6 contains the expected gamma coupling term")
        if '*gamma*s' in eq6_lower or 'gamma*s' in eq6_lower:
            return (True, "eq6 contains gamma*S term")
        return (False, "eq6 missing the expected (Sdot - gamma S/2) structure")

    def test_trace_cases_present(self):
        """
        The three cases (underdamped, critically damped, overdamped) should
        be represented by eq3, eq4, eq5 respectively. Check that at least
        one of them mentions 'cos' or 'exp' as appropriate.
        """
        if self.eq3 is None or self.eq4 is None or self.eq5 is None:
            return (False, "SKIP: one of eq3‑eq5 missing")
        # eq3 (underdamped) should contain 'cos' and 'sin' and 'exp'
        eq3_lower = self.eq3.lower()
        under = ('cos' in eq3_lower and 'sin' in eq3_lower) or ('exp' in eq3_lower)
        # eq4 (critically damped) should have exp but no cos/sin (unless degenerate)
        eq4_lower = self.eq4.lower()
        critical = ('exp' in eq4_lower) and not ('cos' in eq4_lower or 'sin' in eq4_lower)
        # eq5 (overdamped) should have two exponentials (exp(b*t) and exp(-b*t))
        eq5_lower = self.eq5.lower()
        over = ('exp(b*t)' in eq5_lower or 'exp( b*t )' in eq5_lower) and \
               ('exp(-b*t)' in eq5_lower or 'exp( -b*t )' in eq5_lower)
        if under and critical and over:
            return (True, "All three cases (under‑, critical‑, over‑damped) are represented")
        else:
            return (False, "One or more of the case solutions do not match expected form")

    # ----------------------------------------------------------------
    # Monotonicity test – perturb an input and check response
    # ----------------------------------------------------------------
    def test_monotonicity_damping_increases_with_gamma(self):
        """
        For the damped harmonic oscillator equation (eq1), the damping
        rate (real part of characteristic root) is -gamma/2.  This
        becomes more negative (stronger damping) if gamma increases.
        Perturb gamma by +1% and verify that the damping rate changes
        by -0.5 * delta_gamma < 0.
        """
        # We need numeric gamma from givens or fallback to symbolic analysis
        if self.gamma is not None:
            gamma_val = float(self.gamma)
            if gamma_val <= 0:
                return (False, "gamma must be positive")
            delta_gamma = 0.01 * gamma_val
            baseline_rate = -gamma_val / 2.0
            perturbed_rate = -(gamma_val + delta_gamma) / 2.0
            # The perturbed rate should be smaller (more negative)
            if perturbed_rate < baseline_rate:
                return (True, "damping rate decreases (more negative) when gamma increases")
            else:
                return (False, "damping rate did not decrease as gamma increased")
        else:
            # If gamma is not given, do symbolic check
            gamma_sym = sp.symbols('gamma', positive=True)
            baseline_rate_sym = -gamma_sym / 2
            delta = sp.Rational(1, 100) * gamma_sym   # +1% perturbation
            perturbed_rate_sym = -(gamma_sym + delta) / 2
            diff_sym = perturbed_rate_sym - baseline_rate_sym
            # The difference should be negative for positive gamma
            if diff_sym.is_negative or diff_sym == -3*gamma_sym/200:
                return (True, "symbolic analysis confirms damping rate decreases with gamma")
            else:
                return (False, "symbolic check failed")

    # ----------------------------------------------------------------
    # Weak answer sanity – only presence, no hardcoded target values
    # ----------------------------------------------------------------
    def test_answer_sanity_no_contradiction(self):
        """
        Check that the relationship statement (eq9) is not empty and
        expresses some connection ( not just 'None' or empty).
        """
        if self.eq9 is None:
            return (False, "SKIP: eq9 not available")
        if len(self.eq9.strip()) < 5:
            return (False, "eq9 is too short; should contain a meaningful statement")
        if 'identical physical situations' in self.eq9.lower() or 'same physical' in self.eq9.lower():
            return (True, "eq9 mentions the physical equivalence")
        # broad pass: it's a non‑trivial string
        return (True, "eq9 is non‑trivial")

    # ----------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------
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