import math
import traceback

class PhysicsChecker:
    """
    Robust physics checker for the photon-particle scattering problem.
    Extracts inputs and results safely, then runs a suite of targeted tests.
    """

    def __init__(self, solution_instance):
        # Givens: photon momentum p, particle rest mass m, speed of light c
        givens = getattr(solution_instance, 'givens', None) or {}
        self.p = givens.get('p', None)
        self.m = givens.get('m', None)
        self.c = givens.get('c', None)

        # Results from the solution
        results = getattr(solution_instance, 'results', None) or {}
        self.E_cm = results.get('E_cm', None)
        self.q = results.get('q', None)
        self.p_prime = results.get('p_prime', None)

        # Explanation steps (list of step dicts) – used to decide what to audit
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # -------------------------------------------------------------------------
    # Interface tests: required keys exist and are of acceptable type
    # -------------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Check that all expected keys are present in results."""
        required = ['E_cm', 'q', 'p_prime']
        if self.E_cm is None or self.q is None or self.p_prime is None:
            missing = [k for k in required if getattr(self, k) is None]
            return (False, f"Missing result key(s): {missing}")
        return (True, "All required result keys present")

    def test_interface_givens_types(self):
        """Check that givens are numeric (float/int)."""
        for name, val in [('p', self.p), ('m', self.m), ('c', self.c)]:
            if val is None:
                return (False, f"Given '{name}' is missing")
            if not isinstance(val, (int, float)):
                return (False, f"Given '{name}' is not numeric")
        return (True, "Givens are numeric")

    def test_interface_results_types(self):
        """Check that results are numeric (or at least not None)."""
        for name, val in [('E_cm', self.E_cm), ('q', self.q), ('p_prime', self.p_prime)]:
            if val is None:
                return (False, f"Result '{name}' is None")
            if not isinstance(val, (int, float)):
                return (False, f"Result '{name}' is not numeric")
        return (True, "Results are numeric")

    # -------------------------------------------------------------------------
    # Invariant tests: theorem-based physical consistency (use math.isclose)
    # -------------------------------------------------------------------------
    def test_invariant_mass_squared(self):
        """s = m^2*c^4 + 2*p*m*c^3  and  E_cm^2 = s."""
        if any(x is None for x in (self.p, self.m, self.c, self.E_cm)):
            return (False, "SKIP: missing givens or E_cm")
        s = self.m**2 * self.c**4 + 2 * self.p * self.m * self.c**3
        if not math.isclose(self.E_cm**2, s, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"E_cm^2 = {self.E_cm**2:.6e} vs s = {s:.6e}")
        return (True, "Invariant mass squared holds")

    def test_invariant_cm_momentum_formula(self):
        """q = p*m*c^2 / E_cm  (from pseudocode step)."""
        if any(x is None for x in (self.p, self.m, self.c, self.E_cm, self.q)):
            return (False, "SKIP: missing givens or E_cm/q")
        expected_q = self.p * self.m * self.c**2 / self.E_cm
        if not math.isclose(self.q, expected_q, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"q = {self.q:.6e} vs expected {expected_q:.6e}")
        return (True, "CM momentum q derived correctly")

    def test_invariant_cm_energy_relation(self):
        """In CM: E_cm = q*c + sqrt(q^2*c^2 + m^2*c^4)."""
        if any(x is None for x in (self.q, self.m, self.c, self.E_cm)):
            return (False, "SKIP: missing results or givens")
        photon_energy = self.q * self.c
        particle_energy = math.sqrt(self.q**2 * self.c**2 + self.m**2 * self.c**4)
        total = photon_energy + particle_energy
        if not math.isclose(total, self.E_cm, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"E_cm from q = {total:.6e} vs reported {self.E_cm:.6e}")
        return (True, "CM energy relation satisfied")

    # -------------------------------------------------------------------------
    # Modeling / trace tests: check one specific relation from explanation steps
    # -------------------------------------------------------------------------
    def test_trace_p_prime_conservation(self):
        """Check energy and momentum conservation in lab for backward scattering."""
        if any(x is None for x in (self.p, self.m, self.c, self.p_prime)):
            return (False, "SKIP: missing givens or p_prime")
        # Momentum: p = -p' + P  =>  P = p + p'  (magnitude)
        P = self.p + self.p_prime
        # Energy: p*c + m*c^2 = p'*c + sqrt(P^2*c^2 + m^2*c^4)
        lhs = self.p * self.c + self.m * self.c**2
        rhs = self.p_prime * self.c + math.sqrt(P**2 * self.c**2 + self.m**2 * self.c**4)
        if not math.isclose(lhs, rhs, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"Energy conservation fails: LHS {lhs:.6e}, RHS {rhs:.6e}")
        return (True, "Backward scattering energy-momentum conserved")

    def test_trace_p_prime_formula(self):
        """p' = p*m*c / (2p + m*c)."""
        if any(x is None for x in (self.p, self.m, self.c, self.p_prime)):
            return (False, "SKIP: missing givens or p_prime")
        expected = self.p * self.m * self.c / (2 * self.p + self.m * self.c)
        if not math.isclose(self.p_prime, expected, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"p_prime = {self.p_prime:.6e} vs formula {expected:.6e}")
        return (True, "p_prime formula matches solution")

    # -------------------------------------------------------------------------
    # Monotonicity tests: perturb one input and check trend
    # -------------------------------------------------------------------------
    def test_monotonicity_E_cm_increases_with_p(self):
        """Increasing p should increase E_cm (with m,c fixed)."""
        if any(x is None for x in (self.p, self.m, self.c, self.E_cm)):
            return (False, "SKIP: missing data")
        # Perturb p upward by 10%
        p2 = self.p * 1.10
        E_cm_pert = math.sqrt(self.m**2 * self.c**4 + 2 * p2 * self.m * self.c**3)
        if E_cm_pert <= self.E_cm:
            return (False, "E_cm did not increase when p increased")
        return (True, "E_cm monotonic in p")

    def test_monotonicity_q_increases_with_p(self):
        """Increasing p should increase q (with m,c fixed)."""
        if any(x is None for x in (self.p, self.m, self.c, self.q)):
            return (False, "SKIP: missing data")
        p2 = self.p * 1.10
        s2 = self.m**2 * self.c**4 + 2 * p2 * self.m * self.c**3
        E_cm2 = math.sqrt(s2)
        q2 = p2 * self.m * self.c**2 / E_cm2
        if q2 <= self.q:
            return (False, "q did not increase when p increased")
        return (True, "q monotonic in p")

    def test_monotonicity_p_prime_increases_with_p(self):
        """Increasing p should increase p_prime (with m,c fixed)."""
        if any(x is None for x in (self.p, self.m, self.c, self.p_prime)):
            return (False, "SKIP: missing data")
        p2 = self.p * 1.10
        p_prime2 = p2 * self.m * self.c / (2 * p2 + self.m * self.c)
        if p_prime2 <= self.p_prime:
            return (False, "p_prime did not increase when p increased")
        return (True, "p_prime monotonic in p")

    # -------------------------------------------------------------------------
    # Weak answer sanity (at most one; cannot use math.isclose or re-derive)
    # -------------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Final answers should be positive and finite."""
        checks = [
            ('E_cm', self.E_cm, lambda x: x > 0 and math.isfinite(x)),
            ('q', self.q, lambda x: x > 0 and math.isfinite(x)),
            ('p_prime', self.p_prime, lambda x: x > 0 and math.isfinite(x)),
        ]
        for name, val, cond in checks:
            if val is None:
                return (False, f"Result '{name}' is None")
            if not cond(val):
                return (False, f"Result '{name}' fails sanity check (value: {val})")
        return (True, "All results positive and finite")

    # -------------------------------------------------------------------------
    # Main runner: discover and execute all test_ methods
    # -------------------------------------------------------------------------
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