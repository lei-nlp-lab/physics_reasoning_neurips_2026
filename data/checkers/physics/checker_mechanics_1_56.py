import math

class PhysicsChecker:
    """
    Robust checker for Mariner 4 Hohmann transfer problem.
    Tests interface completeness, physical invariants, modeling consistency,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        res = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results

        # Givens – try attribute first, then results dict
        self.G = getattr(solution_instance, 'G', None) or res.get('G', None)
        self.M = getattr(solution_instance, 'M', None) or res.get('M', None)
        self.R_E = getattr(solution_instance, 'R_E', None) or res.get('R_E', None)
        self.R_M = getattr(solution_instance, 'R_M', None) or res.get('R_M', None)

        # Derived quantities (will be None if givens missing)
        self.mu = None
        if self.G is not None and self.M is not None:
            self.mu = self.G * self.M
        self.a = None
        if self.R_E is not None and self.R_M is not None:
            self.a = (self.R_E + self.R_M) / 2.0

        # Emitted results (from solver)
        self.v_rel_depart = getattr(solution_instance, 'v_rel_depart', None) or res.get('v_rel_depart', None)
        self.t_transfer = getattr(solution_instance, 't_transfer', None) or res.get('t_transfer', None)
        self.v_rel_arrive = getattr(solution_instance, 'v_rel_arrive', None) or res.get('v_rel_arrive', None)

        # Intermediate heliocentric speeds (computed if givens available)
        self.v_E = None
        self.v_M = None
        self.v_p = None
        self.v_a = None
        if self.mu is not None and self.R_E is not None:
            self.v_E = math.sqrt(self.mu / self.R_E)
        if self.mu is not None and self.R_M is not None:
            self.v_M = math.sqrt(self.mu / self.R_M)
        if self.mu is not None and self.R_E is not None and self.a is not None:
            self.v_p = math.sqrt(self.mu * (2.0 / self.R_E - 1.0 / self.a))
        if self.v_p is not None and self.R_E is not None and self.R_M is not None:
            self.v_a = self.v_p * self.R_E / self.R_M

    def run_all_tests(self):
        """Discover and run all test_ methods, return (all_pass, logs)."""
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

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------

    def test_interface_required_keys(self):
        """Ensure all three required result keys exist and are numeric."""
        results = {
            'v_rel_depart': self.v_rel_depart,
            't_transfer': self.t_transfer,
            'v_rel_arrive': self.v_rel_arrive,
        }
        missing = [k for k, v in results.items() if v is None]
        if missing:
            return (False, f"Missing results: {missing}")
        # Check they are numeric (int or float)
        for k, v in results.items():
            if not isinstance(v, (int, float)):
                return (False, f"{k} is not numeric (type {type(v).__name__})")
        return (True, "All required keys present and numeric")

    def test_interface_givens_available(self):
        """Check that essential givens (R_E, R_M, G, M) are not None."""
        missing = []
        for name in ['R_E', 'R_M', 'G', 'M']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "Essential givens available")

    # ----------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ----------------------------------------------------------------

    def test_invariant_energy_at_perihelion(self):
        """Check energy conservation at perihelion: v^2/2 - mu/r = -mu/(2a)."""
        if any(x is None for x in [self.v_p, self.mu, self.R_E, self.a]):
            return (False, "SKIP: insufficient data for energy check")
        lhs = 0.5 * self.v_p ** 2 - self.mu / self.R_E
        rhs = -self.mu / (2.0 * self.a)
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy mismatch: lhs={lhs:.6e}, rhs={rhs:.6e}")
        return (True, f"Energy conserved: lhs ≈ {rhs:.6e}")

    def test_invariant_angular_momentum(self):
        """Check angular momentum: R_E * v_p = R_M * v_a."""
        if any(x is None for x in [self.v_p, self.R_E, self.v_a, self.R_M]):
            return (False, "SKIP: insufficient data for angular momentum check")
        lhs = self.v_p * self.R_E
        rhs = self.v_a * self.R_M
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Angular momentum mismatch: lhs={lhs:.6e}, rhs={rhs:.6e}")
        return (True, "Angular momentum conserved")

    def test_invariant_kepler_transfer_time(self):
        """Check that t_transfer = pi * sqrt(a^3/mu)."""
        if any(x is None for x in [self.t_transfer, self.a, self.mu]):
            return (False, "SKIP: insufficient data for Kepler check")
        expected = math.pi * math.sqrt(self.a ** 3 / self.mu)
        if not math.isclose(self.t_transfer, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Transfer time mismatch: got {self.t_transfer:.6e}, expected {expected:.6e}")
        return (True, "Kepler's third law satisfied")

    # ----------------------------------------------------------------
    # Modeling (trace) tests
    # ----------------------------------------------------------------

    def test_modeling_departure_relation(self):
        """Check v_rel_depart = v_p - v_E."""
        if any(x is None for x in [self.v_rel_depart, self.v_p, self.v_E]):
            return (False, "SKIP: insufficient data for departure relation")
        expected = self.v_p - self.v_E
        if not math.isclose(self.v_rel_depart, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Departure velocity mismatch: got {self.v_rel_depart:.6e}, expected {expected:.6e}")
        return (True, "v_rel_depart = v_p - v_E holds")

    def test_modeling_arrival_relation(self):
        """Check v_rel_arrive = v_M - v_a."""
        if any(x is None for x in [self.v_rel_arrive, self.v_M, self.v_a]):
            return (False, "SKIP: insufficient data for arrival relation")
        expected = self.v_M - self.v_a
        if not math.isclose(self.v_rel_arrive, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Arrival velocity mismatch: got {self.v_rel_arrive:.6e}, expected {expected:.6e}")
        return (True, "v_rel_arrive = v_M - v_a holds")

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------

    def test_monotonicity_transfer_time_increases_with_a(self):
        """Check that increasing semi-major axis a increases transfer time."""
        if any(x is None for x in [self.a, self.mu]):
            return (False, "SKIP: insufficient data for monotonicity of t_transfer")
        # Baseline transfer time from formula
        t_base = math.pi * math.sqrt(self.a ** 3 / self.mu)
        # Perturb a upward by 1%
        a_pert = self.a * 1.01
        t_pert = math.pi * math.sqrt(a_pert ** 3 / self.mu)
        if not t_pert > t_base:
            return (False, f"t_transfer did not increase when a increased: base={t_base:.6e}, perturbed={t_pert:.6e}")
        return (True, "t_transfer increases with semi-major axis (monotonic)")

    def test_monotonicity_v_rel_depart_decreases_with_R_E(self):
        """Check that increasing Earth radius R_E reduces departure relative speed."""
        if any(x is None for x in [self.R_E, self.R_M, self.mu]):
            return (False, "SKIP: insufficient data for v_rel_depart monotonicity")
        # Compute baseline v_rel_depart using formulas
        a_base = (self.R_E + self.R_M) / 2.0
        v_p_base = math.sqrt(self.mu * (2.0/self.R_E - 1.0/a_base))
        v_E_base = math.sqrt(self.mu / self.R_E)
        v_rel_base = v_p_base - v_E_base

        # Perturb R_E upward by 1%
        R_E_pert = self.R_E * 1.01
        a_pert = (R_E_pert + self.R_M) / 2.0
        v_p_pert = math.sqrt(self.mu * (2.0/R_E_pert - 1.0/a_pert))
        v_E_pert = math.sqrt(self.mu / R_E_pert)
        v_rel_pert = v_p_pert - v_E_pert

        if not v_rel_pert < v_rel_base:
            return (False, f"v_rel_depart did not decrease with R_E: base={v_rel_base:.6e}, perturbed={v_rel_pert:.6e}")
        return (True, "v_rel_depart decreases with larger Earth orbit (monotonic)")

    # ----------------------------------------------------------------
    # Weak answer sanity tests (no isclose, no re‑derivation)
    # ----------------------------------------------------------------

    def test_answer_sanity_positivity_and_finite(self):
        """Check that all results are positive and finite."""
        for name, val in [('v_rel_depart', self.v_rel_depart),
                          ('t_transfer', self.t_transfer),
                          ('v_rel_arrive', self.v_rel_arrive)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is non-numeric")
            if not (val > 0):
                return (False, f"{name} is not positive (value={val})")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "All results positive and finite")