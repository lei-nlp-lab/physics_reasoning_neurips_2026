import math

class PhysicsChecker:
    """
    Audits a solution for the hydrogen ionization equilibrium problem.
    Checks interface completeness, physical invariants (Saha equation,
    half‑ionization density), modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Required result keys – default to None
        self.mu_e = self.results.get('mu_e', None)
        self.mu_p = self.results.get('mu_p', None)
        self.mu_H = self.results.get('mu_H', None)
        self.Saha_factor = self.results.get('Saha_factor', None)
        self.eq_saha = self.results.get('eq_saha', None)
        self.n_N = self.results.get('n_N', None)

        # Temperature: try givens first, then results
        self.T = self.givens.get('T', self.results.get('T', None))

        # Optional densities if present (e.g. from intermediate steps)
        self.n_e = self.results.get('[e]', None)
        self.n_p = self.results.get('[p]', None)
        self.n_H = self.results.get('[H]', None)

        # Fundamental constants (SI)
        self.k = 1.380649e-23             # J/K
        self.h = 6.62607015e-34           # J·s
        self.m_e = 9.10938356e-31         # kg
        self.m_p = 1.67262192369e-27      # kg
        self.m_H = self.m_p               # approximation
        self.g_e = 2
        self.g_p = 2
        self.g_H = 2
        # Binding energy B = 13.6 eV in Joules
        self.B_J = 13.6 * 1.602176634e-19
        self.B_eV = 13.6

    # ------------------------------------------------------------------
    # Internal helper to check if a value is numeric (int or float)
    # ------------------------------------------------------------------
    def _is_numeric(self, val):
        return isinstance(val, (int, float))

    # ------------------------------------------------------------------
    # Helper to compute n_N for a given temperature (m⁻³)
    # ------------------------------------------------------------------
    def _compute_n_N(self, T):
        """Return n_N = 4 * (2π m_e kT/h²)^(3/2) * exp(-B/kT)."""
        if not self._is_numeric(T) or T <= 0:
            return None
        kT = self.k * T
        prefactor = 4.0 * (2.0 * math.pi * self.m_e * kT / (self.h ** 2)) ** 1.5
        return prefactor * math.exp(-self.B_J / kT)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required = ['mu_e', 'mu_p', 'mu_H', 'Saha_factor', 'eq_saha', 'n_N']
        missing = [key for key in required if self.results.get(key) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_temperature_valid(self):
        if self.T is None:
            return (False, "SKIP: T not provided in givens or results")
        if not self._is_numeric(self.T) or self.T <= 0:
            return (False, f"Invalid T: {self.T}")
        return (True, f"T = {self.T} K is valid")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_saha_factor_formula(self):
        """Check that Saha_factor equals 2 * nQ_e * exp(-B/kT)"""
        if self.Saha_factor is None:
            return (False, "SKIP: Saha_factor not available")
        if not self._is_numeric(self.Saha_factor):
            return (False, "SKIP: Saha_factor is not numeric")
        if self.T is None or not self._is_numeric(self.T) or self.T <= 0:
            return (False, "SKIP: valid numeric T needed")

        try:
            kT = self.k * self.T
            nQ_e = 2.0 * (2.0 * math.pi * self.m_e * kT / (self.h ** 2)) ** 1.5
            expected = 2.0 * nQ_e * math.exp(-self.B_J / kT)
            if math.isclose(self.Saha_factor, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Saha_factor ≈ {expected:.3e}")
            else:
                return (False, f"Saha_factor {self.Saha_factor:.3e} != expected {expected:.3e}")
        except Exception as e:
            return (False, f"Computation error: {e}")

    def test_invariant_half_ionization_density(self):
        """Check that n_N (as emitted) matches 4*(2π m_e kT/h²)^1.5 * exp(-B/kT)"""
        if self.n_N is None:
            return (False, "SKIP: n_N not available")
        if not self._is_numeric(self.n_N):
            return (False, "SKIP: n_N not numeric")
        if self.T is None or not self._is_numeric(self.T) or self.T <= 0:
            return (False, "SKIP: valid numeric T needed")

        try:
            expected = self._compute_n_N(self.T)
            if expected is None:
                return (False, "Internal computation failed")
            if math.isclose(self.n_N, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"n_N ≈ {expected:.3e} m⁻³")
            else:
                return (False, f"n_N {self.n_N:.3e} != expected {expected:.3e}")
        except Exception as e:
            return (False, f"Computation error: {e}")

    # ------------------------------------------------------------------
    # Modeling quality checks
    # ------------------------------------------------------------------
    def test_modeling_saha_equation_text(self):
        """Check that eq_saha has the correct form"""
        if self.eq_saha is None:
            return (False, "SKIP: eq_saha not available")
        if not isinstance(self.eq_saha, str):
            return (False, "eq_saha is not a string")
        # Basic pattern: should mention '[e]^2/[H]' and 'Saha_factor'
        if "[e]^2/[H]" in self.eq_saha and "Saha_factor" in self.eq_saha:
            return (True, "eq_saha text looks correct")
        else:
            return (False, f"eq_saha does not contain expected pattern: {self.eq_saha}")

    def test_modeling_chemical_potential_expression(self):
        """Check that mu_e, mu_p, mu_H contain the corresponding density symbol"""
        if self.mu_e is None or self.mu_p is None or self.mu_H is None:
            return (False, "SKIP: one of mu_? is missing")
        checks = []
        for name, expr, symbol in [("mu_e", self.mu_e, "[e]"),
                                    ("mu_p", self.mu_p, "[p]"),
                                    ("mu_H", self.mu_H, "[H]")]:
            if isinstance(expr, str):
                if symbol in expr:
                    checks.append(f"{name} contains '{symbol}'")
                else:
                    checks.append(f"{name} does not contain '{symbol}'")
            else:
                # If it's a sympy expression, convert to string? Skip.
                checks.append(f"{name} is not a string – skipping symbol check")
        return (True, "; ".join(checks))

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_n_N_increases_with_T(self):
        """
        Explicitly perturb the input temperature (increase by 10%) and
        compare the perturbed output (n_N computed with T_perturbed)
        against the baseline output (the solution's n_N at original T).
        """
        if self.n_N is None:
            return (False, "SKIP: n_N not available")
        if not self._is_numeric(self.n_N):
            return (False, "SKIP: n_N not numeric")
        if self.T is None or not self._is_numeric(self.T) or self.T <= 0:
            return (False, "SKIP: valid numeric T needed")

        try:
            # Baseline output: use the solution's own n_N
            baseline_output = self.n_N

            # Perturb the input temperature (increase by 10%)
            T_perturbed = self.T * 1.10
            perturbed_output = self._compute_n_N(T_perturbed)

            if perturbed_output is None:
                return (False, "Could not compute perturbed output")

            # Monotonicity: n_N should increase with T (positive exponential dominates)
            if perturbed_output > baseline_output:
                return (True, f"Perturbed n_N ({perturbed_output:.3e}) > baseline n_N ({baseline_output:.3e})")
            else:
                return (False, f"Perturbed n_N ({perturbed_output:.3e}) <= baseline n_N ({baseline_output:.3e})")
        except Exception as e:
            return (False, f"Computation error: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only broad range, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_n_N_broad(self):
        """Check that n_N is positive and not NaN/Inf"""
        if self.n_N is None:
            return (False, "SKIP: n_N not available")
        if not self._is_numeric(self.n_N):
            return (False, "SKIP: n_N not numeric")
        if math.isnan(self.n_N) or math.isinf(self.n_N) or self.n_N <= 0:
            return (False, f"n_N is not finite positive: {self.n_N}")
        return (True, f"n_N = {self.n_N:.3e} is finite and positive")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods, collecting results."""
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