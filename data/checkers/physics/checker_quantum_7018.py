import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely, using None as default.
        # Try to get from solution_instance attributes, then from .givens dict, then from .results.
        self.m = None
        self.a = None
        self.b = None
        self.c = None
        self.A = None
        self.hbar = None
        self.pi = None

        # Try to get givens from the instance
        if hasattr(solution_instance, 'm'):
            self.m = solution_instance.m
        if hasattr(solution_instance, 'a'):
            self.a = solution_instance.a
        if hasattr(solution_instance, 'b'):
            self.b = solution_instance.b
        if hasattr(solution_instance, 'c'):
            self.c = solution_instance.c
        if hasattr(solution_instance, 'A'):
            self.A = solution_instance.A
        if hasattr(solution_instance, 'hbar'):
            self.hbar = solution_instance.hbar
        if hasattr(solution_instance, 'pi'):
            self.pi = solution_instance.pi

        # Also check givens dict if exists
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            for key in ['m', 'a', 'b', 'c', 'A', 'hbar', 'pi']:
                if getattr(self, key) is None and key in solution_instance.givens:
                    setattr(self, key, solution_instance.givens[key])

        # Results
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract expected result keys
        self.E_total = self.results.get('E_total', None)
        self.E_total_s = self.results.get('E_total_s', None)
        self.E_total_anti = self.results.get('E_total_anti', None)

        # Precompute numeric forms of constants if possible
        self._numeric_m = None
        self._numeric_a = None
        self._numeric_b = None
        self._numeric_c = None
        self._numeric_A = None
        self._numeric_hbar = None
        self._numeric_pi = None

        self._try_numeric_convert()

    def _try_numeric_convert(self):
        """Attempt to convert symbolic constants to floats."""
        for attr in ['m', 'a', 'b', 'c', 'A', 'hbar', 'pi']:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    numeric_val = float(val)
                    setattr(self, f'_numeric_{attr}', numeric_val)
                except (TypeError, ValueError):
                    setattr(self, f'_numeric_{attr}', None)
            else:
                setattr(self, f'_numeric_{attr}', None)

    def _compute_E0(self):
        """Compute unperturbed ground state energy for one particle in (1,1,1)."""
        if any(getattr(self, f'_numeric_{x}') is None for x in ['hbar', 'pi', 'm', 'a', 'b', 'c']):
            return None
        hb = self._numeric_hbar
        pi = self._numeric_pi
        m = self._numeric_m
        a = self._numeric_a
        b = self._numeric_b
        c = self._numeric_c
        # E_{1,1,1} = (hbar^2 * pi^2)/(2m) * (1/a^2 + 1/b^2 + 1/c^2)
        # For two particles in product state, total E0 = 2 * that.
        # Wait: problem statement: unperturbed Hamiltonian is sum of two single-particle Hamiltonians.
        # The ground state for two distinguishable/bosons is both in (1,1,1), so total unperturbed energy = 2 * E_{1,1,1}.
        # But the target answers show E0 for two particles: (hbar^2 pi^2 / m)*(1/a^2+1/b^2+1/c^2). Let's check: factor 2 missing?
        # Re-reading: "E0 = (ħ²π²/m)(1/a²+1/b²+1/c²)". That's for two particles? Actually one particle energy is (ħ²π²/(2m))(...). For two particles it's (ħ²π²/m)(...). So factor 2 is correct.
        # So E0_two = (hbar^2 * pi^2 / m) * (1/a^2 + 1/b^2 + 1/c^2)
        E0 = (hb**2 * pi**2 / m) * (1/a**2 + 1/b**2 + 1/c**2)
        return E0

    def _compute_deltaE(self):
        """Compute first-order energy shift for product ground state."""
        if any(getattr(self, f'_numeric_{x}') is None for x in ['A', 'a', 'b', 'c']):
            return None
        A = self._numeric_A
        a = self._numeric_a
        b = self._numeric_b
        c = self._numeric_c
        deltaE = 27 * A / (8 * a * b * c)
        return deltaE

    def _compute_E0_anti(self):
        """Compute unperturbed energy for antisymmetric state: one in (1,1,1), one in (2,1,1)."""
        if any(getattr(self, f'_numeric_{x}') is None for x in ['hbar', 'pi', 'm', 'a', 'b', 'c']):
            return None
        hb = self._numeric_hbar
        pi = self._numeric_pi
        m = self._numeric_m
        a = self._numeric_a
        b = self._numeric_b
        c = self._numeric_c
        # E_{2,1,1} = (hbar^2 pi^2)/(2m) * (4/a^2 + 1/b^2 + 1/c^2)
        # Sum with E_{1,1,1} gives (hbar^2 pi^2)/(2m) * (5/a^2 + 2/b^2 + 2/c^2) ???
        # Wait target answer: E'_A = (ħ²π²/m) * (5/(2a²) + 1/b² + 1/c²)
        # Let's derive: E_{1,1,1} + E_{2,1,1} = (ħ²π²/(2m))[(1+4)/a² + (1+1)/b² + (1+1)/c²] = (ħ²π²/(2m))[5/a² + 2/b² + 2/c²].
        # That equals (ħ²π²/m)[5/(2a²) + 1/b² + 1/c²]. Yes, matches target.
        E0_anti = (hb**2 * pi**2 / m) * (5/(2*a**2) + 1/b**2 + 1/c**2)
        return E0_anti

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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that result keys exist and are numeric."""
        missing = []
        for key, val in [('E_total', self.E_total), ('E_total_s', self.E_total_s), ('E_total_anti', self.E_total_anti)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        # Check numeric type (allow sympy expressions that are numeric)
        for key, val in [('E_total', self.E_total), ('E_total_s', self.E_total_s), ('E_total_anti', self.E_total_anti)]:
            if not isinstance(val, (int, float, complex)):
                # Try to convert
                try:
                    float(val)
                except (TypeError, ValueError):
                    return (False, f"Key '{key}' value not numeric: {val}")
        return (True, "All required result keys present and numeric")

    def test_interface_givens_present(self):
        """Check that essential givens are present."""
        missing = []
        for name in ['m', 'a', 'b', 'c', 'A', 'hbar', 'pi']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_perturbation_shift_distinguishable(self):
        """Check that for distinguishable particles, E_total = E0 + deltaE."""
        if self.E_total is None:
            return (False, "SKIP: E_total missing")
        E0 = self._compute_E0()
        deltaE = self._compute_deltaE()
        if E0 is None or deltaE is None:
            return (False, "SKIP: cannot compute E0 or deltaE (missing numeric givens)")
        expected = E0 + deltaE
        # Use math.isclose with tolerance 1e-2 (physics tolerance)
        if math.isclose(self.E_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total matches E0+deltaE (expected {expected}, got {self.E_total})")
        else:
            return (False, f"E_total {self.E_total} does not match expected {expected}")

    def test_invariant_perturbation_shift_bosons(self):
        """Check that for bosons, E_total_s = E0 + deltaE (same as distinguishable)."""
        if self.E_total_s is None:
            return (False, "SKIP: E_total_s missing")
        E0 = self._compute_E0()
        deltaE = self._compute_deltaE()
        if E0 is None or deltaE is None:
            return (False, "SKIP: cannot compute E0 or deltaE")
        expected = E0 + deltaE
        if math.isclose(self.E_total_s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total_s matches E0+deltaE (expected {expected}, got {self.E_total_s})")
        else:
            return (False, f"E_total_s {self.E_total_s} does not match expected {expected}")

    def test_invariant_antisymmetric_deltaE_zero(self):
        """Check that for fermions, E_total_anti = E0_anti (deltaE=0)."""
        if self.E_total_anti is None:
            return (False, "SKIP: E_total_anti missing")
        E0_anti = self._compute_E0_anti()
        if E0_anti is None:
            return (False, "SKIP: cannot compute E0_anti")
        if math.isclose(self.E_total_anti, E0_anti, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total_anti matches E0_anti (expected {E0_anti}, got {self.E_total_anti})")
        else:
            return (False, f"E_total_anti {self.E_total_anti} does not match E0_anti {E0_anti}")

    def test_invariant_antisymmetric_energy_positive(self):
        """Check that antisymmetric energy is positive (for positive givens)."""
        if self.E_total_anti is None:
            return (False, "SKIP: E_total_anti missing")
        try:
            E_val = float(self.E_total_anti)
        except (TypeError, ValueError):
            return (False, "E_total_anti not numeric")
        if E_val > 0:
            return (True, f"E_total_anti = {E_val} > 0")
        else:
            return (False, f"E_total_anti = {E_val} is not positive (if A positive, problem may require sign check)")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------
    def test_modeling_bosons_equals_distinguishable(self):
        """Check that E_total_s equals E_total (both product ground state)."""
        if self.E_total is None or self.E_total_s is None:
            return (False, "SKIP: one of E_total or E_total_s missing")
        if math.isclose(self.E_total, self.E_total_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total_s {self.E_total_s} equals E_total {self.E_total}")
        else:
            return (False, f"E_total_s {self.E_total_s} differs from E_total {self.E_total}")

    def test_modeling_antisymmetric_higher_than_ground(self):
        """Check that antisymmetric energy is greater than the product ground energy (since 5/(2a^2) > 1/a^2)."""
        if self.E_total is None or self.E_total_anti is None:
            return (False, "SKIP: missing energies")
        try:
            E_dist = float(self.E_total)
            E_anti = float(self.E_total_anti)
        except (TypeError, ValueError):
            return (False, "Cannot convert energies to float")
        # For typical positive a,b,c, hbar, the shift deltaE might be positive or negative.
        # But unperturbed part for anti is larger: E0_anti > E0 (since 5/(2a^2) > 1/a^2).
        # However deltaE for distinguishable could be large negative if A negative. But we check E0 difference.
        # So better check unperturbed part. But we can't isolate unperturbed from given total.
        # So we rely on the fact that sum of single-particle energies is higher.
        # The deltaE for antisymmetric is exactly 0, while for distinguishable it can be positive or negative.
        # So we must be careful: check that E_total_anti >= E_total if A>=0? Not necessarily.
        # We'll just check that the unperturbed part for anti is larger, but we cannot extract.
        # Alternative: we can check the derived formula: E_total_anti - (E_total - deltaE) = E0_anti - E0 > 0.
        # But we don't have deltaE directly. We can compute E0_anti and E0 from givens and then check that
        # E_total_anti - E_total > (E0_anti - E0) - deltaE? messy.
        # Simpler: just check that E_total_anti > E_total when we assume small deltaE? Not robust.
        # Let's skip this test because it's not universally true.
        # Instead, we'll test that the unperturbed anti energy formula is correct.
        return (False, "SKIP: Not universally monotonic due to possible negative A")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb one parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_decreases_with_a(self):
        """Check that if a increases, the unperturbed energy (and thus total) decreases.
           We perturb a by +5% and verify that E_total for distinguishable decreases (or at least does not increase significantly)."""
        if self._numeric_a is None or self._numeric_hbar is None or self._numeric_m is None:
            return (False, "SKIP: missing numeric givens for a, hbar, m")
        if self.E_total is None:
            return (False, "SKIP: E_total missing")
        try:
            E_orig = float(self.E_total)
        except (TypeError, ValueError):
            return (False, "E_total not numeric")
        # Compute new 'a' slightly increased: +5%
        a_orig = self._numeric_a
        a_new = a_orig * 1.05
        # Recompute the expected approximate E_total for new a, keeping other givens same.
        # Use formulas: E0 = (hbar^2 pi^2 / m)*(1/a^2+1/b^2+1/c^2)
        # deltaE = 27A/(8abc)
        E0_new = (self._numeric_hbar**2 * self._numeric_pi**2 / self._numeric_m) * (1/a_new**2 + 1/self._numeric_b**2 + 1/self._numeric_c**2)
        deltaE_new = (27 * self._numeric_A) / (8 * a_new * self._numeric_b * self._numeric_c)
        E_new = E0_new + deltaE_new
        # Check that E_new < E_orig (within tolerance since perturbation is small)
        if E_new < E_orig - 1e-12:
            return (True, f"Increasing a from {a_orig} to {a_new} decreases E_total: {E_orig} -> {E_new}")
        elif math.isclose(E_new, E_orig, rel_tol=1e-3):
            return (True, "E_total unchanged within tolerance (maybe a large box)")
        else:
            return (False, f"Increasing a increased E_total: {E_orig} -> {E_new}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no rederivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that all energies are finite numbers."""
        for name, val in [('E_total', self.E_total), ('E_total_s', self.E_total_s), ('E_total_anti', self.E_total_anti)]:
            if val is None:
                return (False, f"Missing {name}")
            try:
                fval = float(val)
            except (TypeError, ValueError):
                return (False, f"{name} not numeric: {val}")
            if not math.isfinite(fval):
                return (False, f"{name} is not finite: {val}")
        return (True, "All energies are finite")