import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for degenerate perturbation theory in 2D harmonic oscillator with H' = Kxy.
    Checks first and second order energy corrections and first order wavefunction corrections.
    """

    def __init__(self, solution_instance):
        # Extract results from solution_instance.results dict (safe defaults, NEVER numeric)
        results = getattr(solution_instance, 'results', {})
        self.first_order_energies = results.get('first_order_energies', None)
        self.good_states = results.get('good_states', None)
        self.plus_corr = results.get('plus_corr', None)
        self.minus_corr = results.get('minus_corr', None)
        self.E_second = results.get('E_second', None)
        self.E_plus_total = results.get('E_plus_total', None)
        self.E_minus_total = results.get('E_minus_total', None)

        # Extract physical constants (optional, for invariant/monotonicity checks)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.m = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.K = getattr(solution_instance, 'K', None)

        # Precompute derived constants if givens are available (used for perturbation checks)
        self.epsilon = None
        self.lambda_ = None
        if self.K is not None and self.m is not None and self.omega is not None and self.hbar is not None:
            self.epsilon = self.K / (2 * self.m * self.omega**2)
            self.lambda_ = self.epsilon * self.hbar * self.omega

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check all expected result keys are present and not None."""
        required = {
            'first_order_energies': self.first_order_energies,
            'good_states': self.good_states,
            'plus_corr': self.plus_corr,
            'minus_corr': self.minus_corr,
            'E_second': self.E_second,
            'E_plus_total': self.E_plus_total,
            'E_minus_total': self.E_minus_total
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present")

    def test_interface_energy_list_type(self):
        """Check first_order_energies is a list of length 2."""
        if self.first_order_energies is None:
            return (False, "SKIP: first_order_energies not available")
        if not isinstance(self.first_order_energies, list) or len(self.first_order_energies) != 2:
            return (False, "first_order_energies must be a list of length 2")
        return (True, "first_order_energies is a list of length 2")

    def test_interface_good_states_type(self):
        """Check good_states is a list of two strings."""
        if self.good_states is None:
            return (False, "SKIP: good_states not available")
        if not isinstance(self.good_states, list) or len(self.good_states) != 2:
            return (False, "good_states must be a list of length 2")
        if not all(isinstance(s, str) for s in self.good_states):
            return (False, "good_states elements must be strings")
        return (True, "good_states is a list of two strings")

    def test_interface_wavefunction_strs(self):
        """Check plus_corr and minus_corr are strings."""
        for name, val in [('plus_corr', self.plus_corr), ('minus_corr', self.minus_corr)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, str):
                return (False, f"{name} must be a string")
        return (True, "plus_corr and minus_corr are strings")

    def test_interface_second_order_type(self):
        """Check E_second is numeric or symbolic."""
        if self.E_second is None:
            return (False, "SKIP: E_second not available")
        if not (isinstance(self.E_second, (int, float)) or isinstance(self.E_second, sp.Basic)):
            return (False, "E_second must be numeric or symbolic")
        return (True, "E_second has valid type")

    def test_interface_total_energy_types(self):
        """Check E_plus_total and E_minus_total are numeric or symbolic."""
        for name, val in [('E_plus_total', self.E_plus_total),
                          ('E_minus_total', self.E_minus_total)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not (isinstance(val, (int, float)) or isinstance(val, sp.Basic)):
                return (False, f"{name} must be numeric or symbolic")
        return (True, "Total energies have valid types")

    # ----------------------------------------------------------------
    # Invariant tests (physics‑based, using math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_degeneracy_dimension(self):
        """Degenerate subspace dimension is 2."""
        if self.first_order_energies is None or self.good_states is None:
            return (False, "SKIP: needed results missing")
        if len(self.first_order_energies) != 2 or len(self.good_states) != 2:
            return (False, "Degenerate subspace must have dimension 2")
        return (True, "Degenerate subspace dimension is 2")

    def test_invariant_first_order_symmetry(self):
        """First‑order energies sum to 4*hbar*omega (symmetric around unperturbed)."""
        if self.first_order_energies is None or self.hbar is None or self.omega is None:
            return (False, "SKIP: missing data for symmetry check")
        E1, E2 = self.first_order_energies
        sum_expected = 4 * self.hbar * self.omega
        if isinstance(E1, sp.Basic):
            diff = sp.simplify(E1 + E2 - sum_expected)
            if diff != 0:
                return (False, f"First-order energies sum to {E1+E2}, expected {sum_expected}")
        elif isinstance(E1, (int, float)):
            if not math.isclose(E1+E2, sum_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"First-order energies sum to {E1+E2}, expected {sum_expected}")
        else:
            return (False, "First-order energies have unknown type")
        return (True, "First-order energies sum to 4*hbar*omega")

    def test_invariant_offdiagonal_consistency(self):
        """Energy difference equals 2*lambda = K*hbar/(m*omega)."""
        if (self.first_order_energies is None or self.K is None or
            self.m is None or self.omega is None or self.hbar is None):
            return (False, "SKIP: missing data for offdiagonal check")
        E1, E2 = self.first_order_energies
        lambda_val = self.K * self.hbar / (2 * self.m * self.omega)
        diff_expected = 2 * lambda_val
        if isinstance(E1, sp.Basic):
            diff = sp.simplify(E1 - E2)
            if sp.simplify(diff - diff_expected) != 0:
                return (False, f"Energy difference {diff} != expected {diff_expected}")
        elif isinstance(E1, (int, float)):
            if not math.isclose(E1-E2, diff_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Energy difference {E1-E2} != expected {diff_expected}")
        else:
            return (False, "Unknown type for energies")
        return (True, "Off-diagonal matrix element consistent with theory")

    def test_invariant_second_order_negative(self):
        """Second‑order correction must be negative (denominators smaller)."""
        if self.E_second is None:
            return (False, "SKIP: E_second not available")
        eps = sp.symbols('epsilon', positive=True)
        if isinstance(self.E_second, sp.Basic):
            val = self.E_second.subs(eps, 0.1).evalf()
            if val > 0:
                return (False, f"E_second appears positive (val={val})")
        elif isinstance(self.E_second, (int, float)):
            if self.E_second >= 0:
                return (False, f"E_second = {self.E_second} is not negative")
        else:
            return (False, "Unknown type for E_second")
        return (True, "E_second is negative")

    # ----------------------------------------------------------------
    # Modeling / trace tests (check wavefunction structure and consistency)
    # ----------------------------------------------------------------
    def test_modeling_plus_corr_structure(self):
        """First‑order correction to |+⟩ uses plus combination |20⟩+|02⟩ (or |12⟩+|21⟩)."""
        if self.plus_corr is None:
            return (False, "SKIP: plus_corr not available")
        if ("|2,0> + |0,2>" in self.plus_corr or "|12> + |21>" in self.plus_corr):
            return (True, "plus_corr structure matches expected")
        return (False, f"plus_corr does not contain expected combination, got: {self.plus_corr}")

    def test_modeling_minus_corr_structure(self):
        """First‑order correction to |−⟩ uses minus combination |20⟩−|02⟩ (or |12⟩−|21⟩)."""
        if self.minus_corr is None:
            return (False, "SKIP: minus_corr not available")
        if ("|2,0> - |0,2>" in self.minus_corr or "|12> - |21>" in self.minus_corr):
            return (True, "minus_corr structure matches expected")
        return (False, f"minus_corr does not contain expected difference, got: {self.minus_corr}")

    def test_trace_energy_consistency(self):
        """
        Consistency checks:
        1. Total energy = first‑order + second‑order.
        2. Difference of total energies equals first‑order splitting (since second order same).
        3. Sum of total energies = 4*hbar*omega + 2*E_second.
        """
        if (self.first_order_energies is None or self.E_second is None or
            self.E_plus_total is None or self.E_minus_total is None):
            return (False, "SKIP: missing results for consistency")
        E1, E2 = self.first_order_energies
        E2nd = self.E_second

        def approx_eq(a, b):
            if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
                return sp.simplify(a - b) == 0
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
            else:
                return False

        # 1. Total = first + second
        if not approx_eq(self.E_plus_total, E1 + E2nd):
            return (False, f"E_plus_total = {self.E_plus_total} != {E1}+{E2nd}")
        if not approx_eq(self.E_minus_total, E2 + E2nd):
            return (False, f"E_minus_total = {self.E_minus_total} != {E2}+{E2nd}")

        # 2. Difference = first‑order splitting
        if not approx_eq(self.E_plus_total - self.E_minus_total, E1 - E2):
            return (False, f"Total energy difference {self.E_plus_total - self.E_minus_total} != splitting {E1-E2}")

        # 3. Sum check: sum of total = (E1+E2) + 2*E2nd
        sum_expected = (E1 + E2) + 2*E2nd  # E1+E2 = 4*hbar*omega (by symmetry)
        if not approx_eq(self.E_plus_total + self.E_minus_total, sum_expected):
            return (False, f"Sum of total {self.E_plus_total+self.E_minus_total} != expected {sum_expected}")

        return (True, "Energy consistency checks passed (addition, difference, sum)")

    # ----------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ----------------------------------------------------------------
    def test_monotonicity_splitting_increases_with_K(self):
        """
        Perturb K → 1.05*K. The first‑order splitting (2*lambda) should increase.
        Compare theoretical value for perturbed K vs baseline (theoretical or solution).
        """
        if self.K is None or self.m is None or self.omega is None or self.hbar is None:
            return (False, "SKIP: missing constants for monotonicity test")
        if self.first_order_energies is None:
            return (False, "SKIP: first_order_energies not available")

        # Baseline splitting (theoretical, based on solution's K)
        lambda_baseline = self.K * self.hbar / (2 * self.m * self.omega)
        splitting_baseline = 2 * lambda_baseline

        # Perturb K upward by 5%
        K_perturbed = 1.05 * self.K
        lambda_pert = K_perturbed * self.hbar / (2 * self.m * self.omega)
        splitting_pert = 2 * lambda_pert

        if not (splitting_pert > splitting_baseline):
            return (False, f"Perturbed splitting {splitting_pert} not > baseline {splitting_baseline}")

        # Optional: check solution splitting matches baseline
        E1, E2 = self.first_order_energies
        if isinstance(E1, (int, float)) and isinstance(E2, (int, float)):
            sol_split = abs(E1 - E2)
            if not math.isclose(sol_split, splitting_baseline, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Solution splitting {sol_split} deviates from theory {splitting_baseline}")

        return (True, f"Splitting increases with K: baseline {splitting_baseline} < perturbed {splitting_pert}")

    def test_monotonicity_second_order_magnitude_increases_with_K(self):
        """
        Perturb K → 1.05*K. The magnitude of second‑order correction (|E_second|) should increase.
        Since E_second ~ -K^2, it becomes more negative.
        """
        if self.K is None or self.m is None or self.omega is None or self.hbar is None:
            return (False, "SKIP: missing constants for monotonicity test")
        if self.E_second is None:
            return (False, "SKIP: E_second not available")

        # Theoretical second order for given K: -2*epsilon^2*hbar*omega
        epsilon_baseline = self.K / (2 * self.m * self.omega**2)
        E2_baseline = -2 * epsilon_baseline**2 * self.hbar * self.omega

        # Perturb K upward by 5%
        K_pert = 1.05 * self.K
        epsilon_pert = K_pert / (2 * self.m * self.omega**2)
        E2_pert = -2 * epsilon_pert**2 * self.hbar * self.omega

        # More negative means larger magnitude (since second order is negative)
        if not (E2_pert < E2_baseline):
            return (False, f"Perturbed E_second {E2_pert} not more negative than baseline {E2_baseline}")

        # Optionally check that solution E_second matches baseline
        if isinstance(self.E_second, (int, float)) or (isinstance(self.E_second, sp.Basic) and self.E_second.is_Float):
            if not math.isclose(float(self.E_second), float(E2_baseline), rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Solution E_second {self.E_second} deviates from theory {E2_baseline}")

        return (True, f"|E_second| increases with K: baseline {E2_baseline} -> perturbed {E2_pert}")

    # ----------------------------------------------------------------
    # Weak answer sanity test (only one, no hardcoded target values)
    # ----------------------------------------------------------------
    def test_answer_sanity_positive_energies(self):
        """Sanity: total energies should be positive for small positive epsilon."""
        eps = sp.symbols('epsilon', positive=True)
        eps_val = 0.1
        for name, val in [('E_plus_total', self.E_plus_total),
                          ('E_minus_total', self.E_minus_total)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if isinstance(val, sp.Basic):
                if val.has(sp.Symbol):
                    try:
                        numeric_val = val.subs(eps, eps_val).evalf()
                    except:
                        continue
                else:
                    numeric_val = val.evalf()
                if numeric_val < 0:
                    return (False, f"{name} (evaluated {numeric_val}) negative for epsilon={eps_val}")
            elif isinstance(val, (int, float)):
                if val <= 0:
                    return (False, f"{name} = {val} is not positive")
        return (True, "All total energies appear positive (or positive for typical epsilon)")

    # ----------------------------------------------------------------
    # Main test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)