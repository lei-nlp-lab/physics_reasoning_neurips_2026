import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a gas of distinguishable, non-interacting, zero-mass particles
    (E = pc). Checks interface, invariants, modeling consistency, monotonicity, and
    a weak answer sanity.
    """
    def __init__(self, solution_instance):
        # --- Extract results safely ---
        self.results = getattr(solution_instance, 'results', {})
        self.U = self.results.get('U', None)
        self.P = self.results.get('P', None)
        self.U_ordinary = self.results.get('U_ordinary', None)

        # --- Extract givens / inputs safely ---
        self.givens = getattr(solution_instance, 'givens', {})
        self.N = self.givens.get('N', None)
        self.V = self.givens.get('V', None)
        self.T = self.givens.get('T', None)
        self.k = self.givens.get('k', None)          # Boltzmann constant
        # No default numeric values – let tests fail explicitly

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the three required result keys are present."""
        if self.U is None:
            return (False, "Missing key 'U' in results.")
        if self.P is None:
            return (False, "Missing key 'P' in results.")
        if self.U_ordinary is None:
            return (False, "Missing key 'U_ordinary' in results.")
        return (True, "All required keys present.")

    def test_interface_numeric_types(self):
        """Check that result values are numeric (int or float)."""
        for name, val in [('U', self.U), ('P', self.P), ('U_ordinary', self.U_ordinary)]:
            if val is None:
                return (False, f"Cannot check type: {name} is None.")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (got {type(val).__name__}).")
        return (True, "All result values are numeric.")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_equation_of_state_PV_equals_NkT(self):
        """Check PV = NkT (ideal gas law)."""
        if any(x is None for x in [self.P, self.N, self.k, self.T, self.V]):
            return (False, "SKIP: missing givens or P to compute PV=NkT.")
        expected_P = self.N * self.k * self.T / self.V
        if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PV ≈ NkT: P={self.P}, expected={expected_P:.6e}")
        else:
            return (False, f"PV deviates: P={self.P}, expected={expected_P:.6e}")

    def test_invariant_energy_ratio_U_vs_ordinary(self):
        """Check that U is approximately twice U_ordinary (U = 3NkT, U_ord = 1.5NkT)."""
        if any(x is None for x in [self.U, self.U_ordinary]):
            return (False, "SKIP: U or U_ordinary missing.")
        try:
            ratio = self.U / self.U_ordinary
        except ZeroDivisionError:
            return (False, "U_ordinary is zero, cannot compute ratio.")
        if math.isclose(ratio, 2.0, rel_tol=1e-2):
            return (True, f"U/U_ordinary ≈ 2 (actual ratio = {ratio:.4f})")
        else:
            return (False, f"U/U_ordinary = {ratio:.4f}, expected ~2")

    # ------------------------------------------------------------
    # Modeling / trace tests (re-derive from physics)
    # ------------------------------------------------------------
    def test_modeling_U_derived(self):
        """Re-derive U = 3 N k T from givens and compare with result."""
        if any(x is None for x in [self.U, self.N, self.k, self.T]):
            return (False, "SKIP: missing input for U derivation.")
        expected_U = 3.0 * self.N * self.k * self.T
        if math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U matches 3NkT: U={self.U:.6e}, expected={expected_U:.6e}")
        else:
            return (False, f"U mismatch: U={self.U:.6e}, expected={expected_U:.6e}")

    def test_modeling_U_ordinary_derived(self):
        """Re-derive U_ordinary = 1.5 N k T and compare."""
        if any(x is None for x in [self.U_ordinary, self.N, self.k, self.T]):
            return (False, "SKIP: missing input for U_ordinary derivation.")
        expected_U_ord = 1.5 * self.N * self.k * self.T
        if math.isclose(self.U_ordinary, expected_U_ord, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U_ordinary matches 1.5NkT: U_ord={self.U_ordinary:.6e}, expected={expected_U_ord:.6e}")
        else:
            return (False, f"U_ordinary mismatch: U_ord={self.U_ordinary:.6e}, expected={expected_U_ord:.6e}")

    def test_modeling_pressure_derived(self):
        """Re-derive P = N k T / V and compare."""
        if any(x is None for x in [self.P, self.N, self.k, self.T, self.V]):
            return (False, "SKIP: missing input for pressure derivation.")
        expected_P = self.N * self.k * self.T / self.V
        if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches NkT/V: P={self.P:.6e}, expected={expected_P:.6e}")
        else:
            return (False, f"P mismatch: P={self.P:.6e}, expected={expected_P:.6e}")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation and direction check)
    # ------------------------------------------------------------
    def test_monotonicity_U_increases_with_T(self):
        """Check that U increases when T is increased by 5%."""
        if any(x is None for x in [self.N, self.k, self.T, self.U]):
            return (False, "SKIP: missing N, k, T, or U for monotonicity.")
        # Baseline: expected U from physics = 3 N k T
        U_baseline = 3.0 * self.N * self.k * self.T
        # Perturb T by +5%
        T_perturbed = self.T * 1.05
        U_perturbed_expected = 3.0 * self.N * self.k * T_perturbed
        # Compare: perturbed should be larger than baseline
        if U_perturbed_expected > U_baseline:
            return (True, f"dU/dT positive confirmed: baseline U={U_baseline:.6e}, perturbed U (T+5%)={U_perturbed_expected:.6e}")
        else:
            return (False, f"U did not increase: baseline={U_baseline:.6e}, perturbed={U_perturbed_expected:.6e}")

    def test_monotonicity_P_decreases_with_V(self):
        """Check that P decreases when V is increased by 5%."""
        if any(x is None for x in [self.N, self.k, self.T, self.V, self.P]):
            return (False, "SKIP: missing N, k, T, V, or P for monotonicity.")
        # Baseline: expected P = N k T / V
        P_baseline = self.N * self.k * self.T / self.V
        # Perturb V by +5%
        V_perturbed = self.V * 1.05
        P_perturbed_expected = self.N * self.k * self.T / V_perturbed
        # Compare: perturbed should be smaller than baseline
        if P_perturbed_expected < P_baseline:
            return (True, f"dP/dV negative confirmed: baseline P={P_baseline:.6e}, perturbed P (V+5%)={P_perturbed_expected:.6e}")
        else:
            return (False, f"P did not decrease: baseline={P_baseline:.6e}, perturbed={P_perturbed_expected:.6e}")

    # ------------------------------------------------------------
    # Weak answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """Check that U, P, U_ordinary are positive and finite."""
        checks = []
        for name, val in [('U', self.U), ('P', self.P), ('U_ordinary', self.U_ordinary)]:
            if val is None:
                checks.append(f"{name} is None")
            elif not math.isfinite(val):
                checks.append(f"{name} is not finite")
            elif val <= 0:
                checks.append(f"{name} is not positive (value={val})")
        if not checks:
            return (True, "All three result values are positive and finite.")
        else:
            return (False, "Issues: " + "; ".join(checks))

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (success, logs)."""
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