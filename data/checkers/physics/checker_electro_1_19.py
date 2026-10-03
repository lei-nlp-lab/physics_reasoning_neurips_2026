import math
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Checker for the physics problem: isolated soap bubble collapsing from 1 cm
    to 1 mm radius, initial potential 100 V, compute change in electrostatic energy.
    """
    def __init__(self, solution_instance: Any):
        # Extract results safely; default to None
        results = getattr(solution_instance, 'results', {}) or {}
        self.delta_U = results.get('delta_U', None)

        # Givens (optional, from solution_instance if provided)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.R_i = givens.get('R_i', None)
        self.R_f = givens.get('R_f', None)
        self.V_i = givens.get('V_i', None)

        # Constants (taken from environment or defaults)
        self.epsilon0 = 8.854187817e-12  # F/m
        self.pi = math.pi

        # Derived quantities (computed in checker from available givens)
        # Formula: delta_U = 2 * pi * epsilon0 * R_i * V_i^2 * (R_i / R_f - 1)
        if (self.R_i is not None and self.R_f is not None and self.V_i is not None
            and self.R_f != 0):
            self.derived_delta_U = 2 * self.pi * self.epsilon0 * self.R_i * (self.V_i**2) * (self.R_i / self.R_f - 1)
        else:
            self.derived_delta_U = None

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_delta_U_exists(self) -> Tuple[bool, str]:
        if self.delta_U is None:
            return (False, "FAIL: results['delta_U'] is None, missing key")
        return (True, "PASS: delta_U key present")

    def test_interface_delta_U_is_numeric(self) -> Tuple[bool, str]:
        if self.delta_U is None:
            return (False, "SKIP: delta_U not available")
        if not isinstance(self.delta_U, (int, float)):
            return (False, f"FAIL: delta_U type is {type(self.delta_U).__name__}, expected int/float")
        return (True, "PASS: delta_U is numeric")

    # ------------------------------------------------------------
    # Invariant (theorem) checks
    # ------------------------------------------------------------
    def test_invariant_energy_change_positive(self) -> Tuple[bool, str]:
        """Energy should increase because final capacitor has smaller radius (smaller capacitance) for constant charge."""
        if self.delta_U is None:
            return (False, "SKIP: delta_U not available")
        if self.delta_U <= 0:
            return (False, f"FAIL: delta_U = {self.delta_U}, expected > 0")
        return (True, "PASS: delta_U > 0")

    def test_invariant_charge_conservation_implicit(self) -> Tuple[bool, str]:
        """
        Check that the solution's delta_U respects charge conservation.
        Derive expected delta_U from givens; compare with solution's delta_U.
        """
        if self.delta_U is None or self.derived_delta_U is None:
            return (False, "SKIP: missing data for charge conservation check")
        if not math.isclose(self.delta_U, self.derived_delta_U, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: delta_U {self.delta_U} differs from derived {self.derived_delta_U}")
        return (True, "PASS: delta_U consistent with charge conservation")

    # ------------------------------------------------------------
    # Modeling checks (using explanation steps / formulas)
    # ------------------------------------------------------------
    def test_modeling_delta_U_formula(self) -> Tuple[bool, str]:
        """
        Direct modeling check: compare solution's delta_U against
        the analytical formula using provided givens (R_i, R_f, V_i).
        """
        if self.delta_U is None or self.derived_delta_U is None:
            return (False, "SKIP: cannot compute reference delta_U (missing givens)")
        if not math.isclose(self.delta_U, self.derived_delta_U, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: delta_U {self.delta_U} != derived {self.derived_delta_U} (within tolerance)")
        return (True, f"PASS: delta_U matches derived formula ({self.derived_delta_U:.3e})")

    def test_modeling_energy_from_capacitance(self) -> Tuple[bool, str]:
        """
        Check that the derived formula is itself correct: compute U_i and U_f
        using spherical capacitance expressions and verify delta_U = U_f - U_i.
        """
        if self.R_i is None or self.R_f is None or self.V_i is None:
            return (False, "SKIP: missing givens for capacitance check")
        # Compute using formulas from physics mapping
        C_i = 4 * self.pi * self.epsilon0 * self.R_i
        Q = C_i * self.V_i
        U_i = 0.5 * C_i * (self.V_i**2)
        C_f = 4 * self.pi * self.epsilon0 * self.R_f
        U_f = 0.5 * (Q**2) / C_f
        delta_U_from_energy = U_f - U_i
        # Compare with solution's delta_U if available, else just report consistency
        if self.delta_U is not None and not math.isclose(self.delta_U, delta_U_from_energy, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: delta_U {self.delta_U} differs from U_f-U_i {delta_U_from_energy:.3e}")
        if self.derived_delta_U is not None and not math.isclose(delta_U_from_energy, self.derived_delta_U, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: U_f-U_i {delta_U_from_energy:.3e} disagrees with derived formula {self.derived_delta_U:.3e}")
        return (True, f"PASS: U_f-U_i = {delta_U_from_energy:.3e} (consistent)")

    def test_modeling_intermediate_charge(self) -> Tuple[bool, str]:
        """
        Modeling check: verify that the charge Q computed from initial conditions
        is physically plausible (positive, finite).
        """
        if self.R_i is None or self.V_i is None:
            return (False, "SKIP: missing R_i or V_i")
        Q = 4 * self.pi * self.epsilon0 * self.R_i * self.V_i
        if not (Q > 0 and math.isfinite(Q)):
            return (False, f"FAIL: Q = {Q} is not positive finite")
        return (True, f"PASS: Q = {Q:.3e} C (positive finite)")

    # ------------------------------------------------------------
    # Monotonicity checks (must perturb an input and compare outputs)
    # ------------------------------------------------------------
    def test_monotonicity_delta_U_increases_with_R_i(self) -> Tuple[bool, str]:
        """
        Perturb initial radius R_i by +10% and check that derived delta_U increases.
        """
        if self.R_i is None or self.R_f is None or self.V_i is None:
            return (False, "SKIP: missing givens for monotonicity")
        # Baseline
        baseline = 2 * self.pi * self.epsilon0 * self.R_i * (self.V_i**2) * (self.R_i / self.R_f - 1)
        # Perturb R_i up by 10%
        R_i_prime = self.R_i * 1.1
        perturbed = 2 * self.pi * self.epsilon0 * R_i_prime * (self.V_i**2) * (R_i_prime / self.R_f - 1)
        if not (perturbed > baseline):
            return (False, f"FAIL: perturbed delta_U {perturbed:.3e} <= baseline {baseline:.3e}")
        return (True, f"PASS: 10% increase in R_i increases delta_U ({baseline:.3e} -> {perturbed:.3e})")

    def test_monotonicity_delta_U_increases_with_V_i(self) -> Tuple[bool, str]:
        """
        Perturb initial potential V_i by +10% and check that derived delta_U increases.
        delta_U is quadratic in V_i, so a positive increase should raise it.
        """
        if self.R_i is None or self.R_f is None or self.V_i is None:
            return (False, "SKIP: missing givens for monotonicity")
        # Baseline (derived formula)
        baseline = 2 * self.pi * self.epsilon0 * self.R_i * (self.V_i**2) * (self.R_i / self.R_f - 1)
        # Perturb V_i up by 10%
        V_i_prime = self.V_i * 1.1
        perturbed = 2 * self.pi * self.epsilon0 * self.R_i * (V_i_prime**2) * (self.R_i / self.R_f - 1)
        if not (perturbed > baseline):
            return (False, f"FAIL: perturbed delta_U {perturbed:.3e} <= baseline {baseline:.3e}")
        return (True, f"PASS: 10% increase in V_i increases delta_U ({baseline:.3e} -> {perturbed:.3e})")

    # ------------------------------------------------------------
    # Weak answer sanity (only one allowed, do not use isclose)
    # ------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self) -> Tuple[bool, str]:
        """Weak sanity: delta_U should be of order 1e-8 J (positive and ~5e-8)."""
        if self.delta_U is None:
            return (False, "SKIP: delta_U not available")
        # Check sign and broad magnitude: between 1e-9 and 1e-7
        if not (1e-9 < self.delta_U < 1e-7):
            return (False, f"FAIL: delta_U = {self.delta_U}, expected ~1e-8 J")
        return (True, f"PASS: delta_U = {self.delta_U:.3e} J (order 1e-8)")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)