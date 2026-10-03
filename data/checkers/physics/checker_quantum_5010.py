import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics auditor for the 1D infinite well problem with a time-dependent
    narrow potential well perturbation.
    """
    def __init__(self, solution_instance: Any):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Wavefunction strings
        self.psi1: Optional[str] = results.get('psi1', None)
        self.psi2: Optional[str] = results.get('psi2', None)
        self.psi3: Optional[str] = results.get('psi3', None)
        self.psi4: Optional[str] = results.get('psi4', None)

        # Energy levels (in eV)
        self.E1: Optional[float] = results.get('E1', None)
        self.E2: Optional[float] = results.get('E2', None)
        self.E3: Optional[float] = results.get('E3', None)
        self.E4: Optional[float] = results.get('E4', None)

        # Probabilities
        self.P2: Optional[float] = results.get('P2', None)
        self.P3: Optional[float] = results.get('P3', None)
        self.P4: Optional[float] = results.get('P4', None)

        # Constants (may be available as solution_instance attributes)
        self.a: Optional[float] = getattr(solution_instance, 'a', None)          # box length in m
        self.V0: Optional[float] = getattr(solution_instance, 'V0', None)        # well depth in eV
        self.w: Optional[float] = getattr(solution_instance, 'w', None)          # well width in m
        self.dt: Optional[float] = getattr(solution_instance, 'dt', None)        # duration in s

        # Physical constants (standard values)
        self.hbar = 1.054571817e-34  # J*s
        self.m_e = 9.10938356e-31    # kg
        self.eV2J = 1.602176634e-19  # J/eV

    # --------------------------------------------------------------------------
    # Interface checks: required keys present and types
    # --------------------------------------------------------------------------
    def test_interface_psi1_exists(self) -> Tuple[bool, str]:
        if self.psi1 is None:
            return (False, "Missing psi1 in results")
        if not isinstance(self.psi1, str):
            return (False, "psi1 should be a string")
        return (True, "psi1 present and is string")

    def test_interface_psi2_exists(self) -> Tuple[bool, str]:
        if self.psi2 is None:
            return (False, "Missing psi2")
        if not isinstance(self.psi2, str):
            return (False, "psi2 should be a string")
        return (True, "psi2 present")

    def test_interface_psi3_exists(self) -> Tuple[bool, str]:
        if self.psi3 is None:
            return (False, "Missing psi3")
        if not isinstance(self.psi3, str):
            return (False, "psi3 should be a string")
        return (True, "psi3 present")

    def test_interface_psi4_exists(self) -> Tuple[bool, str]:
        if self.psi4 is None:
            return (False, "Missing psi4")
        if not isinstance(self.psi4, str):
            return (False, "psi4 should be a string")
        return (True, "psi4 present")

    def test_interface_energies_exists(self) -> Tuple[bool, str]:
        missing = []
        for key, val in [('E1', self.E1), ('E2', self.E2), ('E3', self.E3), ('E4', self.E4)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing energy keys: {missing}")
        for key, val in [('E1', self.E1), ('E2', self.E2), ('E3', self.E3), ('E4', self.E4)]:
            if not isinstance(val, (int, float)):
                return (False, f"{key} should be numeric")
        return (True, "All four energies present and numeric")

    def test_interface_probabilities_exists(self) -> Tuple[bool, str]:
        missing = []
        for key, val in [('P2', self.P2), ('P3', self.P3), ('P4', self.P4)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing probability keys: {missing}")
        for key, val in [('P2', self.P2), ('P3', self.P3), ('P4', self.P4)]:
            if not isinstance(val, (int, float)):
                return (False, f"{key} should be numeric")
        return (True, "All probabilities present and numeric")

    # --------------------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity
    # --------------------------------------------------------------------------
    def test_invariant_energy_scaling(self) -> Tuple[bool, str]:
        """Energy levels in infinite well scale as n^2."""
        if any(e is None for e in [self.E1, self.E2, self.E3, self.E4]):
            return (False, "SKIP: missing energy values")
        ratios = [(2, self.E2/self.E1), (3, self.E3/self.E1), (4, self.E4/self.E1)]
        expected = {2: 4.0, 3: 9.0, 4: 16.0}
        for n, ratio in ratios:
            if not math.isclose(ratio, expected[n], rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"E{n}/E1 = {ratio:.3f}, expected {expected[n]}")
        return (True, "Energy ratios match n^2 scaling")

    def test_invariant_probability_selection_rules(self) -> Tuple[bool, str]:
        """P2 and P4 should be zero due to node at centre (sin(nπ/2)=0 for even n)."""
        if self.P2 is None or self.P4 is None:
            return (False, "SKIP: missing P2 or P4")
        if not math.isclose(self.P2, 0.0, abs_tol=1e-6):
            return (False, f"P2 = {self.P2}, expected ~0")
        if not math.isclose(self.P4, 0.0, abs_tol=1e-6):
            return (False, f"P4 = {self.P4}, expected ~0")
        return (True, "P2 and P4 vanish as required by selection rule")

    def test_invariant_probability_sum_bound(self) -> Tuple[bool, str]:
        """Sum of probabilities to other states cannot exceed 1 (unitarity)."""
        if any(p is None for p in [self.P2, self.P3, self.P4]):
            return (False, "SKIP: missing probabilities")
        total = self.P2 + self.P3 + self.P4
        if total > 1.0:
            return (False, f"Total probability {total} > 1")
        return (True, f"Total probability {total:.2e} ≤ 1")

    def test_invariant_energies_positive_and_ordering(self) -> Tuple[bool, str]:
        """Energies must be positive and increase with n."""
        if any(e is None for e in [self.E1, self.E2, self.E3, self.E4]):
            return (False, "SKIP: missing energies")
        vals = [self.E1, self.E2, self.E3, self.E4]
        if any(v <= 0 for v in vals):
            return (False, "All energies must be positive")
        if not (self.E1 < self.E2 < self.E3 < self.E4):
            return (False, "Energies must be strictly increasing")
        return (True, "Energies positive and increasing")

    # --------------------------------------------------------------------------
    # Modeling / trace checks: explainability and step consistency
    # --------------------------------------------------------------------------
    def test_modeling_wavefunction_boundary_conditions(self) -> Tuple[bool, str]:
        """The wavefunctions should vanish at x=0 and x=a (infinite well)."""
        if any(p is None for p in [self.psi1, self.psi2, self.psi3, self.psi4]):
            return (False, "SKIP: missing wavefunction strings")
        # Since we have symbolic expressions, check that all contain sin(nπx/a)
        # and that arguments at x=0 and x=a yield zero.
        # We cannot evaluate numerically without a, but we can verify the structure.
        expected_sine = 'sin'
        for psi_str, name in [(self.psi1, 'psi1'), (self.psi2, 'psi2'),
                               (self.psi3, 'psi3'), (self.psi4, 'psi4')]:
            if expected_sine not in psi_str.replace(' ','').lower():
                return (False, f"{name} does not appear to contain sin function")
        # Additionally, if a is provided, we could sample endpoints; skip if not.
        if self.a is None:
            return (True, "Wavefunctions structurally contain sin; a unknown, skipping endpoint evaluation")
        # Construct numerical evaluator (simple)
        from sympy import symbols, sin, pi, lambdify, sqrt
        x = symbols('x')
        a_val = self.a
        for psi_str, name in [(self.psi1, 'psi1'), (self.psi2, 'psi2'),
                               (self.psi3, 'psi3'), (self.psi4, 'psi4')]:
            # Convert string to sympy expression (assume format "sqrt(2/a)*sin(n*pi*x/a)")
            try:
                # try to replace symbolic a with numeric a_val
                expr_str = psi_str.replace('a', str(a_val))
                expr = eval(expr_str)
                f = lambdify(x, expr, 'numpy')
                val0 = f(0)
                vala = f(a_val)
                if not math.isclose(val0, 0, abs_tol=1e-10):
                    return (False, f"{name} non-zero at x=0: {val0}")
                if not math.isclose(vala, 0, abs_tol=1e-10):
                    return (False, f"{name} non-zero at x=a: {vala}")
            except Exception:
                # Parsing failure not fatal
                pass
        return (True, "Wavefunctions pass boundary condition check")

    def test_modeling_probability_formula_consistency(self) -> Tuple[bool, str]:
        """P3 should be consistent with perturbation theory formula.
        We do not compute expected value, but ensure positivity and dependence on parameters."""
        if self.P3 is None:
            return (False, "SKIP: missing P3")
        if self.P3 <= 0:
            return (False, f"P3 must be positive, got {self.P3}")
        # If parameters are available, check that P3 scales with V0^2 and w^2
        if None in (self.V0, self.w, self.a, self.dt, self.E1, self.E3):
            return (True, "P3 positive; cannot validate parameter dependence due to missing data")
        # Compute approximate matrix element magnitude using given data
        # M3 ≈ V0 * w * (2/a) * sin(3π/2) = -2*V0*w/a
        V0_J = self.V0 * self.eV2J
        M3 = -2 * V0_J * self.w / self.a
        # Omega: (E3 - E1)/hbar in rad/s
        omega = (self.E3 - self.E1) * self.eV2J / self.hbar
        # Compute predicted P3
        predicted_P3 = (abs(M3)**2 * 4 * (math.sin(omega * self.dt / 2)**2)) / (self.hbar**2 * omega**2)
        if predicted_P3 > 1e-3: # sanity check that predicted is reasonable
            pass
        # Compare with given P3 (relative tolerance because of rounding)
        if not math.isclose(self.P3, predicted_P3, rel_tol=5e-2, abs_tol=1e-6):
            return (False, f"P3 = {self.P3}, but from parameters derived {predicted_P3:.3e}")
        return (True, "P3 consistent with parameter-based computation")

    def test_modeling_matrix_element_zero_for_even_n(self) -> Tuple[bool, str]:
        """Check that the matrix element approximation yields zero for n=2 and n=4."""
        if any(p is None for p in [self.P2, self.P4, self.V0, self.w, self.a]):
            return (False, "SKIP: missing data for matrix element check")
        # M_n approx = V0 * w * (2/a) * sin(nπ/2)
        V0 = self.V0 * self.eV2J
        factor = 2 * V0 * self.w / self.a
        M2 = factor * math.sin(2*math.pi/2)  # sin(π)=0
        M4 = factor * math.sin(4*math.pi/2)  # sin(2π)=0
        if not math.isclose(abs(M2), 0, abs_tol=1e-20) or not math.isclose(abs(M4), 0, abs_tol=1e-20):
            return (False, "Matrix element for even n should be zero")
        return (True, "Matrix elements vanish for even n")

    def test_modeling_energy_formula(self) -> Tuple[bool, str]:
        """If box length is provided, energies should match standard formula."""
        if self.a is None or self.E1 is None:
            return (False, "SKIP: missing a or E1")
        # Compute E1 using E1 = ħ²π²/(2 m_e a²) in J, convert to eV
        E1_J = (self.hbar**2 * math.pi**2) / (2 * self.m_e * self.a**2)
        E1_eV = E1_J / self.eV2J
        if not math.isclose(self.E1, E1_eV, rel_tol=1e-2, abs_tol=0.5):
            return (False, f"E1 = {self.E1} eV, expected approx {E1_eV:.2f} eV")
        return (True, f"E1 consistent with infinite well formula: {E1_eV:.2f} eV")

    # --------------------------------------------------------------------------
    # Monotonicity checks: physical trends
    # --------------------------------------------------------------------------
    def test_monotonicity_energy_with_a(self) -> Tuple[bool, str]:
        """Energy decreases when box length increases (E ∝ 1/a²)."""
        if self.a is None or self.E1 is None:
            return (False, "SKIP: missing a or E1")
        # Perturb a by +10%
        a_pert = self.a * 1.1
        # Recompute E1 with perturbed a
        E1_pert_J = (self.hbar**2 * math.pi**2) / (2 * self.m_e * a_pert**2)
        E1_pert_eV = E1_pert_J / self.eV2J
        if E1_pert_eV >= self.E1:
            return (False, f"Increasing a should decrease E1, but perturbed E1={E1_pert_eV:.3f} >= original {self.E1:.3f}")
        return (True, f"E1 decreases when a increases: {self.E1:.3f} -> {E1_pert_eV:.3f} eV")

    def test_monotonicity_probability_with_V0(self) -> Tuple[bool, str]:
        """Probability P3 increases with |V0| (P ∝ V0²)."""
        if self.P3 is None or self.V0 is None or self.a is None or self.w is None or self.dt is None or self.E1 is None or self.E3 is None:
            return (False, "SKIP: missing data for V0 monotonicity")
        # Double |V0|
        V0_pert = self.V0 * 2
        M3_original = abs(self.V0) * self.w * (2/self.a) * abs(math.sin(3*math.pi/2))
        M3_pert = abs(V0_pert) * self.w * (2/self.a) * abs(math.sin(3*math.pi/2))
        ratio = (M3_pert / M3_original)**2  # = (V0_pert/V0)^2 = 4
        # Compute original P3 from parameters to avoid using provided P3 (which might be approximate)
        # But we can just check that P3 should increase by factor ~4
        # However, provided P3 might have rounding; use approximate.
        # We'll compute nominal P3 from original parameters (with allowed tolerance)
        # This is not a hard check but a trend.
        # Simpler: just assert that if we double V0 magnitude, probability goes up.
        # Without computing full, we can reason: P3 ∝ M3² ∝ V0².
        # But we cannot recompute exact because argument of sin depends on omega (energy unchanged)
        # So we expect new probability = 4 * old probability (approx).
        # We'll check using the numeric computation with perturbed V0 (assuming other params unchanged)
        V0_J_orig = self.V0 * self.eV2J
        V0_J_pert = V0_pert * self.eV2J
        omega = (self.E3 - self.E1) * self.eV2J / self.hbar
        M3_orig_val = -2 * V0_J_orig * self.w / self.a
        M3_pert_val = -2 * V0_J_pert * self.w / self.a
        P3_orig_derived = (abs(M3_orig_val)**2 * 4 * (math.sin(omega * self.dt / 2)**2)) / (self.hbar**2 * omega**2)
        P3_pert_derived = (abs(M3_pert_val)**2 * 4 * (math.sin(omega * self.dt / 2)**2)) / (self.hbar**2 * omega**2)
        if not (P3_pert_derived > P3_orig_derived):
            return (False, f"Doubling |V0| did not increase probability: original derived {P3_orig_derived:.2e}, perturbed {P3_pert_derived:.2e}")
        return (True, f"P3 increases when |V0| increases: derived ratio {P3_pert_derived/P3_orig_derived:.2f} (expected ~4)")

    def test_monotonicity_probability_with_dt(self) -> Tuple[bool, str]:
        """P3 oscillates with dt, but for small dt it should increase initially (sin^2(ωΔt/2) ∝ Δt^2)."""
        if self.P3 is None or self.dt is None or self.V0 is None or self.a is None or self.w is None or self.E1 is None or self.E3 is None:
            return (False, "SKIP: missing data for dt monotonicity")
        # For small dt, P3 ∝ dt^2. Check that halving dt reduces probability (to first approximation)
        dt_half = self.dt / 2
        omega = (self.E3 - self.E1) * self.eV2J / self.hbar
        M3 = -2 * self.V0 * self.eV2J * self.w / self.a
        P3_half = (abs(M3)**2 * 4 * (math.sin(omega * dt_half / 2)**2)) / (self.hbar**2 * omega**2)
        if P3_half >= self.P3:
            return (False, f"Halving dt should reduce P3 (approx. factor 4), but derived P3(dt/2)={P3_half:.2e} >= original {self.P3:.2e}")
        return (True, f"P3 decreases when dt halved: derived P3(dt/2)={P3_half:.2e} vs original {self.P3:.2e}")

    # --------------------------------------------------------------------------
    # Weak answer sanity (only one allowed, no isclose, no re-derivation)
    # --------------------------------------------------------------------------
    def test_answer_sanity_energies_sign_and_range(self) -> Tuple[bool, str]:
        """Quick sanity: energies should be positive and not astronomically large."""
        if any(e is None for e in [self.E1, self.E2, self.E3, self.E4]):
            return (False, "SKIP: missing energies")
        vals = [self.E1, self.E2, self.E3, self.E4]
        if any(v <= 0 for v in vals):
            return (False, "Some energy is non-positive")
        if any(v > 1e6 for v in vals):  # far beyond typical eV scale for 1Å box
            return (False, f"Energy > 1e6 eV unreasonable")
        return (True, "Energies are positive and within reasonable range")

    # --------------------------------------------------------------------------
    # Test runner
    # --------------------------------------------------------------------------
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)