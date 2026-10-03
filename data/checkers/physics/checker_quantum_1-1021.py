import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the δ-potential bound state problem.
    Checks solution: binding energy E_b = m V0^2 / (2 ħ^2), V0<0.
    """

    def __init__(self, solution_instance):
        # Extract inputs from solution_instance safely
        self.m = getattr(solution_instance, 'm', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        results = getattr(solution_instance, 'results', {})
        self.E_b = results.get('E_b', None)

        # Explanation steps for context
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    def run_all_tests(self):
        logs = []
        all_pass = True
        for attr in sorted(dir(self)):
            if not attr.startswith("test_"):
                continue
            method = getattr(self, attr)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, "{} CRASHED: {}".format(attr, e)
            logs.append("{}: {}: {}".format('PASS' if passed else 'FAIL', attr, msg))
            if not passed:
                all_pass = False
        return all_pass, logs

    # ===== Interface checks =====
    def test_interface_required_keys_exist(self):
        """Check that all required input quantities are present."""
        if self.m is None:
            return (False, "Missing input: m (mass)")
        if self.V0 is None:
            return (False, "Missing input: V0 (potential strength)")
        if self.hbar is None:
            return (False, "Missing input: hbar")
        if self.E_b is None:
            return (False, "Missing result: E_b (binding energy)")
        for name, val in [("m", self.m), ("V0", self.V0), ("hbar", self.hbar), ("E_b", self.E_b)]:
            if not isinstance(val, (int, float, complex)):
                return (False, "Non-numeric type for {}: {}".format(name, type(val)))
        return (True, "All required keys present and numeric.")

    def test_interface_V0_negative_for_bound_state(self):
        """Bound state requires V0 < 0."""
        if self.V0 is None:
            return (False, "V0 not available, cannot check sign.")
        if self.V0 >= 0:
            return (False, "V0 = {} must be negative for a bound state.".format(self.V0))
        return (True, "V0 = {} < 0, bound state possible.".format(self.V0))

    # ===== Invariant checks (theorem-based physics sanity) =====
    def test_invariant_energy_negative(self):
        """Binding energy positive implies energy E = -E_b < 0."""
        if self.E_b is None:
            return (False, "E_b not available.")
        E = -self.E_b
        if E >= 0:
            return (False, "Bound state energy E = {} should be negative.".format(E))
        return (True, "Energy E = {} is negative as required.".format(E))

    def test_invariant_kappa_positive(self):
        """From E_b, compute κ = sqrt(2 m E_b)/ħ, must be positive."""
        if self.m is None or self.hbar is None or self.E_b is None:
            return (False, "Missing m, hbar, or E_b to compute κ.")
        m_val = float(self.m)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        kappa = math.sqrt(2.0 * m_val * E_b_val) / hbar_val
        if kappa <= 0:
            return (False, "κ = {} must be positive.".format(kappa))
        return (True, "κ = {} > 0, consistent with exponential decay.".format(kappa))

    def test_invariant_derivative_jump_relation(self):
        """
        Check the jump condition: ψ'(0⁺) - ψ'(0⁻) = (2m V0/ħ²) ψ(0).
        For even ansatz: ψ = A e^{-κ|x|}, jump = -2κA.
        So we must have -2κ = 2m V0/ħ².
        """
        if self.m is None or self.V0 is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data to check jump condition.")
        m_val = float(self.m)
        V0_val = float(self.V0)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        kappa_from_Eb = math.sqrt(2.0 * m_val * E_b_val) / hbar_val
        left = -2.0 * kappa_from_Eb
        right = 2.0 * m_val * V0_val / (hbar_val ** 2)
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Jump condition mismatch: left = {}, right = {}.".format(left, right))
        return (True, "Jump condition satisfied: -2κ = {} ≈ {}.".format(left, right))

    def test_invariant_kappa_from_V0(self):
        """
        Check that κ = -m V0/ħ².
        Use κ derived from E_b and compare with κ_target = -m V0/ħ².
        """
        if self.m is None or self.V0 is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data for κ consistency.")
        m_val = float(self.m)
        V0_val = float(self.V0)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        kappa_from_Eb = math.sqrt(2.0 * m_val * E_b_val) / hbar_val
        kappa_target = -m_val * V0_val / (hbar_val ** 2)
        if not math.isclose(kappa_from_Eb, kappa_target, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "κ mismatch: from E_b = {}, from V0 = {}.".format(kappa_from_Eb, kappa_target))
        return (True, "κ = {} agrees with target {}.".format(kappa_from_Eb, kappa_target))

    def test_invariant_normalization_possible(self):
        """Check that κ>0 ensures wavefunction decays to zero at infinity."""
        if self.m is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data for normalizability check.")
        m_val = float(self.m)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        kappa = math.sqrt(2.0 * m_val * E_b_val) / hbar_val
        if kappa <= 0:
            return (False, "κ must be >0 for normalizable wavefunction.")
        return (True, "κ = {} > 0, wavefunction decays as e^{-κ|x|} → 0.".format(kappa))

    # ===== Modeling/trace checks =====
    def test_modeling_energy_relation_direct(self):
        """
        Re-derive binding energy from given m, V0, hbar.
        Compare with provided E_b. This is permissible because we re-derive inside the checker.
        """
        if self.m is None or self.V0 is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data for energy relation.")
        m_val = float(self.m)
        V0_val = float(self.V0)
        hbar_val = float(self.hbar)
        expected_Eb = m_val * V0_val ** 2 / (2.0 * hbar_val ** 2)
        provided_Eb = float(self.E_b)
        if not math.isclose(expected_Eb, provided_Eb, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "E_b mismatch: expected = {}, provided = {}.".format(expected_Eb, provided_Eb))
        return (True, "E_b = {} matches re-derived value {}.".format(provided_Eb, expected_Eb))

    def test_modeling_explanation_step_presence(self):
        """Check that explanation steps contain key physical concepts."""
        if not self.explanation_steps:
            return (False, "No explanation steps provided; cannot verify modeling.")
        keywords = ['jump', 'derivative', 'binding', 'kappa', 'psi', 'Schrodinger', 'Schrödinger']
        found = any(
            any(kw in step.lower() for kw in keywords)
            for step in self.explanation_steps
        )
        if not found:
            return (False, "Explanation steps do not contain expected physical reasoning.")
        return (True, "Explanation steps include key physical concepts.")

    # ===== Monotonicity checks =====
    def test_monotonicity_binding_energy_increases_with_V0_magnitude(self):
        """
        If |V0| increases, binding energy should increase.
        Perturb V0 by +5% (more negative) and compare.
        """
        if self.m is None or self.V0 is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data for monotonicity test.")
        m_val = float(self.m)
        V0_val = float(self.V0)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        V0_new = V0_val * 1.05  # more negative
        expected_new_Eb = m_val * V0_new ** 2 / (2.0 * hbar_val ** 2)
        if E_b_val >= expected_new_Eb:
            return (False, "Monotonicity violated: old E_b = {}, new E_b (|V0|+5%) = {} should be larger.".format(E_b_val, expected_new_Eb))
        return (True, "Monotonicity holds: E_b increases from {} to {} when |V0| increased.".format(E_b_val, expected_new_Eb))

    def test_monotonicity_binding_energy_increases_with_mass(self):
        """
        If mass increases, binding energy increases.
        Perturb m by +10% and check.
        """
        if self.m is None or self.V0 is None or self.hbar is None or self.E_b is None:
            return (False, "Missing data for mass monotonicity test.")
        m_val = float(self.m)
        V0_val = float(self.V0)
        hbar_val = float(self.hbar)
        E_b_val = float(self.E_b)
        m_new = m_val * 1.10
        expected_new_Eb = m_new * V0_val ** 2 / (2.0 * hbar_val ** 2)
        if E_b_val >= expected_new_Eb:
            return (False, "Monotonicity with mass violated: old E_b = {}, new m+10% E_b = {}.".format(E_b_val, expected_new_Eb))
        return (True, "Monotonicity holds: E_b increases from {} to {} when mass increased.".format(E_b_val, expected_new_Eb))

    # ===== Weak answer sanity check =====
    def test_answer_sanity_positive_and_finite(self):
        """Binding energy must be positive and finite if V0 < 0."""
        if self.V0 is None or self.E_b is None:
            return (False, "Missing V0 or E_b for answer sanity.")
        if self.V0 >= 0:
            return (False, "V0 not negative, bound state not expected.")
        try:
            E_b_val = float(self.E_b)
        except (TypeError, ValueError):
            return (False, "E_b is not a number.")
        if math.isnan(E_b_val) or math.isinf(E_b_val):
            return (False, "E_b is not finite: {}.".format(E_b_val))
        if E_b_val <= 0:
            return (False, "E_b = {} must be positive for a bound state.".format(E_b_val))
        return (True, "E_b is positive and finite.")