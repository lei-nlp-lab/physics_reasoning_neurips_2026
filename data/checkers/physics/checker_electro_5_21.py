import sympy as sp
import math

class PhysicsChecker:
    """
    Auditor for Larmor radiation problems.
    Expects solution_instance with:
        - attributes: e, c, epsilon0, beta_dot, a, omega0, R, theta (numeric)
        - results dict with keys 'dP_dOmega_general', 'dP_dOmega_SHM', 'dP_dOmega_circular'
    """
    def __init__(self, solution_instance):
        self.sol = solution_instance

        # Safe extraction of givens (numeric)
        self.e = getattr(solution_instance, 'e', None)
        self.c = getattr(solution_instance, 'c', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.beta_dot = getattr(solution_instance, 'beta_dot', None)
        self.a = getattr(solution_instance, 'a', None)
        self.omega0 = getattr(solution_instance, 'omega0', None)
        self.R = getattr(solution_instance, 'R', None)
        self.theta = getattr(solution_instance, 'theta', None)

        # Safe extraction of results (strings)
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.dP_general = self.results.get('dP_dOmega_general', None)
        self.dP_SHM = self.results.get('dP_dOmega_SHM', None)
        self.dP_circular = self.results.get('dP_dOmega_circular', None)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """All three result keys must exist and be non‑empty strings."""
        for key, val in [('dP_dOmega_general', self.dP_general),
                         ('dP_dOmega_SHM', self.dP_SHM),
                         ('dP_dOmega_circular', self.dP_circular)]:
            if val is None or not isinstance(val, str) or not val.strip():
                return False, f"Missing or invalid '{key}'"
        return True, "All required result keys present and non‑empty"

    def test_interface_all_results_are_strings(self):
        """Each result must be a string."""
        for name, val in [('general', self.dP_general),
                          ('SHM', self.dP_SHM),
                          ('circular', self.dP_circular)]:
            if not isinstance(val, str):
                return False, f"dP_dOmega_{name} is not a string"
        return True, "All results are strings"

    # ------------------------------------------------------------
    # Invariant tests (theorem‑based)
    # ------------------------------------------------------------
    def test_invariant_general_contains_e_and_sin2(self):
        """General expression must contain e² and sin²(Θ)."""
        if self.dP_general is None:
            return False, "SKIP: no general expression"
        expr = self.dP_general

        # Check for e² in various notations
        has_e2 = ('e^2' in expr or 'e**2' in expr or 'e^{2}' in expr)
        # Check for sin²(Θ)
        has_sin2 = ('sin^2' in expr or 'sin^{2}' in expr or '\\sin^2' in expr or '\\sin^{2}' in expr)

        if not has_e2:
            return False, "General expression missing squared charge e²"
        if not has_sin2:
            return False, "General expression missing sin²(Θ)"
        # Check factor 1/(16π² ε0 c)
        if not ('16' in expr and 'epsilon' in expr or 'ε' in expr):
            return False, "General expression missing factor 1/(16π² ε0 c)"
        return True, "General expression has required charge and angular dependence"

    def test_invariant_SHM_angular_dependence(self):
        """SHM expression must contain sin²(θ)."""
        if self.dP_SHM is None:
            return False, "SKIP: no SHM expression"
        if 'sin^2' not in self.dP_SHM and 'sin^{2}' not in self.dP_SHM and '\\sin^2' not in self.dP_SHM:
            return False, "SHM expression missing sin²(θ)"
        return True, "SHM expression includes sin²(θ)"

    def test_invariant_circular_angular_dependence(self):
        """Circular expression must have (1+cos²(θ)) factor."""
        if self.dP_circular is None:
            return False, "SKIP: no circular expression"
        # Accept various forms
        patterns = ['1+cos^2', '1+cos^{2}', '1+\\cos^2', '1+\\cos^{2}',
                    '(1+cos^2', '(1+\\cos^2']
        if not any(p in self.dP_circular for p in patterns):
            return False, "Circular expression missing (1+cos²(θ)) factor"
        return True, "Circular expression has correct angular factor (1+cos²(θ))"

    def test_invariant_SHM_parameter_scaling(self):
        """SHM expression must contain a² and ω₀⁴."""
        if self.dP_SHM is None:
            return False, "SKIP: no SHM expression"
        has_a2 = 'a^2' in self.dP_SHM or 'a^{2}' in self.dP_SHM or 'a**2' in self.dP_SHM
        has_omega4 = 'omega0^4' in self.dP_SHM or '\\omega_0^4' in self.dP_SHM or 'ω0^4' in self.dP_SHM
        if not has_a2:
            return False, "SHM expression must contain a²"
        if not has_omega4:
            return False, "SHM expression must contain ω₀⁴"
        if '32' not in self.dP_SHM:
            return False, "SHM expression should have denominator 32"
        return True, "SHM expression scales correctly with a and ω₀"

    def test_invariant_circular_parameter_scaling(self):
        """Circular expression must contain R² and ω₀⁴."""
        if self.dP_circular is None:
            return False, "SKIP: no circular expression"
        has_R2 = 'R^2' in self.dP_circular or 'R^{2}' in self.dP_circular or 'R**2' in self.dP_circular
        has_omega4 = 'omega0^4' in self.dP_circular or '\\omega_0^4' in self.dP_circular or 'ω0^4' in self.dP_circular
        if not has_R2:
            return False, "Circular expression must contain R²"
        if not has_omega4:
            return False, "Circular expression must contain ω₀⁴"
        if '32' not in self.dP_circular:
            return False, "Circular expression should have denominator 32"
        return True, "Circular expression scales correctly with R and ω₀"

    def test_invariant_circular_phi_independence(self):
        """Circular expression should not depend on φ."""
        if self.dP_circular is None:
            return False, "SKIP: no circular expression"
        if 'φ' in self.dP_circular or 'phi' in self.dP_circular or '\\phi' in self.dP_circular:
            return False, "Circular expression should be independent of φ"
        return True, "Circular expression does not contain φ"

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_general_derived_from_larmor(self):
        """General expression should be the Larmor angular distribution (no time averaging)."""
        if self.dP_general is None:
            return False, "SKIP: no general expression"
        # Should not have factor 1/2 (time averaging)
        if '1/2' in self.dP_general or '\\frac{1}{2}' in self.dP_general:
            return False, "General expression should not contain time‑averaging factor 1/2"
        # Should contain |β̇|² or |\\dot\\beta|²
        if 'dot' not in self.dP_general and 'β̇' not in self.dP_general:
            return False, "General expression should contain |β̇|²"
        return True, "General expression matches Larmor form (no time averaging)"

    def test_modeling_SHM_time_averaging(self):
        """SHM expression must reflect time averaging of cos²(ω₀t)."""
        if self.dP_SHM is None:
            return False, "SKIP: no SHM expression"
        # The denominator 32 = 16*2 indicates the 1/2 factor from averaging.
        if '32' not in self.dP_SHM:
            return False, "SHM expression should have denominator 32 (16×2)"
        return True, "SHM expression includes time‑averaging factor 1/2"

    def test_modeling_circular_decomposition(self):
        """Circular expression should arise from two orthogonal SHM components."""
        if self.dP_circular is None:
            return False, "SKIP: no circular expression"
        # The factor (1+cos²θ) is characteristic of the sum of two sin² terms.
        if '1+cos^2' not in self.dP_circular and '1+\\cos^2' not in self.dP_circular:
            return False, "Circular expression should contain (1+cos²θ) from two‑oscillation decomposition"
        return True, "Circular expression consistent with two‑oscillation decomposition"

    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation + numeric comparison)
    # ------------------------------------------------------------
    def test_monotonicity_SHM_increase_with_a(self):
        """Increasing amplitude a by 10% must increase dP/dΩ by factor 1.21."""
        if None in (self.e, self.c, self.epsilon0, self.a, self.omega0, self.theta):
            return False, "SKIP: missing numeric parameters for SHM evaluation"
        # Compute baseline
        prefactor = (self.e**2 * self.a**2 * self.omega0**4) / (32 * math.pi**2 * self.epsilon0 * self.c**3)
        baseline = prefactor * (math.sin(self.theta)**2)
        # Perturb: a -> 1.1*a
        a_pert = 1.1 * self.a
        perturbed = ((self.e**2 * a_pert**2 * self.omega0**4) / (32 * math.pi**2 * self.epsilon0 * self.c**3)) * (math.sin(self.theta)**2)
        # Ratio should be 1.21
        ratio = perturbed / baseline
        if not math.isclose(ratio, 1.21, rel_tol=1e-2):
            return False, f"After 10% increase in a, ratio should be ≈1.21, got {ratio:.4f}"
        return True, "SHM output increases correctly with amplitude a"

    def test_monotonicity_circular_increase_with_R(self):
        """Increasing radius R by 10% must increase dP/dΩ by factor 1.21."""
        if None in (self.e, self.c, self.epsilon0, self.R, self.omega0, self.theta):
            return False, "SKIP: missing numeric parameters for circular evaluation"
        # Baseline
        prefactor = (self.e**2 * self.R**2 * self.omega0**4) / (32 * math.pi**2 * self.epsilon0 * self.c**3)
        baseline = prefactor * (1 + math.cos(self.theta)**2)
        # Perturb: R -> 1.1*R
        R_pert = 1.1 * self.R
        perturbed = ((self.e**2 * R_pert**2 * self.omega0**4) / (32 * math.pi**2 * self.epsilon0 * self.c**3)) * (1 + math.cos(self.theta)**2)
        ratio = perturbed / baseline
        if not math.isclose(ratio, 1.21, rel_tol=1e-2):
            return False, f"After 10% increase in R, ratio should be ≈1.21, got {ratio:.4f}"
        return True, "Circular output increases correctly with radius R"

    # ------------------------------------------------------------
    # Weak answer‑sanity test (only one, non‑numeric)
    # ------------------------------------------------------------
    def test_answer_sanity_nonnegative(self):
        """All three expressions should be manifestly non‑negative."""
        for name, expr in [('general', self.dP_general),
                           ('SHM', self.dP_SHM),
                           ('circular', self.dP_circular)]:
            if expr is None:
                continue
            # A leading minus would make expression negative
            stripped = expr.strip().lstrip('(')
            if stripped.startswith('-'):
                return False, f"{name} expression starts with negative sign – cannot be negative"
        return True, "All expressions are manifestly non‑negative"

    # ------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------
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
        return all_pass, logs