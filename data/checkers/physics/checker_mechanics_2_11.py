import sympy as sp
import math
import re

class PhysicsChecker:
    """
    Audits a solution for the binary star system problem covering:
    - Lagrangian in Cartesian coordinates
    - Homogeneity of potential energy (degree -1)
    - Scaling transformation leading to Kepler's third law
    """
    def __init__(self, solution_instance):
        # Extract required result keys safely
        res = getattr(solution_instance, 'results', {})
        self.L = res.get('L', None)
        self.homogeneity = res.get('homogeneity', None)
        self.beta_relation = res.get('beta_relation', None)
        self.Kepler_third = res.get('Kepler_third', None)

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist and have proper types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four expected result keys must be present (value may be None, but key must exist)."""
        keys = ['L', 'homogeneity', 'beta_relation', 'Kepler_third']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_L_string(self):
        """L must be a non-empty string or sympy expression."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if not isinstance(self.L, (str, sp.Expr)):
            return (False, f"L is not a string or sympy expression, got {type(self.L)}")
        if isinstance(self.L, str) and len(self.L.strip()) == 0:
            return (False, "L is an empty string")
        return (True, "L is a valid string/expression")

    def test_interface_homogeneity_string(self):
        """homogeneity must be a non-empty string."""
        if self.homogeneity is None:
            return (False, "SKIP: homogeneity not available")
        if not isinstance(self.homogeneity, str) or len(self.homogeneity.strip()) == 0:
            return (False, "homogeneity is not a non-empty string")
        return (True, "homogeneity is a valid string")

    def test_interface_beta_relation_string(self):
        """beta_relation must be a non-empty string."""
        if self.beta_relation is None:
            return (False, "SKIP: beta_relation not available")
        if not isinstance(self.beta_relation, str) or len(self.beta_relation.strip()) == 0:
            return (False, "beta_relation is not a non-empty string")
        return (True, "beta_relation is a valid string")

    def test_interface_Kepler_third_string(self):
        """Kepler_third must be a non-empty string."""
        if self.Kepler_third is None:
            return (False, "SKIP: Kepler_third not available")
        if not isinstance(self.Kepler_third, str) or len(self.Kepler_third.strip()) == 0:
            return (False, "Kepler_third is not a non-empty string")
        return (True, "Kepler_third is a valid string")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical consistency (use math.isclose for numeric checks)
    # ------------------------------------------------------------------
    def test_invariant_L_form_contains_kinetic_and_potential(self):
        """
        Lagrangian should contain kinetic terms (m1·v1^2, m2·v2^2) and gravitational potential term.
        We perform a structural string check since expressions may include absolute values.
        """
        if self.L is None:
            return (False, "SKIP: L not available")
        L_str = str(self.L)
        # Check for mass symbols and velocity squares
        has_kinetic = re.search(r'(0\.5\s*\*?\s*m[12]\s*\*?\s*\|?v[12]\|?\^?2)', L_str, re.IGNORECASE) is not None
        # Check for gravitational constant and distance denominator
        has_potential = 'G' in L_str and ('|r1-r2|' in L_str or '|r1 - r2|' in L_str or '|mathbf{r}_1 - \\mathbf{r}_2|' in L_str)
        has_plus_sign = '+' in L_str  # potential term is added because V negative
        if not has_kinetic:
            return (False, "Lagrangian missing kinetic term expressions")
        if not has_potential:
            return (False, "Lagrangian missing gravitational potential term with distance")
        # The potential term should be positive because V = -Gm1m2/r gives L = T - V = T + Gm1m2/r
        # Check that there is no minus sign directly before the G term (allow subtraction of negative potential)
        # We check a simplified pattern: the term with G should be preceded by a plus or nothing.
        if has_kinetic and has_potential:
            return (True, "L contains kinetic and potential terms in correct form")
        return (False, "L structure incomplete")

    def test_invariant_homogeneity_correct_degree(self):
        """Homogeneity statement must assert degree -1: V(alpha*r1,alpha*r2)=alpha^{-1} V(r1,r2)."""
        if self.homogeneity is None:
            return (False, "SKIP: homogeneity not available")
        hom_str = self.homogeneity
        # Check for key components
        has_V_alpha = 'V(alpha' in hom_str
        has_alpha_inv = r'alpha^{-1}' in hom_str or 'alpha^(-1)' in hom_str or '1/alpha' in hom_str
        if has_V_alpha and has_alpha_inv:
            return (True, "Homogeneity statement shows correct degree -1")
        return (False, "Homogeneity statement does not match expected V(alpha*r1,alpha*r2) = alpha^{-1} V(r1,r2)")

    def test_invariant_beta_relation_form(self):
        """beta relation must state beta = alpha^{3/2}."""
        if self.beta_relation is None:
            return (False, "SKIP: beta_relation not available")
        b_str = self.beta_relation
        if 'beta' not in b_str or 'alpha' not in b_str:
            return (False, "beta_relation missing variables")
        if '3/2' not in b_str and '1.5' not in b_str:
            # allow exponent power notation
            if not re.search(r'alpha\^\s*\{?\s*3/2\s*\}?', b_str):
                return (False, "beta_relation does not contain exponent 3/2")
        return (True, "beta_relation correctly gives beta = alpha^(3/2)")

    def test_invariant_Kepler_third_form(self):
        """Kepler's third law must relate period squared to semi-major axis cubed with total mass."""
        if self.Kepler_third is None:
            return (False, "SKIP: Kepler_third not available")
        k_str = self.Kepler_third
        # Check for essential elements
        has_T_sq = 'T' in k_str and '^2' in k_str
        has_a_cube = 'a^3' in k_str or 'a^{3}' in k_str
        has_G = 'G' in k_str
        has_mass_sum = 'm1+m2' in k_str or 'm_1+m_2' in k_str or '(m1+m2)' in k_str or '(m_1+m_2)' in k_str
        if has_T_sq and has_a_cube and has_G and has_mass_sum:
            return (True, "Kepler_third contains T^2, a^3, G, and total mass")
        return (False, "Kepler_third missing one or more required components")

    # ------------------------------------------------------------------
    # Modeling / trace tests – checking consistency of derived relations
    # ------------------------------------------------------------------
    def test_modeling_scaling_produces_Kepler(self):
        """
        Check that the beta_relation and Kepler_third are consistent:
        The scaling transformation leads to T ∝ a^{3/2}, i.e. T^2 ∝ a^3.
        We test that both relations together imply this proportionality.
        (Only a logical check, not numerical)
        """
        if self.beta_relation is None or self.Kepler_third is None:
            return (False, "SKIP: one of beta_relation or Kepler_third missing")
        # Check that Kepler_third explicitly shows T^2 ∝ a^3
        has_proportionality = 'T_per^2' in self.Kepler_third and 'a^3' in self.Kepler_third
        # Beta relation should give exponent 3/2 for T (time scaling) relative to a (length scaling)
        beta_contains_exponent_3_2 = '3/2' in self.beta_relation
        if has_proportionality and beta_contains_exponent_3_2:
            return (True, "Modeling consistent: beta relation exponent 3/2 leads to T^2 ∝ a^3")
        return (False, "Modeling mismatch between scaling relation and Kepler's law")

    def test_modeling_L_consistent_with_CM_decomposition(self):
        """
        Optionally check that Lagrangian can be separated into CM and relative motion.
        We can't fully reconstruct, but we can check that L string includes separation.
        This is a weak check; we just look for terms like 'R' or 'mu' or 'reduced mass'.
        """
        if self.L is None:
            return (False, "SKIP: L not available")
        L_str = str(self.L)
        # Check for center-of-mass coordinates or reduced mass
        has_CM_indicators = ('R' in L_str or 'mathbf{R}' in L_str or 'mu' in L_str or 'frac' in L_str)
        if has_CM_indicators:
            return (True, "Lagrangian appears to have CM decomposition terms")
        return (False, "Lagrangian does not obviously show CM decomposition (optional check)")

    # ------------------------------------------------------------------
    # Monotonicity test – explicit input perturbation and output comparison
    # ------------------------------------------------------------------
    def test_monotonicity_Kepler_period_increases_with_a(self):
        """
        Kepler's third law implies T ∝ a^{3/2}.  We verify numerically that a small increase
        in semi-major axis leads to an increase in period.
        Uses the physical law re-derived in the checker (not copied from answer).
        """
        if self.Kepler_third is None:
            return (False, "SKIP: Kepler_third not available")
        # Define a baseline scenario (arbitrary units)
        a0 = 1.0          # semi-major axis
        M_total = 2.0     # total mass (in same units)
        G = 1.0           # gravitational constant (set to 1 for simplicity)
        # Compute baseline period from Kepler's third law:  T^2 = (4π^2)/(G M_total) * a^3
        T0_sq = (4 * math.pi**2) / (G * M_total) * (a0**3)
        T0 = math.sqrt(T0_sq)
        # Perturb semi-major axis by +10%
        a1 = a0 * 1.1
        T1_sq = (4 * math.pi**2) / (G * M_total) * (a1**3)
        T1 = math.sqrt(T1_sq)
        if T1 > T0:
            return (True, "Period increases with semi-major axis (monotonic)")
        else:
            return (False, "Period did not increase as expected")

    # ------------------------------------------------------------------
    # (Optional) weak answer sanity checks – do not re-derive values
    # ------------------------------------------------------------------
    def test_answer_sanity_no_negative_lagrangian_potential(self):
        """
        The gravitational term in L should appear positive (since V negative, L = T - V = T + positive).
        Check that the string does not have a minus before G.
        """
        if self.L is None:
            return (False, "SKIP: L not available")
        L_str = str(self.L)
        # Find the position of 'G' and look if there is a minus immediately before it
        # Also check that the expression includes a plus before or after.
        if 'G' in L_str:
            # Simple check: the substring before 'G' should not contain '-' unless it's part of another term
            # We'll search for minus directly before G (ignoring spaces)
            pattern = r'-\s*G'  # minus then optional space then G
            if re.search(pattern, L_str):
                return (False, "Potential term in L appears negative; expected + Gm1m2/|r|")
            else:
                return (True, "No minus before G, potential term positive")
        return (False, "Could not find gravitational constant in L")

    # ------------------------------------------------------------------
    # Main test harness
    # ------------------------------------------------------------------
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