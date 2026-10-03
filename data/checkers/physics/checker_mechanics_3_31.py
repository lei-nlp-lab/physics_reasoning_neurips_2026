import math
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results; default None for missing keys
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.M0 = self.results.get('M0', None)
        self.V = self.results.get('V', None)
        self.v2 = self.results.get('v2', None)
        self.v2_1 = self.results.get('v2_1', None)
        self.v2_2 = self.results.get('v2_2', None)
        # Givens (hardcoded from problem statement)
        self.m1 = 1.0          # g
        self.m2 = 10.0         # g
        self.c = 1.0           # natural units

    # ----------------------------------------------------------------------
    # Helper: extract numeric value from result (float or string with units)
    # ----------------------------------------------------------------------
    def _to_float(self, val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Try to extract the first number (including decimal)
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if m:
                return float(m.group())
        return None

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all five expected keys exist in results."""
        expected = ['M0', 'V', 'v2', 'v2_1', 'v2_2']
        missing = [k for k in expected if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_not_none(self):
        """Check that none of the required results is None."""
        keys = ['M0', 'V', 'v2', 'v2_1', 'v2_2']
        none_keys = [k for k in keys if getattr(self, k) is None]
        if none_keys:
            return (False, f"None results: {none_keys}")
        return (True, "All results are non-None")

    def test_interface_numeric_type(self):
        """Check that all results can be converted to floats."""
        for name in ['M0', 'V', 'v2', 'v2_1', 'v2_2']:
            val = getattr(self, name)
            num = self._to_float(val)
            if num is None:
                return (False, f"{name} cannot be converted to number")
        return (True, "All results are numeric or parseable")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_part_a_composite_mass(self):
        """Check that M0 satisfies invariant mass formula for part (a)."""
        M0_num = self._to_float(self.M0)
        if M0_num is None:
            return (False, "SKIP: M0 not available")
        # Givens for part (a): v1=0.9c, v2=0
        v1 = 0.9
        v2 = 0.0
        gamma1 = 1.0 / math.sqrt(1 - v1**2)
        E1 = gamma1 * self.m1
        E2 = self.m2        # since v2=0, gamma=1
        p1 = gamma1 * self.m1 * v1
        p2 = 0.0
        Etot = E1 + E2
        ptot = p1 + p2
        M0_calc = math.sqrt(Etot**2 - ptot**2)   # c=1
        if math.isclose(M0_calc, M0_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M0={M0_num} matches invariant mass formula (calc={M0_calc:.3f})")
        else:
            return (False, f"M0={M0_num} does not match invariant mass formula (calc={M0_calc:.3f})")

    def test_invariant_part_a_velocity(self):
        """Check that V satisfies p_tot / E_tot for part (a)."""
        V_num = self._to_float(self.V)
        if V_num is None:
            return (False, "SKIP: V not available")
        v1 = 0.9
        v2 = 0.0
        gamma1 = 1.0 / math.sqrt(1 - v1**2)
        E1 = gamma1 * self.m1
        E2 = self.m2
        p1 = gamma1 * self.m1 * v1
        p2 = 0.0
        Etot = E1 + E2
        ptot = p1 + p2
        V_calc = ptot / Etot
        if math.isclose(V_calc, V_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V={V_num} matches p_tot/E_tot (calc={V_calc:.3f})")
        else:
            return (False, f"V={V_num} does not match p_tot/E_tot (calc={V_calc:.3f})")

    def test_invariant_part_b_mass_match(self):
        """Check that with v2 = results['v2'] and v1=0, invariant mass equals M0."""
        v2_num = self._to_float(self.v2)
        M0_num = self._to_float(self.M0)
        if v2_num is None or M0_num is None:
            return (False, "SKIP: v2 or M0 not available")
        v1 = 0.0
        gamma2 = 1.0 / math.sqrt(1 - v2_num**2)
        E1 = self.m1
        E2 = gamma2 * self.m2
        p1 = 0.0
        p2 = gamma2 * self.m2 * v2_num
        Etot = E1 + E2
        ptot = p1 + p2
        M0_calc = math.sqrt(Etot**2 - ptot**2)
        if math.isclose(M0_calc, M0_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Part (b) invariant mass matches M0 (calc={M0_calc:.3f})")
        else:
            return (False, f"Part (b) invariant mass {M0_calc:.3f} != M0 {M0_num}")

    def test_invariant_part_c_velocity_match(self):
        """Check that with v2_1 or v2_2 and v1=0, composite velocity equals V."""
        V_num = self._to_float(self.V)
        v2_1_num = self._to_float(self.v2_1)
        v2_2_num = self._to_float(self.v2_2)
        if V_num is None or v2_1_num is None or v2_2_num is None:
            return (False, "SKIP: V, v2_1, or v2_2 not available")
        v1 = 0.0
        # Check both roots
        results_ok = []
        for root_val in [v2_1_num, v2_2_num]:
            gamma2 = 1.0 / math.sqrt(1 - root_val**2)
            Etot = self.m1 + gamma2 * self.m2
            ptot = gamma2 * self.m2 * root_val
            V_calc = ptot / Etot
            if math.isclose(V_calc, V_num, rel_tol=1e-2, abs_tol=1e-2):
                results_ok.append(f"v2={root_val:.3f} gives V={V_calc:.3f}")
        if results_ok:
            return (True, "Found matching v2 values: " + "; ".join(results_ok))
        else:
            return (False, f"Neither v2_1={v2_1_num} nor v2_2={v2_2_num} yields V={V_num} (within tolerance)")

    # ----------------------------------------------------------------------
    # Modeling/trace checks (relations from explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_part_a_mass_formula(self):
        """Check that M0 follows the formula m = sqrt(m1^2 + m2^2 + 2*gamma1*m1*m2)."""
        M0_num = self._to_float(self.M0)
        if M0_num is None:
            return (False, "SKIP: M0 not available")
        v1 = 0.9
        gamma1 = 1.0 / math.sqrt(1 - v1**2)
        m_calc = math.sqrt(self.m1**2 + self.m2**2 + 2 * gamma1 * self.m1 * self.m2)
        if math.isclose(m_calc, M0_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M0={M0_num} matches formula {m_calc:.3f}")
        else:
            return (False, f"M0={M0_num} does not match formula {m_calc:.3f}")

    def test_modeling_part_b_gamma2(self):
        """Check that gamma2 derived from v2 satisfies the invariant-mass relation for part (b)."""
        v2_num = self._to_float(self.v2)
        M0_num = self._to_float(self.M0)
        if v2_num is None or M0_num is None:
            return (False, "SKIP: v2 or M0 not available")
        gamma2 = 1.0 / math.sqrt(1 - v2_num**2)
        # From M0^2 = m1^2 + m2^2 + 2*m1*m2*gamma2
        expected_gamma2 = (M0_num**2 - self.m1**2 - self.m2**2) / (2 * self.m1 * self.m2)
        if math.isclose(gamma2, expected_gamma2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"gamma2={gamma2:.3f} matches expected {expected_gamma2:.3f}")
        else:
            return (False, f"gamma2={gamma2:.3f} != expected {expected_gamma2:.3f}")

    def test_modeling_part_c_quadratic(self):
        """Check that v2_1 and v2_2 satisfy the quadratic derived in explanation steps for part (c)."""
        V_num = self._to_float(self.V)
        v2_1_num = self._to_float(self.v2_1)
        v2_2_num = self._to_float(self.v2_2)
        if V_num is None or v2_1_num is None or v2_2_num is None:
            return (False, "SKIP: V, v2_1, or v2_2 not available")
        beta = V_num
        m1 = self.m1
        m2 = self.m2
        # Quadratic: (m2^2 + m1^2 beta^2) beta2^2 - 2 m2^2 beta beta2 + (m2^2 - m1^2) beta^2 = 0
        A = m2**2 + m1**2 * beta**2
        B = -2 * m2**2 * beta
        C = (m2**2 - m1**2) * beta**2
        passes = []
        for root in [v2_1_num, v2_2_num]:
            val = A * root**2 + B * root + C
            if math.isclose(val, 0.0, abs_tol=1e-2):
                passes.append(f"v2={root:.3f} gives residual {val:.5f}")
        if passes:
            return (True, "Quadratic satisfied: " + "; ".join(passes))
        else:
            return (False, f"No root satisfies quadratic (A={A:.5f}, B={B:.5f}, C={C:.5f})")

    # ----------------------------------------------------------------------
    # Monotonicity checks (trend checks via parameter perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_part_a_velocity_increases_with_v1(self):
        """Check that V increases if incident velocity v1 is perturbed upward."""
        V_num = self._to_float(self.V)
        if V_num is None:
            return (False, "SKIP: V not available")
        # Baseline V computed from givens
        v1_base = 0.9
        gamma1_base = 1.0 / math.sqrt(1 - v1_base**2)
        Etot_base = gamma1_base * self.m1 + self.m2
        ptot_base = gamma1_base * self.m1 * v1_base
        V_base = ptot_base / Etot_base
        # Perturb v1 upward by 10%
        v1_pert = v1_base * 1.1  # 0.99, still < c
        if v1_pert >= 1.0:
            # If perturbation would exceed c, reduce to 0.99c
            v1_pert = 0.99
        gamma1_pert = 1.0 / math.sqrt(1 - v1_pert**2)
        Etot_pert = gamma1_pert * self.m1 + self.m2
        ptot_pert = gamma1_pert * self.m1 * v1_pert
        V_pert = ptot_pert / Etot_pert
        if V_pert > V_base:
            return (True, f"Increasing v1 from {v1_base} to {v1_pert:.3f} increases V from {V_base:.4f} to {V_pert:.4f}")
        else:
            return (False, f"V did not increase as expected: base={V_base:.4f}, perturbed={V_pert:.4f}")

    def test_monotonicity_part_b_speed_decreases_with_m1(self):
        """Check that required v2 decreases if stationary mass m1 is increased."""
        v2_num = self._to_float(self.v2)
        M0_num = self._to_float(self.M0)
        if v2_num is None or M0_num is None:
            return (False, "SKIP: v2 or M0 not available")
        # Compute baseline gamma2 from v2
        gamma2_base = 1.0 / math.sqrt(1 - v2_num**2)
        # Perturb m1 upward by 10%
        m1_pert = self.m1 * 1.1
        # Expected gamma2 from invariant mass: gamma2_expected = (M0^2 - m1^2 - m2^2) / (2*m1*m2)
        # Compute baseline gamma2 from formula to verify monotonicity direction
        gamma2_from_formula_base = (M0_num**2 - self.m1**2 - self.m2**2) / (2 * self.m1 * self.m2)
        gamma2_from_formula_pert = (M0_num**2 - m1_pert**2 - self.m2**2) / (2 * m1_pert * self.m2)
        if gamma2_from_formula_pert < gamma2_from_formula_base:
            # Since gamma decreases, v2 decreases (monotonic)
            return (True, f"Increasing m1 from {self.m1} to {m1_pert:.2f} reduces gamma2 from {gamma2_from_formula_base:.3f} to {gamma2_from_formula_pert:.3f}, implying v2 decreases")
        else:
            return (False, f"Gamma2 did not decrease: base={gamma2_from_formula_base:.3f}, perturbed={gamma2_from_formula_pert:.3f}")

    def test_monotonicity_part_c_larger_root_increases_with_m1(self):
        """Check that the larger root v2_1 (= (1+m1/m2)*beta) increases if m1 is increased."""
        V_num = self._to_float(self.V)
        v2_1_num = self._to_float(self.v2_1)
        if V_num is None or v2_1_num is None:
            return (False, "SKIP: V or v2_1 not available")
        beta = V_num
        # Baseline value from formula
        v2_1_base = (1 + self.m1/self.m2) * beta
        # Perturb m1 upward by 10%
        m1_pert = self.m1 * 1.1
        v2_1_pert = (1 + m1_pert/self.m2) * beta
        if v2_1_pert > v2_1_base:
            return (True, f"Increasing m1 from {self.m1} to {m1_pert:.2f} increases larger root from {v2_1_base:.4f} to {v2_1_pert:.4f}")
        else:
            return (False, f"Larger root did not increase: base={v2_1_base:.4f}, perturbed={v2_1_pert:.4f}")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_physicality(self):
        """Basic checks: all masses positive, all speeds between 0 and c."""
        M0_num = self._to_float(self.M0)
        V_num = self._to_float(self.V)
        v2_num = self._to_float(self.v2)
        v2_1_num = self._to_float(self.v2_1)
        v2_2_num = self._to_float(self.v2_2)
        issues = []
        if M0_num is not None and M0_num <= 0:
            issues.append("M0 <= 0")
        if V_num is not None and not (0 < V_num < 1):
            issues.append("V not in (0,1)")
        if v2_num is not None and not (0 < v2_num < 1):
            issues.append("v2 not in (0,1)")
        if v2_1_num is not None and not (0 < v2_1_num < 1):
            issues.append("v2_1 not in (0,1)")
        if v2_2_num is not None and not (0 < v2_2_num < 1):
            issues.append("v2_2 not in (0,1)")
        if issues:
            return (False, "Issues: " + ", ".join(issues))
        return (True, "All available values pass basic physicality checks")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
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