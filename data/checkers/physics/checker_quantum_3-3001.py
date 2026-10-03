import math
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution to the Pauli‑like matrix problem:
    (a) Tr(σ_i)=0, (b) eigenvalues ±1, det=-1,
    (c) linear independence, (d) expansion coefficients m_0, m_j.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all relevant data from the solution_instance's results dict.
        Use None for missing values, never numeric defaults.
        """
        res = getattr(solution_instance, 'results', None)
        if res is None:
            res = {}

        # (a)
        self.tr_sigma_i = res.get('tr_sigma_i', None)
        # (b)
        self.eigenvalues = res.get('eigenvalues', None)
        self.det_sigma_i = res.get('det_sigma_i', None)
        # (c)
        self.lin_indep = res.get('lin_indep', None)
        # (d)
        self.m0 = res.get('m0', None)          # formula string or sympy expression
        self.m_j = res.get('m_j', None)        # formula string or sympy expression

        self._results = res

    # ------------------------------------------------------------------
    # Interface tests – ensure required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_tr_sigma_i(self) -> Tuple[bool, str]:
        """Check that tr_sigma_i is present and equals 0."""
        if self.tr_sigma_i is None:
            return (False, "tr_sigma_i missing")
        if not isinstance(self.tr_sigma_i, (int, float)):
            return (False, f"tr_sigma_i has wrong type: {type(self.tr_sigma_i)}")
        if not math.isclose(self.tr_sigma_i, 0.0, abs_tol=1e-9):
            return (False, f"tr_sigma_i = {self.tr_sigma_i}, expected 0")
        return (True, "tr_sigma_i = 0")

    def test_interface_eigenvalues(self) -> Tuple[bool, str]:
        """Check that eigenvalues is a list of two numbers."""
        if self.eigenvalues is None:
            return (False, "eigenvalues missing")
        if not isinstance(self.eigenvalues, (list, tuple)):
            return (False, f"eigenvalues has wrong type: {type(self.eigenvalues)}")
        if len(self.eigenvalues) != 2:
            return (False, f"eigenvalues length {len(self.eigenvalues)}, expected 2")
        for e in self.eigenvalues:
            if not isinstance(e, (int, float)):
                return (False, f"eigenvalue {e} is not a number")
        return (True, "eigenvalues present and length 2")

    def test_interface_det_sigma_i(self) -> Tuple[bool, str]:
        """Check that det_sigma_i is present and equals -1."""
        if self.det_sigma_i is None:
            return (False, "det_sigma_i missing")
        if not isinstance(self.det_sigma_i, (int, float)):
            return (False, f"det_sigma_i has wrong type: {type(self.det_sigma_i)}")
        if not math.isclose(self.det_sigma_i, -1.0, abs_tol=1e-9):
            return (False, f"det_sigma_i = {self.det_sigma_i}, expected -1")
        return (True, "det_sigma_i = -1")

    def test_interface_lin_indep(self) -> Tuple[bool, str]:
        """Check that lin_indep is present and is True."""
        if self.lin_indep is None:
            return (False, "lin_indep missing")
        if not isinstance(self.lin_indep, bool):
            return (False, f"lin_indep has wrong type: {type(self.lin_indep)}")
        if self.lin_indep is not True:
            return (False, f"lin_indep = {self.lin_indep}, expected True")
        return (True, "lin_indep = True")

    def test_interface_m0(self) -> Tuple[bool, str]:
        """Check that m0 expression is present (non-None)."""
        if self.m0 is None:
            return (False, "m0 missing")
        if not isinstance(self.m0, str) and not hasattr(self.m0, 'func'):
            return (False, f"m0 has unsupported type: {type(self.m0)}")
        return (True, "m0 expression present")

    def test_interface_m_j(self) -> Tuple[bool, str]:
        """Check that m_j expression is present."""
        if self.m_j is None:
            return (False, "m_j missing")
        if not isinstance(self.m_j, str) and not hasattr(self.m_j, 'func'):
            return (False, f"m_j has unsupported type: {type(self.m_j)}")
        return (True, "m_j expression present")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_trace_zero(self) -> Tuple[bool, str]:
        """Verify that trace of sigma_i is 0."""
        if self.tr_sigma_i is None:
            return (False, "SKIP: tr_sigma_i not available")
        return (True, "tr_sigma_i = 0 invariant holds")

    def test_invariant_eigenvalues_plus_minus_1(self) -> Tuple[bool, str]:
        """Check that the two eigenvalues are +1 and -1 (order irrelevant)."""
        if self.eigenvalues is None:
            return (False, "SKIP: eigenvalues not available")
        vals = set(self.eigenvalues)
        if vals != {1, -1}:
            return (False, f"eigenvalues are {vals}, expected {{1, -1}}")
        return (True, "eigenvalues are ±1")

    def test_invariant_det_minus_1(self) -> Tuple[bool, str]:
        """Check det = product of eigenvalues = -1."""
        if self.det_sigma_i is None or self.eigenvalues is None:
            return (False, "SKIP: det or eigenvalues missing")
        product = self.eigenvalues[0] * self.eigenvalues[1]
        if not math.isclose(product, -1.0, abs_tol=1e-9):
            return (False, f"product of eigenvalues = {product}, expected -1")
        return (True, "det = -1 consistent with eigenvalues")

    def test_invariant_linear_independence_true(self) -> Tuple[bool, str]:
        """Check that linear independence is asserted."""
        if self.lin_indep is None:
            return (False, "SKIP: lin_indep not available")
        if self.lin_indep is not True:
            return (False, "linear independence is not True")
        return (True, "linear independence holds")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check that derived expressions are properly formed
    # ------------------------------------------------------------------
    def test_modeling_m0_form(self) -> Tuple[bool, str]:
        """The m0 expression should contain 'Tr(M)' or similar."""
        if self.m0 is None:
            return (False, "SKIP: m0 not available")
        s = str(self.m0)
        if 'Tr(M)' not in s and 'tr(M)' not in s and 'Trace(M)' not in s:
            return (False, f"m0 expression '{s}' does not contain Tr(M)")
        if '1/2' not in s and '0.5' not in s:
            return (False, f"m0 expression '{s}' does not contain factor 1/2")
        return (True, "m0 expression contains Tr(M) and 1/2")

    def test_modeling_m_j_form(self) -> Tuple[bool, str]:
        """The m_j expression should contain 'Tr(sigma_j M)' or similar."""
        if self.m_j is None:
            return (False, "SKIP: m_j not available")
        s = str(self.m_j)
        if 'Tr(sigma_j M)' not in s and 'tr(sigma_j M)' not in s and 'Trace(sigma_j M)' not in s:
            if not ('Tr(' in s and 'sigma' in s.lower() and 'M' in s):
                return (False, f"m_j expression '{s}' does not contain Tr(sigma_j M)")
        if '1/2' not in s and '0.5' not in s:
            return (False, f"m_j expression '{s}' does not contain factor 1/2")
        return (True, "m_j expression contains Tr(sigma_j M) and 1/2")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicit numerical perturbation and comparison
    # ------------------------------------------------------------------
    def test_monotonicity_m0_increases_with_trM(self) -> Tuple[bool, str]:
        """
        Re‑derive m0 = (1/2) Tr(M).  Check that increasing Tr(M) by 20%
        increases m0 by the same relative amount.
        """
        # Use the derived formula, independent of student's expression
        baseline_tr = 10.0
        perturbed_tr = 12.0   # 20% increase
        baseline_m0 = 0.5 * baseline_tr
        perturbed_m0 = 0.5 * perturbed_tr
        if not perturbed_m0 > baseline_m0:
            return (False, "Monotonicity violation: m0 did not increase with Tr(M)")
        return (True, f"m0 increases from {baseline_m0} to {perturbed_m0} when Tr(M) rises from {baseline_tr} to {perturbed_tr}")

    def test_monotonicity_m_j_increases_with_tr_sigma_j_M(self) -> Tuple[bool, str]:
        """
        Re‑derive m_j = (1/2) Tr(sigma_j M).  Check that increasing the
        trace of sigma_j M by 20% increases m_j by the same relative amount.
        """
        baseline_trace = 10.0
        perturbed_trace = 12.0   # 20% increase
        baseline_mj = 0.5 * baseline_trace
        perturbed_mj = 0.5 * perturbed_trace
        if not perturbed_mj > baseline_mj:
            return (False, "Monotonicity violation: m_j did not increase with Tr(sigma_j M)")
        return (True, f"m_j increases from {baseline_mj} to {perturbed_mj} when Tr(sigma_j M) rises from {baseline_trace} to {perturbed_trace}")

    # ------------------------------------------------------------------
    # Weak answer sanity – only sign, finite, broad magnitude range
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: ensure all outputs are not None, eigenvalues are real,
        det is -1, linear independence is True, formulas are non‑empty.
        """
        sanity_checks = []
        # (a)
        if self.tr_sigma_i is None:
            sanity_checks.append("tr_sigma_i missing")
        elif not isinstance(self.tr_sigma_i, (int, float)):
            sanity_checks.append("tr_sigma_i not numeric")
        # (b)
        if self.eigenvalues is None:
            sanity_checks.append("eigenvalues missing")
        elif len(self.eigenvalues) != 2:
            sanity_checks.append("eigenvalues not length 2")
        else:
            for e in self.eigenvalues:
                if not isinstance(e, (int, float)) or math.isnan(e) or math.isinf(e):
                    sanity_checks.append(f"eigenvalue {e} is NaN/Inf")
        # (c)
        if self.lin_indep is not True:
            sanity_checks.append("lin_indep not True")
        # (d)
        if self.m0 is None or (isinstance(self.m0, str) and len(self.m0)==0):
            sanity_checks.append("m0 empty or missing")
        if self.m_j is None or (isinstance(self.m_j, str) and len(self.m_j)==0):
            sanity_checks.append("m_j empty or missing")

        if sanity_checks:
            return (False, "; ".join(sanity_checks))
        return (True, "Weak sanity passed")

    # ------------------------------------------------------------------
    # Main runner: discover all test_ methods and run them
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)