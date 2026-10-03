import math
import sympy as sp
from typing import Any, Dict, List, Optional, Tuple, Callable

class PhysicsChecker:
    """Auditor for parallel-plate capacitor problem.
    Checks invariance, modeling quality, monotonicity, and weak answer sanity.
    """

    # Known physical constant (S.I. units)
    EPS0 = 8.854e-12  # F/m

    def __init__(self, solution_instance: Any):
        # Extract inputs and results safely using None defaults
        self.A = getattr(solution_instance, 'A', None)
        self.d_i = getattr(solution_instance, 'd_i', None)
        self.d_f = getattr(solution_instance, 'd_f', None)
        self.V_i = getattr(solution_instance, 'V_i', None)
        self.eps0_provided = getattr(solution_instance, 'eps0', None)

        # Results dict (expected keys 'W' and 'V_f')
        results = getattr(solution_instance, 'results', {})
        self.W = results.get('W', None)
        self.V_f = results.get('V_f', None)

        # Derived quantities we compute ourselves for checks
        self.C_i = None
        self.C_f = None
        self.Q = None
        if self.A is not None and self.d_i is not None:
            eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0
            self.C_i = eps0 * self.A / self.d_i
            if self.V_i is not None:
                self.Q = self.C_i * self.V_i
        if self.A is not None and self.d_f is not None:
            eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0
            self.C_f = eps0 * self.A / self.d_f

    def _isclose(self, a: float, b: float) -> bool:
        """Tolerance 1e-2 for physics invariants."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect results."""
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
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that both result keys are present."""
        if self.W is None:
            return False, "Missing result key 'W'"
        if self.V_f is None:
            return False, "Missing result key 'V_f'"
        return True, "Both 'W' and 'V_f' present"

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that essential givens are provided (allows None skipping later)."""
        missing = []
        if self.A is None:
            missing.append("A")
        if self.d_i is None:
            missing.append("d_i")
        if self.V_i is None:
            missing.append("V_i")
        if self.W is None or self.V_f is None:
            missing.append("results['W'] or 'V_f'")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All essential givens present"

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based physical laws)
    # ------------------------------------------------------------------
    def test_invariant_charge_conservation(self) -> Tuple[bool, str]:
        """Q = C_i * V_i must equal Q_f = C_f * V_f."""
        if None in (self.C_i, self.C_f, self.V_i, self.V_f, self.Q):
            return False, "SKIP: missing data for charge conservation"
        Q_initial = self.C_i * self.V_i
        Q_final = self.C_f * self.V_f
        if not self._isclose(Q_initial, Q_final):
            return False, f"Charge not conserved: Q_i={Q_initial}, Q_f={Q_final}"
        return True, f"Charge conserved: Q={Q_initial}"

    def test_invariant_capacitance_relation_initial(self) -> Tuple[bool, str]:
        """C_i = eps0 * A / d_i."""
        if None in (self.C_i, self.A, self.d_i):
            return False, "SKIP: missing data for initial capacitance check"
        eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0
        expected = eps0 * self.A / self.d_i
        if not self._isclose(self.C_i, expected):
            return False, f"Initial capacitance mismatch: C_i={self.C_i}, expected={expected}"
        return True, f"Initial capacitance correct: C_i={self.C_i}"

    def test_invariant_capacitance_relation_final(self) -> Tuple[bool, str]:
        """C_f = eps0 * A / d_f."""
        if None in (self.C_f, self.A, self.d_f):
            return False, "SKIP: missing data for final capacitance check"
        eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0
        expected = eps0 * self.A / self.d_f
        if not self._isclose(self.C_f, expected):
            return False, f"Final capacitance mismatch: C_f={self.C_f}, expected={expected}"
        return True, f"Final capacitance correct: C_f={self.C_f}"

    def test_invariant_capacitance_doubling(self) -> Tuple[bool, str]:
        """Given d_f = 2*d_i, then C_f = C_i/2."""
        if None in (self.C_i, self.C_f, self.d_i, self.d_f):
            return False, "SKIP: missing data for capacitance doubling check"
        if not self._isclose(self.d_f, 2 * self.d_i):
            return False, "d_f is not twice d_i"
        expected = self.C_i / 2
        if not self._isclose(self.C_f, expected):
            return False, f"C_f not equal to C_i/2: C_f={self.C_f}, expected={expected}"
        return True, "C_f = C_i/2 (d_f=2*d_i)"

    def test_invariant_work_energy_relation(self) -> Tuple[bool, str]:
        """W = (Q^2/2)*(1/C_f - 1/C_i)."""
        if None in (self.W, self.Q, self.C_i, self.C_f):
            return False, "SKIP: missing data for work-energy relation"
        expected = (self.Q**2 / 2) * (1 / self.C_f - 1 / self.C_i)
        if not self._isclose(self.W, expected):
            return False, f"Work energy mismatch: W={self.W}, expected={expected}"
        return True, f"Work equals change in stored energy: W={self.W}"

    def test_invariant_final_voltage_via_q(self) -> Tuple[bool, str]:
        """V_f = Q / C_f."""
        if None in (self.V_f, self.Q, self.C_f):
            return False, "SKIP: missing data for V_f = Q/C_f"
        expected = self.Q / self.C_f
        if not self._isclose(self.V_f, expected):
            return False, f"V_f not equal to Q/C_f: V_f={self.V_f}, expected={expected}"
        return True, f"V_f = Q/C_f = {self.V_f}"

    def test_invariant_final_voltage_doubling(self) -> Tuple[bool, str]:
        """For d_f=2*d_i, V_f should equal 2*V_i."""
        if None in (self.V_f, self.V_i, self.d_i, self.d_f):
            return False, "SKIP: missing data for V_f doubling check"
        if not self._isclose(self.d_f, 2 * self.d_i):
            return False, "d_f is not twice d_i, cannot apply V_f = 2*V_i rule"
        expected = 2 * self.V_i
        if not self._isclose(self.V_f, expected):
            return False, f"V_f not twice V_i: V_f={self.V_f}, expected={expected}"
        return True, f"V_f = 2*V_i = {self.V_f}"

    # ------------------------------------------------------------------
    # Modeling quality checks
    # ------------------------------------------------------------------
    def test_modeling_work_positive(self) -> Tuple[bool, str]:
        if self.W is None:
            return False, "SKIP: W not available"
        if self.W <= 0:
            return False, f"W should be positive (plates attract), got {self.W}"
        return True, "Work is positive"

    def test_modeling_voltage_increases(self) -> Tuple[bool, str]:
        """V_f > V_i because C decreases."""
        if None in (self.V_f, self.V_i):
            return False, "SKIP: voltages not available"
        if self.V_f <= self.V_i:
            return False, f"V_f should be > V_i, got V_f={self.V_f}, V_i={self.V_i}"
        return True, "Final voltage is higher than initial"

    def test_modeling_finite_work(self) -> Tuple[bool, str]:
        if self.W is None:
            return False, "SKIP: W not available"
        if not math.isfinite(self.W):
            return False, f"Work is not finite: {self.W}"
        return True, "Work is finite"

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_work_increases_with_spacing(self) -> Tuple[bool, str]:
        """If we increase d_f by 10%, the computed work should increase."""
        if None in (self.W, self.Q, self.C_i, self.d_f, self.d_i, self.A):
            return False, "SKIP: insufficient data for work monotonicity"
        eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0

        # Baseline: current work (already computed)
        baseline_W = self.W

        # Perturb input: increase d_f by 10%
        d_perturbed = 1.1 * self.d_f
        C_perturbed = eps0 * self.A / d_perturbed
        if C_perturbed is None:
            return False, "Cannot compute capacitance for perturbed spacing"
        perturbed_W = (self.Q**2 / 2) * (1 / C_perturbed - 1 / self.C_i)

        # Expect: perturbed_W > baseline_W
        if perturbed_W <= baseline_W:
            return False, f"Work should increase when spacing increases: baseline={baseline_W}, perturbed={perturbed_W}"
        return True, f"Work monotonic: baseline W={baseline_W}, perturbed W+10%={perturbed_W}"

    def test_monotonicity_voltage_increases_with_spacing(self) -> Tuple[bool, str]:
        """If we increase d_f by 10%, the final voltage should increase."""
        if None in (self.V_f, self.Q, self.d_f, self.d_i, self.A):
            return False, "SKIP: insufficient data for voltage monotonicity"
        eps0 = self.eps0_provided if self.eps0_provided is not None else PhysicsChecker.EPS0

        # Baseline: current final voltage
        baseline_Vf = self.V_f

        # Perturb input: increase d_f by 10%
        d_perturbed = 1.1 * self.d_f
        C_perturbed = eps0 * self.A / d_perturbed
        if C_perturbed is None:
            return False, "Cannot compute capacitance for perturbed spacing"
        perturbed_Vf = self.Q / C_perturbed

        # Expect: perturbed_Vf > baseline_Vf
        if perturbed_Vf <= baseline_Vf:
            return False, f"Voltage should increase when spacing increases: baseline={baseline_Vf}, perturbed={perturbed_Vf}"
        return True, f"Voltage monotonic: baseline V_f={baseline_Vf}, perturbed V_f+10%={perturbed_Vf}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, broad check)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Broad checks: positive, finite, plausible magnitudes."""
        issues = []
        if self.W is None or not math.isfinite(self.W):
            issues.append("W not finite")
        elif self.W <= 0:
            issues.append("W not positive")
        elif self.W > 1e-2:   # too large for this problem
            issues.append(f"W suspiciously large: {self.W}")
        elif self.W < 1e-10:  # too small
            issues.append(f"W suspiciously small: {self.W}")

        if self.V_f is None or not math.isfinite(self.V_f):
            issues.append("V_f not finite")
        elif self.V_f <= 0:
            issues.append("V_f not positive")
        elif self.V_f > 1e6:
            issues.append(f"V_f unreasonably large: {self.V_f}")

        if issues:
            return False, "Sanity issues: " + "; ".join(issues)
        return True, "Broad sanity checks passed"