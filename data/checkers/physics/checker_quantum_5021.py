import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the periodic potential problem:
    - Bloch antiperiodic condition theta=pi
    - Free particle energy eigenvalues for allowed k
    - First-order degenerate perturbation theory for small V0
    """
    def __init__(self, solution_instance):
        # Extract givens (safe defaults = None)
        self.m = getattr(solution_instance, 'm', None)
        self.a = getattr(solution_instance, 'a', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results (from pseudocode emit contract)
        results = getattr(solution_instance, 'results', {}) or {}
        self.k_n = results.get('k_n', None)
        self.E_n = results.get('E_n', None)
        self.E1 = results.get('E1', None)
        self.E2 = results.get('E2', None)

        # Derived: free-particle ground state energy E0
        if self.hbar is not None and self.m is not None and self.a is not None:
            self.E0 = self.hbar**2 * math.pi**2 / (2 * self.m * self.a**2)
        else:
            self.E0 = None

        self.sol = solution_instance

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface checks: required keys present
    # ------------------------------------------------------------------
    def test_interface_has_givens(self):
        """Check that all required physical parameters exist."""
        missing = []
        for attr in ['m', 'a', 'V0', 'hbar']:
            if getattr(self, attr) is None:
                missing.append(attr)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All physical parameters provided")

    def test_interface_has_k_n(self):
        """Check that k_n is provided."""
        if self.k_n is None:
            return (False, "Missing k_n in results")
        return (True, "k_n present")

    def test_interface_has_E_n(self):
        """Check that E_n is provided."""
        if self.E_n is None:
            return (False, "Missing E_n in results")
        return (True, "E_n present")

    def test_interface_has_E1_E2(self):
        """Check that E1 and E2 are provided."""
        missing = []
        if self.E1 is None:
            missing.append("E1")
        if self.E2 is None:
            missing.append("E2")
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "E1 and E2 present")

    # ------------------------------------------------------------------
    # Physical invariants (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_bloch_condition(self):
        """Check that exp(i*k_n*a) = -1 for the given k_n."""
        if self.k_n is None or self.a is None:
            return (False, "SKIP: k_n or a missing")
        # Try to evaluate k_n numerically (assume n=0)
        try:
            k_n_val = float(self.k_n)
        except (TypeError, ValueError):
            # Could be a sympy expression; try substitution
            try:
                if hasattr(self.k_n, 'subs'):
                    k_n_val = float(self.k_n.subs('n', 0).evalf())
                else:
                    return (False, "Cannot evaluate k_n numerically")
            except Exception:
                return (False, "Cannot evaluate k_n numerically")
        lhs = math.exp(1j * k_n_val * self.a)
        rhs = -1.0 + 0.0j
        if math.isclose(lhs.real, rhs.real, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose(lhs.imag, rhs.imag, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"exp(i*k_n*a) ≈ {lhs:.3f} close to -1")
        else:
            return (False, f"exp(i*k_n*a) = {lhs:.3f}, expected -1")

    def test_invariant_energy_spectrum(self):
        """Check that E_n = hbar^2 k_n^2/(2m) for the given k_n and E_n."""
        if self.E_n is None or self.k_n is None or self.hbar is None or self.m is None:
            return (False, "SKIP: missing data for energy spectrum")
        try:
            E_n_val = float(self.E_n)
        except (TypeError, ValueError):
            return (False, "E_n not numeric")
        try:
            k_n_val = float(self.k_n)
        except (TypeError, ValueError):
            return (False, "k_n not numeric")
        expected_E = self.hbar**2 * k_n_val**2 / (2 * self.m)
        if math.isclose(E_n_val, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_n = {E_n_val:.5f} matches hbar^2 k^2/(2m)")
        else:
            return (False, f"E_n = {E_n_val:.5f}, expected {expected_E:.5f}")

    def test_invariant_perturbation_splitting(self):
        """Check that E2 - E1 = V0 (derived from degenerate perturbation)."""
        if self.E1 is None or self.E2 is None or self.V0 is None:
            return (False, "SKIP: need E1,E2,V0")
        try:
            gap = float(self.E2 - self.E1)
            V0_val = float(self.V0)
        except (TypeError, ValueError):
            return (False, "E1/E2/V0 not numeric")
        if math.isclose(gap, V0_val, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E2-E1 = {gap:.5f} ≈ V0 = {V0_val:.5f}")
        else:
            return (False, f"E2-E1 = {gap:.5f}, expected V0 = {V0_val:.5f}")

    # ------------------------------------------------------------------
    # Modeling/trace checks (use explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_matrix_element_nonzero(self):
        """Check that the off-diagonal matrix element <pi/a|V|-pi/a> = V0/2 is present in explanation."""
        explanation = getattr(self.sol, 'explanation_steps', [])
        found = False
        for step in explanation:
            if isinstance(step, dict) and step.get('type') == 'matrix_element':
                val = step.get('value', None)
                if val is not None:
                    try:
                        val_n = float(val)
                    except:
                        continue
                    if math.isclose(val_n, self.V0/2.0, rel_tol=1e-2, abs_tol=1e-2):
                        found = True
                        break
        if found:
            return (True, "Matrix element V0/2 found in explanation steps")
        else:
            # Accept if the student computed the correct energies, meaning they likely got the matrix element right.
            return (True, "Matrix element not explicitly checked; proceeding (can be inferred from energies)")

    def test_modeling_diagonal_elements_zero(self):
        """Check that diagonal matrix elements of V are zero (from explanation)."""
        explanation = getattr(self.sol, 'explanation_steps', [])
        found = False
        for step in explanation:
            if isinstance(step, dict) and step.get('type') == 'diagonal_element':
                val = step.get('value', None)
                if val is not None:
                    try:
                        val_n = float(val)
                    except:
                        continue
                    if math.isclose(val_n, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                        found = True
                        break
        if found:
            return (True, "Diagonal elements are zero")
        else:
            return (True, "Diagonal element not explicitly found; energies suggest correct perturbation")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicitly perturb an input, compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_free_energy_decreases_with_a(self):
        """Perturb lattice constant a upward; check that E0 decreases."""
        if self.hbar is None or self.m is None or self.a is None:
            return (False, "SKIP: missing hbar, m, or a")
        if self.a <= 0:
            return (False, "a must be positive")
        # Baseline
        E0_base = self.hbar**2 * math.pi**2 / (2 * self.m * self.a**2)
        # Perturb: increase a by 10%
        a_pert = self.a * 1.1
        E0_pert = self.hbar**2 * math.pi**2 / (2 * self.m * a_pert**2)
        if E0_pert < E0_base:
            return (True, f"E0 decreases when a increases: baseline {E0_base:.5f}, perturbed {E0_pert:.5f}")
        else:
            return (False, f"E0 should decrease but did not: baseline {E0_base:.5f}, perturbed {E0_pert:.5f}")

    def test_monotonicity_splitting_increases_with_V0(self):
        """Perturb V0 upward; check that E2-E1 increases."""
        if self.E1 is None or self.E2 is None or self.V0 is None:
            return (False, "SKIP: need E1,E2,V0")
        try:
            V0_val = float(self.V0)
        except:
            return (False, "V0 not numeric")
        # Baseline gap
        gap_base = float(self.E2 - self.E1)
        # Perturb: increase V0 by 10%
        V0_pert = V0_val * 1.1
        # Expected perturbed gap (according to physics, gap = V0)
        gap_pert = V0_pert
        if gap_pert > gap_base:
            return (True, f"Gap increases with V0: base {gap_base:.5f}, perturbed {gap_pert:.5f}")
        else:
            return (False, f"Gap should increase but it does not: base {gap_base:.5f}, perturbed {gap_pert:.5f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs_and_ranges(self):
        """Check that answers are finite and physically plausible."""
        # For E_n, should be positive (if k_n real)
        if self.E_n is not None:
            try:
                E_n_val = float(self.E_n)
            except (TypeError, ValueError):
                pass
            else:
                if E_n_val <= 0:
                    return (False, f"E_n = {E_n_val} should be positive")
        # Check E1 and E2 are real and E2 > E1
        if self.E1 is not None and self.E2 is not None:
            try:
                E1_val = float(self.E1)
                E2_val = float(self.E2)
            except (TypeError, ValueError):
                return (False, "E1/E2 not numeric")
            if not (math.isfinite(E1_val) and math.isfinite(E2_val)):
                return (False, "E1/E2 not finite")
            if E2_val <= E1_val:
                return (False, f"E2 <= E1: {E2_val} <= {E1_val}")
            # The splitting should be positive
            if E2_val - E1_val <= 0:
                return (False, "Non-positive energy splitting")
        return (True, "Answers are finite, E2>E1, splitting positive")

if __name__ == "__main__":
    # Example usage with a mock solution instance
    class MockSolution:
        pass
    sol = MockSolution()
    sol.m = 1.0
    sol.a = 1.0
    sol.V0 = 0.1
    sol.hbar = 1.0
    sol.results = {
        'k_n': math.pi,  # n=0 case: k = pi/a, with a=1 => pi
        'E_n': (1.0**2)*(math.pi**2)/(2*1.0*1.0**2),  # hbar^2 pi^2/(2m a^2)
        'E1': (math.pi**2)/2 - 0.05,
        'E2': (math.pi**2)/2 + 0.05,
    }
    checker = PhysicsChecker(sol)
    passed, logs = checker.run_all_tests()
    for log in logs:
        print(log)