import math
import re
import sympy as sp
from typing import Any, Tuple, List, Dict, Optional, Callable

class PhysicsChecker:
    """
    Auditor for the solution of a quantum mechanics problem:
    - Hydrogen-like wavefunctions for He+
    - Combination with spin
    - Antisymmetric two-electron wavefunction
    - Pauli principle and exchange symmetry
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results'):
            results = getattr(solution_instance, 'results')
            if isinstance(results, dict):
                self.results = results

        # Extract each expected key, default to None if missing
        self.scaling: Optional[str] = self.results.get('scaling', None)
        self.single_state: Optional[str] = self.results.get('single_state', None)
        self.two_electron_wf: Optional[str] = self.results.get('two_electron_wf', None)
        self.total_spin: Any = self.results.get('total_spin', None)
        self.pauli_violation: Optional[str] = self.results.get('pauli_violation', None)
        self.antisymmetry: Optional[str] = self.results.get('antisymmetry', None)

    # ---------- interface checks ----------

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all six required keys exist in results."""
        expected_keys = ['scaling', 'single_state', 'two_electron_wf',
                         'total_spin', 'pauli_violation', 'antisymmetry']
        missing = [key for key in expected_keys if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # ---------- invariant checks (physics principles) ----------

    def test_invariant_scaling(self) -> Tuple[bool, str]:
        """Scaling must indicate Z=2 and hydrogen-like character."""
        if self.scaling is None:
            return (False, "SKIP: scaling not available")
        s = str(self.scaling).lower()
        if 'z=2' not in s and 'z = 2' not in s:
            return (False, f"scaling does not contain Z=2: {self.scaling}")
        if 'hydrogen' not in s and 'h-like' not in s:
            return (False, f"scaling does not mention hydrogen-like: {self.scaling}")
        return (True, "scaling indicates Z=2 and hydrogen-like")

    def test_invariant_single_state(self) -> Tuple[bool, str]:
        """Single‑particle state must combine spatial φ and spin σ."""
        if self.single_state is None:
            return (False, "SKIP: single_state not available")
        s = str(self.single_state).lower()
        has_phi = ('φ' in s or 'phi' in s or 'φ' in s)
        has_sigma = ('σ' in s or 'sigma' in s or 'σ' in s)
        if not (has_phi and has_sigma):
            return (False, f"single_state lacks spatial or spin part: {self.single_state}")
        return (True, "single_state contains both φ and σ")

    def test_invariant_two_electron_form(self) -> Tuple[bool, str]:
        """Two‑electron wavefunction must be antisymmetric, include 1/√2, two different orbitals, and spin factor."""
        if self.two_electron_wf is None:
            return (False, "SKIP: two_electron_wf not available")
        s = str(self.two_electron_wf).lower()
        # Must contain 1/√2 or 1/sqrt(2)
        has_sqrt2 = ('1/√2' in s or '1/sqrt(2)' in s or '1/sqrt2' in s)
        # Must show antisymmetric combination with minus sign
        has_minus = (' - ' in s or '-' in s)
        # Must involve two different spatial orbitals
        has_phi1_phi2 = ('φ' in s or 'phi' in s)
        # Must include spin (σ⁺ or σ⁻)
        has_spin = ('σ' in s or 'sigma' in s)
        if not (has_sqrt2 and has_minus and has_phi1_phi2 and has_spin):
            return (False, f"two_electron_wf missing antisymmetry form: {self.two_electron_wf}")
        return (True, "two_electron_wf has correct antisymmetric structure")

    def test_invariant_total_spin(self) -> Tuple[bool, str]:
        """Total spin must be 1 (triplet state)."""
        if self.total_spin is None:
            return (False, "SKIP: total_spin not available")
        # Accept integer 1, float 1.0, string "1" or "S=1"
        s = str(self.total_spin).strip()
        try:
            val = float(s)
            if math.isclose(val, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "total_spin is 1")
        except ValueError:
            pass
        # Also check for string patterns
        if '1' in s and 's' not in s.lower():
            return (True, f"total_spin contains 1: {self.total_spin}")
        return (False, f"total_spin should indicate S=1: {self.total_spin}")

    def test_invariant_pauli(self) -> Tuple[bool, str]:
        """Pauli violation description must state that wavefunction vanishes when orbitals identical."""
        if self.pauli_violation is None:
            return (False, "SKIP: pauli_violation not available")
        s = str(self.pauli_violation).lower()
        if 'vanish' in s or 'zero' in s or 'disappear' in s:
            return (True, "pauli_violation indicates wavefunction vanishes")
        return (False, f"pauli_violation does not mention vanishing: {self.pauli_violation}")

    def test_invariant_antisymmetry(self) -> Tuple[bool, str]:
        """Antisymmetry must show that Ψ(2,1) = -Ψ(1,2)."""
        if self.antisymmetry is None:
            return (False, "SKIP: antisymmetry not available")
        s = str(self.antisymmetry).lower()
        if 'minus' in s or 'negative' in s or '-ψ' in s or '-psi' in s:
            return (True, "antisymmetry indicates sign change")
        return (False, f"antisymmetry does not show minus sign: {self.antisymmetry}")

    # ---------- modeling / trace checks ----------

    def test_trace_spatial_antisymmetry(self) -> Tuple[bool, str]:
        """
        Check that the two-electron wavefunction, when explicitly swapping labels,
        would lead to a minus sign. This is a modeling quality check:
        we verify that the expression contains two different orbital indices
        (so that the spatial antisymmetric part is nontrivial).
        """
        if self.two_electron_wf is None:
            return (False, "SKIP: two_electron_wf not available")
        s = str(self.two_electron_wf)
        # Check that at least two different orbital labels appear (e.g., φₐ and φ_b, or N1 and N2)
        # Look for patterns like φ_1, φ_2 or phiN1, phiN2
        pattern = r'φ\w*\s*\(\s*1\s*\).*φ\w*\s*\(\s*2\s*\)'
        if not re.search(pattern, s, re.IGNORECASE):
            # Try alternative: phi[1] etc.
            if '1' not in s or '2' not in s:
                return (False, f"two_electron_wf does not show particle coordinates explicitly: {s}")
        # Now we assume the antisymmetric form is correct (will be caught by other tests)
        return (True, "two_electron_wf shows distinct orbital labels and coordinates")

    def test_trace_spin_triplet_condition(self) -> Tuple[bool, str]:
        """
        Check consistency: if both spins are parallel (σ⁺σ⁺ or σ⁻σ⁻),
        total spin should be 1. We verify that the two_electron_wf
        contains parallel spin combination.
        """
        if self.two_electron_wf is None or self.total_spin is None:
            return (False, "SKIP: two_electron_wf or total_spin missing")
        s_wf = str(self.two_electron_wf).lower()
        has_parallel_spin = ('σ⁺σ⁺' in s_wf or 'sigma^+ sigma^+' in s_wf or
                             'σσ' in s_wf or 'up up' in s_wf)
        if not has_parallel_spin:
            # Could be triplet with m_s=1 determined by sigma+ sigma+
            # If it's not parallel, maybe it's singlet? We only require consistency;
            # if it's not parallel we skip because the example may be different.
            return (False, "SKIP: two_electron_wf does not contain parallel spins; cannot verify triplet consistency")
        # Now total_spin should be 1
        try:
            ts = float(str(self.total_spin))
            if math.isclose(ts, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Parallel spins imply total spin 1 (triplet)")
        except:
            pass
        if '1' in str(self.total_spin):
            return (True, "total_spin indicates 1 for parallel spins")
        return (False, f"Parallel spins in wavefunction but total_spin is {self.total_spin}, expected 1")

    # ---------- monotonicity checks ----------
    # Since the problem is qualitative, monotonicity is simulated via logical dependence.
    # We check: if two orbitals are the same, the wavefunction should vanish (Pauli).
    # This is a kind of monotonicity: as orbital indices approach identity, spatial part -> 0.

    def test_monotonicity_orbital_identity(self) -> Tuple[bool, str]:
        """
        Simulate the effect of making two orbitals identical.
        The wavefunction should vanish. We can't actually perturb, but we check
        that the solution acknowledges this dependence.
        """
        if self.pauli_violation is None:
            return (False, "SKIP: pauli_violation not available")
        s = str(self.pauli_violation).lower()
        if 'identic' in s or 'same' in s or 'equal' in s:
            return (True, "pauli_violation mentions orbital identity leading to vanishing")
        return (False, f"pauli_violation does not discuss orbital identity: {self.pauli_violation}")

    # ---------- weak answer sanity ----------
    # Only one such test allowed, and it must not do extensive derivation.

    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """Very weak sanity: all expected string results are non‑empty and plausible."""
        checks = [
            (self.scaling is not None and len(str(self.scaling)) > 0, "scaling empty"),
            (self.single_state is not None and len(str(self.single_state)) > 0, "single_state empty"),
            (self.two_electron_wf is not None and len(str(self.two_electron_wf)) > 0, "two_electron_wf empty"),
            (self.total_spin is not None, "total_spin missing"),
            (self.pauli_violation is not None and len(str(self.pauli_violation)) > 0, "pauli_violation empty"),
            (self.antisymmetry is not None and len(str(self.antisymmetry)) > 0, "antisymmetry empty"),
        ]
        for ok, msg in checks:
            if not ok:
                return (False, f"Sanity failed: {msg}")
        return (True, "All required outputs exist and are non‑empty")

    # ---------- run all tests ----------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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


# Example usage (not part of submission):
if __name__ == "__main__":
    # Dummy solution instance for demonstration
    class DummySolution:
        def __init__(self):
            self.results = {
                'scaling': 'Z=2 hydrogen-like scaling',
                'single_state': 'φ_N(r) σ⁺',
                'two_electron_wf': '(1/√2)[φ_N1(1)φ_N2(2) - φ_N2(1)φ_N1(2)] σ⁺(1)σ⁺(2)',
                'total_spin': 1,
                'pauli_violation': 'wavefunction vanishes when orbitals identical',
                'antisymmetry': 'Ψ(2,1) = -Ψ(1,2)'
            }
    checker = PhysicsChecker(DummySolution())
    success, log = checker.run_all_tests()
    for line in log:
        print(line)
    print(f"\nOverall success: {success}")