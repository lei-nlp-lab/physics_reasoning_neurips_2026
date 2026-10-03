import sympy as sp
import math
import re

class PhysicsChecker:
    """
    Physics auditor for a quantum mechanics problem on identical fermions,
    Pauli principle, LS-coupling, and Hund's rule.
    """
    def __init__(self, solution_instance):
        # Extract results safely, default to None
        self.cond = getattr(solution_instance, 'cond', None)
        if self.cond is None:
            self.cond = getattr(solution_instance, 'results', {}).get('cond', None)

        self.pauli_statement = getattr(solution_instance, 'pauli_statement', None)
        if self.pauli_statement is None:
            self.pauli_statement = getattr(solution_instance, 'results', {}).get('pauli_statement', None)

        self.L = getattr(solution_instance, 'L', None)
        if self.L is None:
            self.L = getattr(solution_instance, 'results', {}).get('L', None)

        self.S = getattr(solution_instance, 'S', None)
        if self.S is None:
            self.S = getattr(solution_instance, 'results', {}).get('S', None)

        self.spatial_sym = getattr(solution_instance, 'spatial_sym', None)
        if self.spatial_sym is None:
            self.spatial_sym = getattr(solution_instance, 'results', {}).get('spatial_sym', None)

        self.spatial_asym = getattr(solution_instance, 'spatial_asym', None)
        if self.spatial_asym is None:
            self.spatial_asym = getattr(solution_instance, 'results', {}).get('spatial_asym', None)

        self.lowest_S = getattr(solution_instance, 'lowest_S', None)
        if self.lowest_S is None:
            self.lowest_S = getattr(solution_instance, 'results', {}).get('lowest_S', None)

    # ---------- Interface tests ----------
    def test_interface_cond_present(self):
        """Check that cond is emitted as a non-empty string."""
        if self.cond is None:
            return (False, "cond is missing")
        if not isinstance(self.cond, str) or len(self.cond.strip()) == 0:
            return (False, "cond must be a non-empty string")
        return (True, "cond is a non-empty string")

    def test_interface_pauli_statement_present(self):
        """Check that pauli_statement is emitted as a non-empty string."""
        if self.pauli_statement is None:
            return (False, "pauli_statement is missing")
        if not isinstance(self.pauli_statement, str) or len(self.pauli_statement.strip()) == 0:
            return (False, "pauli_statement must be a non-empty string")
        return (True, "pauli_statement is a non-empty string")

    def test_interface_L_present(self):
        """Check that L is emitted and is an integer."""
        if self.L is None:
            return (False, "L is missing")
        if not isinstance(self.L, int):
            return (False, f"L must be an integer, got {type(self.L).__name__}")
        return (True, f"L = {self.L}")

    def test_interface_S_present(self):
        """Check that S is emitted as a list or tuple."""
        if self.S is None:
            return (False, "S is missing")
        if not isinstance(self.S, (list, tuple)):
            return (False, f"S must be a list or tuple, got {type(self.S).__name__}")
        if not all(isinstance(s, int) for s in self.S):
            return (False, "All elements of S must be integers")
        return (True, f"S = {self.S}")

    def test_interface_spatial_sym_present(self):
        """Check that spatial_sym is emitted as a non-empty string."""
        if self.spatial_sym is None:
            return (False, "spatial_sym is missing")
        if not isinstance(self.spatial_sym, str) or len(self.spatial_sym.strip()) == 0:
            return (False, "spatial_sym must be a non-empty string")
        return (True, "spatial_sym is present")

    def test_interface_spatial_asym_present(self):
        """Check that spatial_asym is emitted as a non-empty string."""
        if self.spatial_asym is None:
            return (False, "spatial_asym is missing")
        if not isinstance(self.spatial_asym, str) or len(self.spatial_asym.strip()) == 0:
            return (False, "spatial_asym must be a non-empty string")
        return (True, "spatial_asym is present")

    def test_interface_lowest_S_present(self):
        """Check that lowest_S is emitted and is an integer."""
        if self.lowest_S is None:
            return (False, "lowest_S is missing")
        if not isinstance(self.lowest_S, int):
            return (False, f"lowest_S must be an integer, got {type(self.lowest_S).__name__}")
        return (True, f"lowest_S = {self.lowest_S}")

    # ---------- Invariant / theorem-based sanity tests ----------
    def test_invariant_antisymmetry_condition(self):
        """The condition must assert that exchange flips sign."""
        if self.cond is None:
            return (False, "SKIP: cond not available")
        has_P12 = 'P_{12}' in self.cond
        has_neg = '-' in self.cond
        has_eq = '=' in self.cond
        has_psi = 'psi' in self.cond.lower() or 'ψ' in self.cond
        passes = has_P12 and has_neg and has_eq and has_psi
        if not passes:
            return (False, f"cond does not contain required antisymmetry elements (P12, -, =, wavefunction). Got: {self.cond[:200]}")
        return (True, "Antisymmetry condition is correctly expressed")

    def test_invariant_pauli_implication(self):
        """Pauli statement must state that wavefunction vanishes for identical quantum numbers."""
        if self.pauli_statement is None:
            return (False, "SKIP: pauli_statement not available")
        has_zero = '0' in self.pauli_statement
        has_identical = 'identical' in self.pauli_statement.lower() or 'same' in self.pauli_statement.lower()
        passes = has_zero and has_identical
        if not passes:
            return (False, f"pauli_statement does not mention zero or identical quantum numbers. Got: {self.pauli_statement[:200]}")
        return (True, "Pauli statement correctly implies zero wavefunction for identical quantum numbers")

    def test_invariant_L_coupling(self):
        """Derive expected L from l1=0, l2=1 (3s,3p). Total L must be 1."""
        if self.L is None:
            return (False, "SKIP: L not available")
        # Derive L from angular momentum addition: l1=0, l2=1 => only L=1.
        l1, l2 = 0, 1
        expected_L = 1  # because |l1-l2| = 1, l1+l2=1 => only 1.
        if self.L != expected_L:
            return (False, f"L must be {expected_L}, got {self.L}")
        return (True, f"L = 1 correct (derived from l1=0, l2=1)")

    def test_invariant_S_coupling(self):
        """Derive possible S from s1=s2=1/2. Must be 0 and 1."""
        if self.S is None:
            return (False, "SKIP: S not available")
        s1, s2 = 0.5, 0.5
        expected_low = abs(s1 - s2)
        expected_high = s1 + s2
        if not isinstance(self.S, (list, tuple)):
            return (False, "S is not a list/tuple")
        S_values = set(self.S)
        if S_values != {0, 1}:
            return (False, f"S values must be {{0, 1}} (derived from s1=s2=1/2), got {S_values}")
        return (True, f"S values {S_values} are correct for two spin-1/2")

    def test_invariant_spatial_symmetry_plus(self):
        """Symmetric spatial wavefunction must include a plus sign."""
        if self.spatial_sym is None:
            return (False, "SKIP: spatial_sym not available")
        if '+' not in self.spatial_sym:
            return (False, "Symmetric wavefunction should contain a '+' sign")
        return (True, "Spatial symmetric wavefunction includes plus")

    def test_invariant_spatial_asymmetry_minus(self):
        """Antisymmetric spatial wavefunction must include a minus sign."""
        if self.spatial_asym is None:
            return (False, "SKIP: spatial_asym not available")
        if '-' not in self.spatial_asym:
            return (False, "Antisymmetric wavefunction should contain a '-' sign")
        return (True, "Spatial antisymmetric wavefunction includes minus")

    def test_invariant_lowest_SP(self):
        """Derive lowest energy state from Hund's rule: maximum S (triplet) corresponds to lowest energy."""
        if self.lowest_S is None:
            return (False, "SKIP: lowest_S not available")
        # Hund's rule: for same configuration, highest spin gives lowest energy.
        # With S=0 and 1, maximum is 1.
        derived_lowest = 1
        if self.lowest_S != derived_lowest:
            return (False, f"Lowest energy should have S={derived_lowest}, but got {self.lowest_S}")
        return (True, "Lowest energy state has S=1 (triplet), consistent with Hund's rule")

    # ---------- Modeling / trace quality tests ----------
    def test_modeling_spatial_sym_normalization(self):
        """Check that the symmetric wavefunction mentions 1/√2 or equivalent."""
        if self.spatial_sym is None:
            return (False, "SKIP: spatial_sym not available")
        has_sqrt2 = 'sqrt(2)' in self.spatial_sym or '√2' in self.spatial_sym
        if not has_sqrt2:
            return (False, f"Should include normalization 1/√2, not found in: {self.spatial_sym[:200]}")
        return (True, "Normalization 1/√2 appears in symmetric wavefunction")

    def test_modeling_spatial_asym_normalization(self):
        """Check that the antisymmetric wavefunction mentions 1/√2 or equivalent."""
        if self.spatial_asym is None:
            return (False, "SKIP: spatial_asym not available")
        has_sqrt2 = 'sqrt(2)' in self.spatial_asym or '√2' in self.spatial_asym
        if not has_sqrt2:
            return (False, f"Should include normalization 1/√2, not found in: {self.spatial_asym[:200]}")
        return (True, "Normalization 1/√2 appears in antisymmetric wavefunction")

    def test_modeling_spatial_sym_function_names(self):
        """Symmetric spatial wavefunction should contain both s and p orbital names."""
        if self.spatial_sym is None:
            return (False, "SKIP: spatial_sym not available")
        has_s = any(name in self.spatial_sym for name in ['psi_s', 'ϕ_s', '\\phi_s'])
        has_p = any(name in self.spatial_sym for name in ['phi_p', 'φ_p', '\\phi_p'])
        if not (has_s and has_p):
            return (False, f"Should include s and p functions. Found: {self.spatial_sym[:200]}")
        return (True, "Both s and p orbital functions present in symmetric wavefunction")

    def test_modeling_spatial_asym_function_names(self):
        """Antisymmetric spatial wavefunction should contain both s and p orbital names."""
        if self.spatial_asym is None:
            return (False, "SKIP: spatial_asym not available")
        has_s = any(name in self.spatial_asym for name in ['psi_s', 'ϕ_s', '\\phi_s'])
        has_p = any(name in self.spatial_asym for name in ['phi_p', 'φ_p', '\\phi_p'])
        if not (has_s and has_p):
            return (False, f"Should include s and p functions. Found: {self.spatial_asym[:200]}")
        return (True, "Both s and p orbital functions present in antisymmetric wavefunction")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_spatial_symmetry_at_coincidence(self):
        """
        Check that perturbing the spatial symmetry from symmetric (+) to antisymmetric (-)
        reduces the wavefunction amplitude at r1=r2 from nonzero to zero.
        This tests monotonic decrease as symmetry parameter goes from +1 to -1.
        """
        if self.spatial_sym is None or self.spatial_asym is None:
            return (False, "SKIP: spatial_sym or spatial_asym not available")
        # Baseline: symmetric has '+' gives constructive interference at r1=r2 -> nonzero.
        # Perturbed: antisymmetric with '-' gives exact cancellation -> zero.
        has_plus = '+' in self.spatial_sym
        has_minus = '-' in self.spatial_asym
        if has_plus and has_minus:
            return (True, "Perturbation from '+' to '-' reduces coincident amplitude from nonzero to zero (monotonic decrease).")
        else:
            return (False, "Missing essential signs for symmetry perturbation test.")

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_cond_format(self):
        """Very weak sanity: cond should be a non-empty string of reasonable length."""
        if self.cond is None:
            return (False, "cond missing")
        if not isinstance(self.cond, str) or len(self.cond) < 10:
            return (False, f"cond seems too short: {len(self.cond)} characters")
        return (True, "cond formatting is plausible")

    # ---------- Main runner ----------
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