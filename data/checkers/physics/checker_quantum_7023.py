import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics auditor for the H2+ molecular ion problem.
    Checks interface, physical invariants, modeling quality, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {}) or {}
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract expected keys with None defaults
        self.potential_form: Optional[str] = self.results.get('potential_form', None)
        self.psi_plus: Optional[str] = self.results.get('psi_plus', None)
        self.psi_minus: Optional[str] = self.results.get('psi_minus', None)
        self.E_infinity: Optional[str] = self.results.get('E_infinity', None)

        # Additional info if present (optional)
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)
        self.phys_cons: Optional[Dict] = getattr(solution_instance, 'phys_cons', None)

    # ---------- Interface Tests ----------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """Check that all four required keys exist in results."""
        required = ['potential_form', 'psi_plus', 'psi_minus', 'E_infinity']
        missing = [k for k in required if k not in self.results]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"

    def test_interface_potential_form_type(self) -> Tuple[bool, str]:
        """potential_form must be a string."""
        if self.potential_form is None:
            return False, "potential_form is None"
        if not isinstance(self.potential_form, str):
            return False, f"potential_form is not a string, got {type(self.potential_form).__name__}"
        return True, "potential_form is a valid string"

    def test_interface_psi_plus_type(self) -> Tuple[bool, str]:
        """psi_plus must be a string."""
        if self.psi_plus is None:
            return False, "psi_plus is None"
        if not isinstance(self.psi_plus, str):
            return False, f"psi_plus is not a string, got {type(self.psi_plus).__name__}"
        return True, "psi_plus is a valid string"

    def test_interface_psi_minus_type(self) -> Tuple[bool, str]:
        """psi_minus must be a string."""
        if self.psi_minus is None:
            return False, "psi_minus is None"
        if not isinstance(self.psi_minus, str):
            return False, f"psi_minus is not a string, got {type(self.psi_minus).__name__}"
        return True, "psi_minus is a valid string"

    def test_interface_E_infinity_type(self) -> Tuple[bool, str]:
        """E_infinity must be a string."""
        if self.E_infinity is None:
            return False, "E_infinity is None"
        if not isinstance(self.E_infinity, str):
            return False, f"E_infinity is not a string, got {type(self.E_infinity).__name__}"
        return True, "E_infinity is a valid string"

    # ---------- Invariant Tests (theorem-based physical sanity) ----------
    def test_invariant_potential_symmetry(self) -> Tuple[bool, str]:
        """Check that the potential string references both Coulomb wells symmetrically."""
        if self.potential_form is None:
            return False, "SKIP: potential_form not available"
        pot = self.potential_form
        # Check for two absolute distance terms (|...| or abs(...))
        count_abs = pot.count('|')
        if count_abs < 2:
            return False, "Potential string does not show two absolute value terms (expected two Coulomb wells)"
        # Check for both plus and minus distance variants (symmetric about midpoint)
        has_plus = '+' in pot
        has_minus = '-' in pot[1:]  # ignore leading sign
        if not (has_plus and has_minus):
            return False, "Potential string lacks both plus and minus distance terms"
        return True, "Potential shows symmetric double-well structure"

    def test_invariant_psi_plus_symmetric(self) -> Tuple[bool, str]:
        """psi_plus should be a symmetric combination (phi(1s_A) + phi(1s_B))."""
        if self.psi_plus is None:
            return False, "SKIP: psi_plus not available"
        if '+' not in self.psi_plus or 'phi(1s' not in self.psi_plus:
            return False, "psi_plus string does not contain a '+' or 'phi(1s'"
        return True, "psi_plus string indicates symmetric combination (bonding)"

    def test_invariant_psi_minus_antisymmetric(self) -> Tuple[bool, str]:
        """psi_minus should be an antisymmetric combination (phi(1s_A) - phi(1s_B))."""
        if self.psi_minus is None:
            return False, "SKIP: psi_minus not available"
        minus_count = self.psi_minus.count('-')
        if minus_count < 1:
            return False, "psi_minus string does not contain a '-' sign (expected antisymmetric)"
        if 'phi(1s' not in self.psi_minus:
            return False, "psi_minus string missing 'phi(1s' orbitals"
        return True, "psi_minus string indicates antisymmetric combination (antibonding)"

    def test_invariant_large_separation_energy(self) -> Tuple[bool, str]:
        """E_infinity should refer to the hydrogen ground state energy (-13.6 eV)."""
        if self.E_infinity is None:
            return False, "SKIP: E_infinity not available"
        e_str = self.E_infinity
        contains_13_6 = '13.6' in e_str
        contains_E1s = 'E_1s' in e_str
        contains_ground = 'ground state' in e_str.lower()
        if not (contains_13_6 or contains_E1s or contains_ground):
            return False, "E_infinity string does not indicate hydrogen ground state energy (-13.6 eV)"
        return True, "E_infinity string references hydrogen 1s energy/ground state"

    # ---------- Modeling/Trace Tests ----------
    def test_modeling_normalization_factor(self) -> Tuple[bool, str]:
        """Check that psi_plus and psi_minus include the 1/√2 normalization."""
        missing_factor = []
        if self.psi_plus is not None:
            if '1/sqrt(2)' not in self.psi_plus.replace(' ', '') and '1/√2' not in self.psi_plus:
                missing_factor.append('psi_plus')
        if self.psi_minus is not None:
            if '1/sqrt(2)' not in self.psi_minus.replace(' ', '') and '1/√2' not in self.psi_minus:
                missing_factor.append('psi_minus')
        if missing_factor:
            return False, f"Missing normalization factor 1/√2 in: {', '.join(missing_factor)}"
        return True, "Both psi_plus and psi_minus contain proper normalization factor"

    def test_modeling_orbital_labels(self) -> Tuple[bool, str]:
        """Check that wave functions reference hydrogenic 1s orbitals (phi(1s_A) and phi(1s_B))."""
        issues = []
        if self.psi_plus is not None:
            if 'phi' not in self.psi_plus:
                issues.append("psi_plus missing 'phi' orbital notation")
            if '1s' not in self.psi_plus.lower():
                issues.append("psi_plus missing '1s' in orbital reference")
        if self.psi_minus is not None:
            if 'phi' not in self.psi_minus:
                issues.append("psi_minus missing 'phi' orbital notation")
            if '1s' not in self.psi_minus.lower():
                issues.append("psi_minus missing '1s' in orbital reference")
        if issues:
            return False, "; ".join(issues)
        return True, "Orbital labels correctly reference hydrogenic 1s orbitals"

    # ---------- Monotonicity Test (explicit perturbation) ----------
    def test_monotonicity_potential_at_midpoint(self) -> Tuple[bool, str]:
        """
        Monotonicity: as the proton separation R increases, the potential at the midpoint
        becomes less negative (increases). We compute the potential using the known
        formula V_mid = -4*k*e^2 / R, then perturb R by +10% and verify the increase.
        """
        # Use numerical constants from the problem
        k = 8.98755e9   # 1/(4πϵ0) in N·m²/C²
        e = 1.602e-19   # elementary charge in C
        R_baseline = 1.06e-10  # in meters

        # Compute baseline potential at midpoint (x=0)
        V_baseline = -4.0 * k * e**2 / R_baseline

        # Perturb: increase R by 10%
        R_perturbed = R_baseline * 1.10
        V_perturbed = -4.0 * k * e**2 / R_perturbed

        # Check monotonicity: V_perturbed should be greater than V_baseline (less negative)
        if V_perturbed <= V_baseline:
            return False, f"Monotonicity violated: V_baseline={V_baseline:.3e}, V_perturbed={V_perturbed:.3e} (should increase as R increases)"
        return True, f"Monotonicity holds: V increases from {V_baseline:.3e} to {V_perturbed:.3e} when R increases by 10%"

    # ---------- Answer Sanity (weak) ----------
    def test_answer_sanity_strings_nonempty(self) -> Tuple[bool, str]:
        """All result strings should be non-empty and contain relevant physics terms."""
        failures = []
        for name, val in zip(['potential_form','psi_plus','psi_minus','E_infinity'],
                              [self.potential_form, self.psi_plus, self.psi_minus, self.E_infinity]):
            if val is None:
                failures.append(f"{name} is None")
            elif not isinstance(val, str) or len(val.strip()) == 0:
                failures.append(f"{name} is empty string")
        if failures:
            return False, "; ".join(failures)
        return True, "All answer strings are non-empty"

    # ---------- Runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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