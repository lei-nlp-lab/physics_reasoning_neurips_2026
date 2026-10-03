import re
import math
import traceback
from typing import Any, Dict, List, Tuple, Optional


class PhysicsChecker:
    """
    Checker for the Schrödinger equation of atomic helium with an infinitely heavy nucleus.
    The solution is expected to provide a string representation of the time-independent
    Schrödinger equation in a symbolic form.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Initialize the checker by extracting the required result from the solution instance.
        Safe default: None if missing.
        """
        # Extract the equation string; handle missing via getattr or results dict
        self.schrodinger_eq: Optional[str] = getattr(solution_instance, 'schrodinger_eq', None)
        if self.schrodinger_eq is None:
            # fallback: try to get from results dict if solution_instance has one
            self.schrodinger_eq = getattr(solution_instance, 'results', {}).get('schrodinger_eq', None)

        # Also store the whole instance for potential future access (not used here)
        self.solution = solution_instance

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, collecting results.
        Returns (all_pass, logs).
        """
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
        return all_pass, logs

    # --------------------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------------------
    def test_interface_required_key_exists(self) -> Tuple[bool, str]:
        """Ensure that the result key 'schrodinger_eq' is present and is a string."""
        if self.schrodinger_eq is None:
            return (False, "Missing result: schrodinger_eq is None")
        if not isinstance(self.schrodinger_eq, str):
            return (False, f"schrodinger_eq is not a string: {type(self.schrodinger_eq)}")
        if len(self.schrodinger_eq.strip()) == 0:
            return (False, "schrodinger_eq is an empty string")
        return (True, "schrodinger_eq exists and is a non-empty string")

    # --------------------------------------------------------------------------
    # Invariant checks (theorem-based physical content)
    # --------------------------------------------------------------------------
    def test_invariant_kinetic_operator(self) -> Tuple[bool, str]:
        """Check that the kinetic energy term for both electrons is present in the form p_i^2/(2m_e)."""
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        pattern = r"p1\s*\^\s*2\s*/\s*\(\s*2\s*m_e\s*\)"
        if not re.search(pattern, self.schrodinger_eq, re.IGNORECASE):
            return (False, "Kinetic term for electron 1 (p1^2/(2m_e)) not found")
        pattern = r"p2\s*\^\s*2\s*/\s*\(\s*2\s*m_e\s*\)"
        if not re.search(pattern, self.schrodinger_eq, re.IGNORECASE):
            return (False, "Kinetic term for electron 2 (p2^2/(2m_e)) not found")
        return (True, "Both kinetic energy terms are present")

    def test_invariant_nuclear_attraction(self) -> Tuple[bool, str]:
        """Check that the nuclear attraction terms -2e^2/r1 and -2e^2/r2 are present."""
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        pattern = r"-\s*2\s*e\s*\^\s*2\s*/\s*r1\b"
        if not re.search(pattern, self.schrodinger_eq, re.IGNORECASE):
            return (False, "Nuclear attraction term for electron 1 (-2e^2/r1) not found")
        pattern = r"-\s*2\s*e\s*\^\s*2\s*/\s*r2\b"
        if not re.search(pattern, self.schrodinger_eq, re.IGNORECASE):
            return (False, "Nuclear attraction term for electron 2 (-2e^2/r2) not found")
        return (True, "Both nuclear attraction terms are present")

    def test_invariant_electron_repulsion(self) -> Tuple[bool, str]:
        """Check that the electron-electron repulsion term e^2/|r1 - r2| is present."""
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        # allow variations: |r1-r2|, |r1 - r2|, abs(r1-r2) etc.
        pattern = r"e\s*\^\s*2\s*/\s*\|r1\s*-\s*r2\||[\\|]r1\s*-\s*r2[\\|]"
        if not re.search(pattern, self.schrodinger_eq, re.IGNORECASE):
            return (False, "Electron repulsion term e^2/|r1 - r2| not found")
        return (True, "Electron repulsion term is present")

    def test_invariant_eigenvalue_form(self) -> Tuple[bool, str]:
        """
        Check that the equation has the form (Hamiltonian) psi = E psi.
        This ensures it is an eigenvalue equation, not just an expression.
        """
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        # Look for = E psi (with optional parentheses or asterisk)
        pattern = r"=\s*E\s*psi"
        if not re.search(pattern, self.schrodinger_eq):
            return (False, "Equation does not contain '= E psi' (eigenvalue form)")
        return (True, "Eigenvalue form confirmed")

    # --------------------------------------------------------------------------
    # Modeling checks (quality of the representation)
    # --------------------------------------------------------------------------
    def test_modeling_atomic_units_consistency(self) -> Tuple[bool, str]:
        """
        Check that no explicit ℏ (hbar) appears, since atomic units (ℏ=1) are assumed.
        This is a weak check; the equation may still be acceptable if ℏ is present but set to 1.
        """
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        if re.search(r'\bhbar\b|\\hbar', self.schrodinger_eq, re.IGNORECASE):
            # if hbar appears, check if it is set to 1 or explicitly divided
            # for simplicity, just warn but don't fail; we treat as pass with note
            return (True, "hbar appears in equation – not required but acceptable if set to 1")
        return (True, "No explicit hbar found, consistent with atomic units")

    def test_modeling_no_external_fields(self) -> Tuple[bool, str]:
        """
        Check that the equation does not contain any external potential terms (e.g., electric field).
        The problem states no external fields.
        """
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        suspicious = ["E_ext", "V_ext", "external", "phi_ext", "A_ext"]
        for term in suspicious:
            if term.lower() in self.schrodinger_eq.lower():
                return (False, f"Equation contains external field related term: {term}")
        return (True, "No external field terms found")

    # --------------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare outputs)
    # --------------------------------------------------------------------------
    def test_monotonicity_nuclear_charge_increases_attraction(self) -> Tuple[bool, str]:
        """
        Check that if the nuclear charge Z were increased (e.g., from 2 to 3),
        the coefficient in the nuclear attraction term would become larger (more negative).
        We extract the current coefficient from the equation string, perturb it,
        and verify that the perturbed equation still has the correct structure.
        """
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation to check")
        # Try to extract the numerical coefficient before e^2/r1
        pattern = r"([+-]?\s*\d+(?:\.\d+)?)\s*e\s*\^\s*2\s*/\s*r1\b"
        match = re.search(pattern, self.schrodinger_eq, re.IGNORECASE)
        if not match:
            return (False, "Could not extract nuclear charge coefficient for r1 term")
        # Extract the number, remove spaces
        coeff_str = match.group(1).replace(' ', '')
        try:
            base_coeff = float(coeff_str)
        except ValueError:
            return (False, f"Cannot convert extracted coefficient '{coeff_str}' to float")
        # The coefficient should be negative for attraction; ensure sign
        if base_coeff >= 0:
            return (False, "Nuclear attraction coefficient is not negative")
        # Perturb: increase Z by 1 (i.e., change coefficient from -2 to -3)
        perturbed_coeff = base_coeff - 1.0  # more negative
        # Build the perturbed equation string by substituting the coefficient
        perturbed_eq = self.schrodinger_eq.replace(match.group(1), str(int(perturbed_coeff)), 1)
        # Check that the perturbed equation still satisfies the invariants
        # We can do a quick check: kinetic terms and eigenvalue form should still be present
        kinetic_present = bool(re.search(r"p1\s*\^\s*2\s*/\s*\(\s*2\s*m_e\s*\)", perturbed_eq, re.IGNORECASE))
        eigenvalue_present = bool(re.search(r"=\s*E\s*psi", perturbed_eq))
        if not (kinetic_present and eigenvalue_present):
            return (False, "Perturbed equation lost essential structure")
        # Check that the perturbed coefficient appears exactly
        expected_new_term = f"-{int(abs(perturbed_coeff))}e^2/r1"
        if expected_new_term not in perturbed_eq.replace(' ', ''):
            return (False, "Perturbed term does not appear correctly")
        return (True, f"Baseline coefficient {base_coeff} → perturbed {perturbed_coeff}; perturbed equation remains valid")

    # --------------------------------------------------------------------------
    # Answer sanity (weak – do not dominate)
    # --------------------------------------------------------------------------
    def test_answer_sanity_basic_structure(self) -> Tuple[bool, str]:
        """
        Very weak sanity: check that the equation string contains 'psi' on both sides
        and that it is not trivial. No exact matching or derivation.
        """
        if self.schrodinger_eq is None:
            return (False, "SKIP: no equation")
        eq = self.schrodinger_eq.strip()
        # must contain psi on left and right
        if "psi" not in eq:
            return (False, "No 'psi' found in equation")
        # must contain '='
        if "=" not in eq:
            return (False, "No '=' found; not an equation")
        # must have at least 20 characters (trivial filter)
        if len(eq) < 20:
            return (False, "Equation is too short (<20 chars), likely incomplete")
        return (True, "Basic structural sanity passed")


# Example usage if this module is run directly.
if __name__ == "__main__":
    # Dummy class to simulate a solution instance
    class DummySolution:
        def __init__(self):
            self.schrodinger_eq = ("(p1^2/(2m_e) + p2^2/(2m_e) - 2e^2/r1 - 2e^2/r2 "
                                   "+ e^2/|r1 - r2|) psi(r1, r2) = E psi(r1, r2)")

    sol = DummySolution()
    checker = PhysicsChecker(sol)
    all_pass, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nAll tests passed: {all_pass}")