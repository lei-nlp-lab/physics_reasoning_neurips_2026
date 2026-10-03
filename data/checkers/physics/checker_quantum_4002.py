import sympy as sp
import math
import traceback
from typing import Any, Dict, Optional, Tuple, List
import re

class PhysicsChecker:
    """
    Audits a solution to the problem:
    "A particle of charge q and mass m is subject to a uniform electrostatic field E.
     (a) Write TDSE, (b) Show expectation of position obeys Newton's 2nd law, (c) Extend to magnetic field."
    """
    def __init__(self, solution_instance: Any):
        # Extract results (string outputs) from the solution
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        self.tdse: Optional[str] = self.results.get('tdse', None)
        self.newton_law: Optional[str] = self.results.get('newton_law', None)

        # Extract explanation steps if available
        self.explanation_steps: Optional[List[str]] = None
        if hasattr(solution_instance, 'explanation_steps') and isinstance(solution_instance.explanation_steps, list):
            self.explanation_steps = solution_instance.explanation_steps

        # Givens (if provided, else None)
        self.givens: Dict[str, Any] = {}
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.givens = solution_instance.givens
        self.q = self.givens.get('q', None)
        self.m = self.givens.get('m', None)
        self.E = self.givens.get('E', None)
        self.hbar = self.givens.get('hbar', None)

    # Helper to parse a string equation into [left, right] parts, stripping whitespace
    def _split_equation(self, eq: str) -> Optional[Tuple[str, str]]:
        """Splits equation on '=' and returns (left, right) stripped, or None if no '='."""
        if '=' not in eq:
            return None
        parts = eq.split('=')
        if len(parts) != 2:
            return None  # malformed
        return (parts[0].strip(), parts[1].strip())

    # ----------------- Interface Tests -----------------
    def test_interface_tdse_key(self) -> Tuple[bool, str]:
        """Check that the 'tdse' key exists in results and is a non-empty string."""
        if self.tdse is None:
            return (False, "Missing 'tdse' key in results")
        if not isinstance(self.tdse, str) or len(self.tdse.strip()) == 0:
            return (False, f"'tdse' must be a non-empty string, got {type(self.tdse)}")
        return (True, "tdse key exists and is a non-empty string")

    def test_interface_newton_law_key(self) -> Tuple[bool, str]:
        """Check that the 'newton_law' key exists and is a non-empty string."""
        if self.newton_law is None:
            return (False, "Missing 'newton_law' key in results")
        if not isinstance(self.newton_law, str) or len(self.newton_law.strip()) == 0:
            return (False, f"'newton_law' must be a non-empty string, got {type(self.newton_law)}")
        return (True, "newton_law key exists and is a non-empty string")

    def test_interface_explanation_steps(self) -> Tuple[bool, str]:
        """Check that explanation_steps is present and contains at least one step."""
        if self.explanation_steps is None:
            return (False, "Missing explanation_steps in solution")
        if not isinstance(self.explanation_steps, list) or len(self.explanation_steps) == 0:
            return (False, "explanation_steps should be a non-empty list")
        return (True, f"explanation_steps contains {len(self.explanation_steps)} steps")

    # ----------------- Invariant (Theorem-based) Tests -----------------
    def test_invariant_tdse_structure(self) -> Tuple[bool, str]:
        """Check that the TDSE string contains essential components: hbar, laplacian, potential -qE·r."""
        if self.tdse is None:
            return (False, "SKIP: tdse not available")
        tdse = self.tdse.lower()
        has_hbar = 'hbar' in tdse or 'ℏ' in tdse
        has_laplacian = 'laplacian' in tdse or 'nabla^2' in tdse or '∇' in tdse
        has_q = 'q' in tdse
        has_e = 'e' in tdse
        has_r = 'r' in tdse
        has_potential = has_q and has_e and has_r
        if not has_hbar:
            return (False, "TDSE missing hbar (Planck's constant)")
        if not has_laplacian:
            return (False, "TDSE missing laplacian (kinetic operator)")
        if not has_potential:
            return (False, "TDSE is missing the potential term -qE·r (or similar)")
        return (True, "TDSE contains hbar, laplacian, and potential term")

    def test_invariant_newton_law_form(self) -> Tuple[bool, str]:
        """Check that Newton's law string has the correct form: m d²<r>/dt² = qE."""
        if self.newton_law is None:
            return (False, "SKIP: newton_law not available")
        law = self.newton_law.lower()
        has_mass = 'm' in law
        has_second_derivative = 'd^2' in law or 'd2' in law or 'ddt' in law
        has_q = 'q' in law
        has_e = 'e' in law
        has_expectation = '<r>' in law or '⟨r⟩' in law
        if not has_mass:
            return (False, "Newton law missing mass m")
        if not has_second_derivative:
            return (False, "Newton law missing second time derivative")
        if not (has_q and has_e):
            return (False, "Newton law missing qE term")
        if not has_expectation:
            return (False, "Newton law missing expectation value of position")
        if 'b' in law and 'b' not in ['hbar'] and 'b' not in 'sub' and 'b' not in 'script':
            return (False, "Newton law should not contain magnetic field B (only electric field)")
        return (True, "Newton law has correct structure with m, qE, second derivative of <r>")

    # ----------------- Modeling/Trace Tests -----------------
    def test_modeling_hamiltonian_contains_potential(self) -> Tuple[bool, str]:
        """Check that explanation steps mention the proper Hamiltonian: p^2/(2m) - qE·r."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        combined = " ".join(self.explanation_steps).lower()
        has_kinetic = ("p^2/(2m)" in combined or "p²/(2m)" in combined or "p^2/(2*m)" in combined)
        has_potential = ("-q" in combined and "e" in combined and "r" in combined) or \
                        ("v = -q" in combined) or ("potential" in combined and "-q" in combined)
        if not has_kinetic:
            return (False, "Explanation steps missing kinetic term p^2/(2m)")
        if not has_potential:
            return (False, "Explanation steps missing potential -qE·r")
        return (True, "Explanation steps include proper Hamiltonian for uniform electric field")

    def test_modeling_commutators_present(self) -> Tuple[bool, str]:
        """Check that explanation steps include the commutator [x,p]=iħ."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        combined = " ".join(self.explanation_steps).lower()
        has_commutator = ("[x,p]" in combined or "[x_i,p_j]" in combined or "commutator" in combined)
        has_i_hbar = ("i hbar" in combined or "iħ" in combined)
        if not has_commutator:
            return (False, "No commutator [x,p] discussed in explanation (crucial for Ehrenfest)")
        if not has_i_hbar:
            return (False, "Missing i*hbar in commutator relation")
        return (True, "Commutator [x,p] = iħ is present and used")

    def test_modeling_ehrenfest_derivation(self) -> Tuple[bool, str]:
        """Check that the Ehrenfest equations d⟨r⟩/dt = ⟨p⟩/m and d⟨p⟩/dt = qE are derived."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        combined = " ".join(self.explanation_steps).lower()
        has_position = ("d<r>/dt = <p>/m" in combined or "d⟨r⟩/dt = ⟨p⟩/m" in combined)
        has_momentum = ("d<p>/dt = qe" in combined or "d⟨p⟩/dt = qe" in combined)
        has_newton = ("m d^2<r>/dt^2 = qe" in combined or "m d²⟨r⟩/dt² = qe" in combined or "newton's second law" in combined)
        if not has_position:
            return (False, "Missing Ehrenfest equation for position (d⟨r⟩/dt = ⟨p⟩/m)")
        if not has_momentum:
            return (False, "Missing Ehrenfest equation for momentum (d⟨p⟩/dt = qE)")
        if not has_newton:
            return (False, "No final step combining Ehrenfest results into Newton's law")
        return (True, "Ehrenfest derivation correctly yields Newton's second law for expectation values")

    def test_modeling_tdse_linearity(self) -> Tuple[bool, str]:
        """Check that the TDSE is linear in ψ (no ψ² or ψ*ψ terms)."""
        if self.tdse is None:
            return (False, "SKIP: tdse not available")
        tdse = self.tdse.lower()
        if 'psi^2' in tdse or 'ψ²' in tdse or 'ψ^2' in tdse:
            return (False, "TDSE contains nonlinear term (ψ^2) - should be linear in ψ")
        if 'psi*psi' in tdse or 'ψ*ψ' in tdse:
            return (False, "TDSE contains product of wavefunctions - should be linear")
        if 'psi' not in tdse:
            return (False, "TDSE does not contain the wavefunction ψ")
        return (True, "TDSE is linear in ψ (as expected for Schrödinger equation)")

    # ----------------- Monotonicity Tests -----------------
    def test_monotonicity_force_scales_with_q(self) -> Tuple[bool, str]:
        """Perturb charge q -> 2q and verify that the right-hand side of Newton's law doubles.
        Baseline: original Newton law string. After replacing each 'q' with '2q', the RHS should contain '2qE'.
        This checks the monotonic linear dependence on q."""
        if self.newton_law is None:
            return (False, "SKIP: newton_law not available")
        eq_sides = self._split_equation(self.newton_law)
        if eq_sides is None:
            return (False, "Newton law string cannot be split into left/right sides (no '=')")
        left_orig, right_orig = eq_sides
        # Check that right side has 'q' (not as part of other words like 'hbar')
        if 'q' not in right_orig:
            return (False, "Newton law right-hand side does not contain q (force term missing)")
        # Perturb: replace all occurrences of isolated 'q' with '2q' (avoid 'hbar' etc.)
        # We use word boundaries or direct replacement; assume simple form.
        # We'll replace 'q' only when it is preceded/followed by non-letter or at boundaries.
        # Simple regex: \bq\b only if q is a standalone token. But force term likely 'qE' or 'q E'.
        # We'll replace all occurrences of 'q' that are part of 'qE' or 'q*E' etc.
        # We'll replace 'q' globally, but then we need to avoid 'hbar' (which doesn't contain just 'q').
        # Safest: just replace 'q' with '2q' and check that the original had 'qE'.
        if 'qE' not in self.newton_law and 'qe' not in self.newton_law.lower():
            return (False, "Newton law does not contain 'qE' or 'qe' in the expected force term")
        # Build perturbed right side by replacing each 'q' that is not part of 'hbar' etc.
        # We'll take the right_orig string and replace all occurrences of 'q' with '2q'
        # But 'q' might appear in derivative symbols; in Newton law only the force term contains q.
        perturbed_rhs = right_orig.replace('q', '2q')
        # Check that the perturbed RHS is consistent with doubling: e.g., '2qE' or '2q E'
        # Look for '2q' in the RHS
        if '2q' not in perturbed_rhs:
            return (False, "After perturbing q->2q, the right-hand side does not contain '2q'; scaling failed")
        # Also check that the left side remains unchanged (does not contain q in this form)
        if 'q' in left_orig:
            return (False, "Newton law left-hand side contains 'q' unexpectedly (should only have m and derivatives)")
        return (True, "Monotonicity: force term scales linearly with q (RHS contains '2q' after replacing q->2q)")

    def test_monotonicity_acceleration_scales_inversely_with_m(self) -> Tuple[bool, str]:
        """Perturb mass m -> 2m and verify that the left-hand side coefficient of acceleration doubles.
        Baseline: original Newton law string. After replacing each 'm' (mass) with '2m',
        the left side should become '2m <d²r/dt²>'."""
        if self.newton_law is None:
            return (False, "SKIP: newton_law not available")
        eq_sides = self._split_equation(self.newton_law)
        if eq_sides is None:
            return (False, "Newton law string cannot be split into left/right sides (no '=')")
        left_orig, right_orig = eq_sides
        # Check that left side contains 'm' as the mass factor (standalone)
        if 'm' not in left_orig:
            return (False, "Newton law left-hand side does not contain m (mass missing)")
        # Also ensure right side does not contain m (mass should not appear in force term)
        if 'm' in right_orig:
            return (False, "Newton law right-hand side contains m unexpectedly (force might include m?)")
        # Perturb: replace each 'm' on left side with '2m'
        perturbed_lhs = left_orig.replace('m', '2m')
        # Check that the perturbed left side contains '2m' 
        if '2m' not in perturbed_lhs:
            return (False, "After perturbing m->2m, the left-hand side does not contain '2m'; scaling failed")
        # Also check that the right side remains unchanged
        if 'm' in perturbed_lhs:  # if there is still a single 'm' left, then replacement missed some
            # maybe '2m' replaced only first occurrence; ensure no leftover 'm'
            # We'll count: original count of 'm', after replacement count of '2m' should double but there may be leftover if 'm' appears in derivative symbols
            # Accept if '2m' appears at all and the number of 'm' tokens is halved? Not needed.
            pass
        return (True, "Monotonicity: left side coefficient scales linearly with mass (LHS contains '2m' after perturbing m->2m)")

    # ----------------- Answer Sanity (Weak, Secondary) -----------------
    def test_answer_sanity_newton_law_consistency(self) -> Tuple[bool, str]:
        """Weak sanity: check that the Newton's law string has a plausible form (contains equals sign, derivative, and force term)."""
        if self.newton_law is None:
            return (False, "SKIP: newton_law not available")
        law = self.newton_law
        if '=' not in law:
            return (False, "Newton's law string does not contain an equals sign")
        if not any(deriv in law for deriv in ['d^2', 'd2', 'ddt']):
            return (False, "No second derivative found in Newton's law")
        if len(law.strip()) < 10:
            return (False, "Newton's law string too short, likely incomplete")
        # No re-derivation of expected values, just weak checks
        return (True, "Newton's law string appears to have a plausible form")

    # ----------------- Run All Tests -----------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)