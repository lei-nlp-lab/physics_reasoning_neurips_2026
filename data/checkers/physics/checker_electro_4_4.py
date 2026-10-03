import math
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits solutions to the plane wave Maxwell's equations problem.

    Expected solution_instance attributes:
        - results: dict containing 'K_dot_A0' and 'dispersion'
        - A0: vector (list, tuple, or SymPy Matrix) with components A0x, A0y, A0z
        - K: vector (list, tuple) of wavevector components
        - omega: scalar frequency
        - c: speed of light (default symbolic 'c')
        - explanation_steps: list of step descriptions (optional)
        - pseudocode: dict or string describing pseudocode (optional)
        - givens: dict of input parameters (optional)
    """
    def __init__(self, solution_instance: Any):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Extract potentials and parameters (use None defaults)
        self.A0 = getattr(solution_instance, 'A0', None)
        self.K = getattr(solution_instance, 'K', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.c = getattr(solution_instance, 'c', sp.symbols('c'))  # fallback symbolic

        # Extract secondary info
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        self.givens = getattr(solution_instance, 'givens', None)

    # ----------------------------------------------------------------------
    # Interface checks: required keys and types
    # ----------------------------------------------------------------------
    def test_interface_has_K_dot_A0_key(self) -> Tuple[bool, str]:
        if 'K_dot_A0' not in self.results:
            return (False, "Missing key 'K_dot_A0' in results")
        return (True, "'K_dot_A0' present in results")

    def test_interface_has_dispersion_key(self) -> Tuple[bool, str]:
        if 'dispersion' not in self.results:
            return (False, "Missing key 'dispersion' in results")
        return (True, "'dispersion' present in results")

    def test_interface_A0_not_none(self) -> Tuple[bool, str]:
        if self.A0 is None:
            return (False, "A0 is None; cannot verify constraints")
        return (True, "A0 is provided")

    def test_interface_K_not_none(self) -> Tuple[bool, str]:
        if self.K is None:
            return (False, "K is None; cannot verify wavevector")
        return (True, "K is provided")

    def test_interface_omega_not_none(self) -> Tuple[bool, str]:
        if self.omega is None:
            return (False, "omega is None; cannot verify frequency")
        return (True, "omega is provided")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem-based physical correctness
    # ----------------------------------------------------------------------
    def test_invariant_transversality_dot_product_zero(self) -> Tuple[bool, str]:
        """Check that K·A0 = 0 (transversality constraint from ∇·E=0)."""
        if self.K is None or self.A0 is None:
            return (False, "SKIP: K or A0 missing")

        # Try numeric check first
        try:
            K = [float(x) for x in self.K]
            A0 = [float(x) for x in self.A0]
            dot = sum(K[i] * A0[i] for i in range(3))
            if math.isclose(dot, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return (True, f"K·A0 ≈ 0 (computed dot = {dot})")
            else:
                return (False, f"K·A0 = {dot}, expected ≈ 0")
        except (TypeError, ValueError, AttributeError):
            pass

        # Fallback: symbolic check if sympy vectors available
        try:
            K_vec = sp.Matrix(self.K)
            A0_vec = sp.Matrix(self.A0)
            dot_expr = K_vec.dot(A0_vec)
            simplified = sp.simplify(dot_expr)
            if simplified == 0:
                return (True, "K·A0 symbolically simplifies to 0")
            else:
                return (False, f"K·A0 = {simplified}, expected 0")
        except Exception as e:
            return (False, f"Cannot evaluate K·A0: {e}")

    def test_invariant_dispersion_relation(self) -> Tuple[bool, str]:
        """Check that ω² = c² |K|²."""
        if self.K is None or self.omega is None:
            return (False, "SKIP: K or omega missing")

        # Numeric check
        try:
            K_norm = math.sqrt(sum(float(x) ** 2 for x in self.K))
            omega_val = float(self.omega)
            # Try to get c numerically, else assume symbolic c
            try:
                c_val = float(self.c)
            except (TypeError, ValueError):
                c_val = 1.0  # placeholder, will fail if mismatch
            if math.isclose(omega_val ** 2, c_val ** 2 * K_norm ** 2, rel_tol=1e-5, abs_tol=1e-5):
                return (True, f"ω² = c²K² holds (ω={omega_val}, K={K_norm})")
            else:
                return (False, f"ω² = {omega_val ** 2}, c²K² = {c_val ** 2 * K_norm ** 2}")
        except (TypeError, ValueError):
            pass

        # Symbolic check
        try:
            K_vec = sp.Matrix(self.K)
            K_sq = K_vec.dot(K_vec)
            omega_sq = self.omega ** 2
            c_sq = self.c ** 2
            eq = sp.simplify(omega_sq - c_sq * K_sq)
            if eq == 0:
                return (True, "Dispersion relation ω² = c²K² holds symbolically")
            else:
                return (False, f"Dispersion deviation: {eq}")
        except Exception as e:
            return (False, f"Cannot verify dispersion: {e}")

    def test_invariant_equations_b_and_d_identical(self) -> Tuple[bool, str]:
        """Check that equations (b) and (d) lead to the same condition."""
        if self.explanation_steps:
            combined = " ".join(self.explanation_steps).lower()
            if "same" in combined and ("equation" in combined or "constraint" in combined):
                return (True, "Explanation indicates (b) and (d) are identical")
            else:
                return (False, "Explanation does not state (b) and (d) give same condition")
        return (False, "SKIP: explanation_steps missing")

    def test_invariant_no_constraint_from_divB(self) -> Tuple[bool, str]:
        """Check that ∇·B=0 imposes no constraint."""
        if self.explanation_steps:
            combined = " ".join(self.explanation_steps).lower()
            if "identically" in combined or "no constraint" in combined:
                return (True, "Correctly identifies no constraint from ∇·B=0")
            else:
                return (False, "∇·B=0 constraint not discussed")
        return (False, "SKIP: explanation_steps missing")

    # ----------------------------------------------------------------------
    # Modeling/trace checks: derivation quality based on explanation steps
    # ----------------------------------------------------------------------
    def test_modeling_derivation_c_transversality(self) -> Tuple[bool, str]:
        """Check that derivation correctly yields K·A0=0 from ∇·E=0."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps missing")
        combined = " ".join(self.explanation_steps).lower()
        if "k·a" in combined and "0" in combined:
            return (True, "Derivation includes K·A0=0")
        else:
            return (False, "Transversality condition not derived in steps")

    def test_modeling_derivation_dispersion(self) -> Tuple[bool, str]:
        """Check that derivation yields ω = c|K|."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps missing")
        combined = " ".join(self.explanation_steps).lower()
        if ("omega" in combined and "c" in combined and "k" in combined) or \
           ("dispersion" in combined and "relation" in combined):
            return (True, "Derivation includes dispersion relation")
        else:
            return (False, "Dispersion relation not derived in steps")

    def test_modeling_pseudocode_consistency(self) -> Tuple[bool, str]:
        """Check that pseudocode properly uses transversality to derive dispersion."""
        if self.pseudocode is None:
            return (False, "SKIP: pseudocode missing")
        pseudo_str = str(self.pseudocode).lower()
        if "transversality" in pseudo_str or "k·a0" in pseudo_str:
            if "dispersion" in pseudo_str or "omega" in pseudo_str:
                return (True, "Pseudocode links transversality to dispersion")
        return (False, "Pseudocode does not show logical flow")

    def test_modeling_explanation_steps_not_empty(self) -> Tuple[bool, str]:
        if self.explanation_steps is None or len(self.explanation_steps) == 0:
            return (False, "No explanation steps provided")
        return (True, f"Explanation steps present: {len(self.explanation_steps)} steps")

    # ----------------------------------------------------------------------
    # Monotonicity checks: explicit input perturbation and baseline comparison
    # ----------------------------------------------------------------------
    def test_monotonicity_dispersion_increases_with_wavenumber(self) -> Tuple[bool, str]:
        """
        Check monotonic relation: as |K| increases, ω increases (dispersion relation ω = c|K|).
        We perturb the wavevector magnitude by a small factor and verify that
        the predicted frequency from the dispersion relation also increases.
        """
        if self.K is None or self.omega is None:
            return (False, "SKIP: K or omega missing")

        # Try numeric perturbation
        try:
            # Baseline values
            K_orig = [float(x) for x in self.K]
            norm_orig = math.sqrt(sum(k ** 2 for k in K_orig))
            omega_orig = float(self.omega)
            # Compute c from the given values; if c is symbolic, attempt numeric conversion
            try:
                c_val = float(self.c)
            except (TypeError, ValueError):
                # If c cannot be converted, derive it from the relation (assuming correct)
                if norm_orig > 0:
                    c_val = omega_orig / norm_orig
                else:
                    return (False, "Zero wavenumber, cannot compute c")

            # Perturb: increase |K| by 10%
            K_pert = [1.1 * k for k in K_orig]
            norm_pert = math.sqrt(sum(k ** 2 for k in K_pert))
            omega_pert = c_val * norm_pert

            # Check monotonicity: larger |K| -> larger ω
            if omega_pert > omega_orig:
                return (True, f"ω increases when |K| increases (ω_orig={omega_orig}, ω_pert={omega_pert})")
            else:
                return (False, f"ω does not increase: ω_orig={omega_orig}, ω_pert={omega_pert}")
        except (TypeError, ValueError, AttributeError) as e:
            return (False, f"Cannot perform numeric perturbation: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity checks (do not re-derive final answer)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_non_none(self) -> Tuple[bool, str]:
        """Check that final result values are not None and are sensible."""
        if 'K_dot_A0' not in self.results:
            return (False, "Missing 'K_dot_A0'")
        if 'dispersion' not in self.results:
            return (False, "Missing 'dispersion'")
        val_dot = self.results['K_dot_A0']
        disp = self.results['dispersion']
        if val_dot is None:
            return (False, "K_dot_A0 is None")
        if disp is None:
            return (False, "dispersion is None")
        # Basic string check for dispersion
        if isinstance(disp, str):
            if 'omega' not in disp.lower() or 'c' not in disp.lower():
                return (False, f"Dispersion string '{disp}' appears incomplete")
        return (True, "Result keys are present and non-null")

    # ----------------------------------------------------------------------
    # Main driver
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)