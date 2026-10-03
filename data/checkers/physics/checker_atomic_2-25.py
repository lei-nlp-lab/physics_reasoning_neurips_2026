import math
import sympy as sp
from typing import Any, Dict, List, Optional, Tuple


class PhysicsChecker:
    """
    Audits solutions for the nuclear binding energy problem.
    Checks consistency with semi-empirical mass formula, stability condition,
    Fermi gas symmetry term estimate, and monotonicity of Z/A ratio.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens, results, explanation steps, pseudocode safely
        self.givens: Optional[Dict[str, Any]] = getattr(
            solution_instance, "givens", None
        )
        self.results: Optional[Dict[str, Any]] = getattr(
            solution_instance, "results", None
        )
        self.explanation_steps: Optional[List[str]] = getattr(
            solution_instance, "explanation_steps", None
        )
        self.pseudocode: Optional[str] = getattr(
            solution_instance, "pseudocode", None
        )

        # Extract expected result keys with None defaults
        self.Z_relation = None
        self.a4_est = None
        if self.results is not None:
            self.Z_relation = self.results.get("Z_relation", None)
            self.a4_est = self.results.get("a4", None)

        # Try to extract a3 and a4 from givens (if provided)
        self.a3 = None
        self.a4 = None  # parameter used in Z_relation, not the estimated a4
        if self.givens is not None:
            self.a3 = self.givens.get("a3", None)
            self.a4 = self.givens.get("a4", None)

        # For Fermi gas estimate, look for R0 or other constants in givens
        self.R0 = None
        if self.givens is not None:
            self.R0 = self.givens.get("R0", None)  # e.g., 1.2 fm

    # ------------------------------------------------------------------
    # Test methods: each returns (passed: bool, message: str)
    # ------------------------------------------------------------------

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that expected result keys exist and are not None."""
        if self.results is None:
            return (False, "SKIP: 'results' attribute is None")
        missing = []
        for key in ["Z_relation", "a4"]:
            if key not in self.results or self.results[key] is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_Z_relation_format(self) -> Tuple[bool, str]:
        """Z_relation should be a string or sympy expression representing a formula."""
        if self.Z_relation is None:
            return (False, "SKIP: Z_relation not available")
        if not isinstance(self.Z_relation, (str, sp.Basic)):
            return (False, "Z_relation should be a string or sympy expression.")
        return (True, "Z_relation has valid format.")

    def test_interface_a4_format(self) -> Tuple[bool, str]:
        """a4 should be numeric (int or float)."""
        if self.a4_est is None:
            return (False, "SKIP: a4 not available")
        if not isinstance(self.a4_est, (int, float)):
            return (False, "a4 should be numeric (int or float).")
        return (True, "a4 has valid format.")

    def test_invariant_stability_condition(self) -> Tuple[bool, str]:
        """
        Verify that Z_relation satisfies ∂(BE)/∂Z = 0 for isobars.
        We use symbolic manipulation for the derivative.
        """
        if self.Z_relation is None:
            return (False, "SKIP: Z_relation not available")
        # Parse Z_relation if it's a string
        if isinstance(self.Z_relation, str):
            try:
                Z_expr = sp.sympify(self.Z_relation, locals={
                    'A': sp.Symbol('A'),
                    'a3': sp.Symbol('a3'),
                    'a4': sp.Symbol('a4'),
                })
            except Exception:
                return (False, "SKIP: Cannot parse Z_relation as sympy expression")
        else:
            Z_expr = self.Z_relation

        # Define symbols
        A, a3, a4 = sp.symbols('A a3 a4', positive=True)
        Z = sp.Symbol('Z', real=True)
        # Stability equation: -2*a3*Z*A^{-1/3} + 4*a4*(A-2*Z)*A^{-1} = 0
        stability_eq = -2*a3*Z*A**(-sp.Rational(1,3)) + 4*a4*(A-2*Z)*A**(-1)
        try:
            substituted = stability_eq.subs(Z, Z_expr)
            simplified = sp.simplify(substituted)
            if simplified == 0:
                return (True, "Z_relation satisfies ∂BE/∂Z=0 symbolically.")
            else:
                return (False, f"Z_relation does NOT satisfy ∂BE/∂Z=0; simplified to {simplified}")
        except Exception as e:
            return (False, f"Cannot perform symbolic substitution: {e}")

    def test_invariant_a4_positive(self) -> Tuple[bool, str]:
        """a4 estimate must be positive (binding energy symmetry term)."""
        if self.a4_est is None:
            return (False, "SKIP: a4 not available")
        if self.a4_est > 0:
            return (True, f"a4 = {self.a4_est} > 0")
        else:
            return (False, f"a4 = {self.a4_est} is not positive")

    def test_invariant_Z_relation_values(self) -> Tuple[bool, str]:
        """
        For a typical set of parameters (if available), the Z_relation should
        yield Z between 0 and A.
        """
        if self.Z_relation is None:
            return (False, "SKIP: Z_relation not available")
        if self.a3 is None or self.a4 is None:
            return (False, "SKIP: a3 and a4 not available in givens")

        if not isinstance(self.a3, (int, float)) or not isinstance(self.a4, (int, float)):
            return (False, "SKIP: a3 and a4 are not numeric")

        # Parse Z_relation if string
        if isinstance(self.Z_relation, str):
            try:
                Z_expr = sp.sympify(self.Z_relation, locals={
                    'A': sp.Symbol('A'),
                    'a3': sp.Symbol('a3'),
                    'a4': sp.Symbol('a4'),
                })
            except Exception:
                return (False, "SKIP: Cannot parse Z_relation")
        else:
            Z_expr = self.Z_relation

        A_vals = [50, 100, 150, 200]
        for A_val in A_vals:
            try:
                Z_val = float(Z_expr.subs({sp.Symbol('A'): A_val,
                                           sp.Symbol('a3'): self.a3,
                                           sp.Symbol('a4'): self.a4}))
            except Exception as e:
                return (False, f"Evaluation failed for A={A_val}: {e}")
            if not (0 < Z_val < A_val):
                return (False, f"Z={Z_val} not between 0 and A={A_val}")
        return (True, f"Z_relation gives plausible Z values for A={A_vals}")

    def test_modeling_symmetry_term_in_explanation(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps include the symmetry term structure
        a4*(A-2Z)^2/A (or equivalent form).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation_steps available")
        steps_text = "\n".join(self.explanation_steps)
        # Look for patterns: symmetry term, asymmetry term, a4*(A-2Z)^2/A
        keywords = ["symmetry", "asymmetry", "(A-2Z)^2", "a4"]
        found = [kw for kw in keywords if kw in steps_text]
        if len(found) >= 2:
            return (True, f"Explanation contains symmetry term keywords: {found}")
        else:
            return (False, "Explanation does not clearly include the symmetry term structure.")

    def test_modeling_fermi_expansion_derivation(self) -> Tuple[bool, str]:
        """
        Re-derive a4 from the Fermi gas model using standard physical constants.
        If the solution's a4 estimate is available, check that it is within 30%
        of the re-derived value. (This tests consistency with the Fermi gas model.)
        """
        if self.a4_est is None:
            return (False, "SKIP: no a4 estimate provided")

        # Use standard constants common in nuclear physics (approximate)
        hbar_c = 197.0  # MeV·fm
        m_n = 939.0    # MeV/c^2 (nucleon mass)
        R0 = 1.2       # fm (typical nuclear radius parameter)
        # If solution provided a different R0, we could use it; otherwise default.
        if self.R0 is not None and isinstance(self.R0, (int, float)):
            R0 = self.R0

        # Compute density: ρ0 = 3/(4π R0^3)
        rho0 = 3.0 / (4.0 * math.pi * R0**3)  # fm^{-3}
        # Compute (3π^2 ρ0)^{2/3}
        factor = (3 * math.pi**2 * rho0) ** (2/3)
        # Compute hbar^2/(2m) = (hbar*c)^2/(2*m*c^2)
        hbar2_over_2m = (hbar_c**2) / (2.0 * m_n)  # MeV·fm^2
        # Fermi energy epsilon_F = hbar2_over_2m * factor
        epsilon_F = hbar2_over_2m * factor
        # According to Fermi gas expansion (kinetic energy), a4 = epsilon_F / 6
        # More precisely: a4 = (1/6)*epsilon_F (see derivation in many textbooks)
        derived_a4 = epsilon_F / 6.0

        # Check if provided a4 is within 30% (rel_tol=0.3)
        if math.isclose(self.a4_est, derived_a4, rel_tol=0.3, abs_tol=5.0):
            return (True,
                    f"Provided a4={self.a4_est:.2f} is consistent with re-derived a4={derived_a4:.2f} (rel_tol=0.3)")
        else:
            return (False,
                    f"Provided a4={self.a4_est:.2f} deviates from re-derived a4={derived_a4:.2f} (expected ~epsilon_F/6)")

    def test_modeling_trace_pseudocode_steps(self) -> Tuple[bool, str]:
        """
        Check that the pseudocode (if present) includes the key steps:
        stability derivative, expansion of (1+delta/A)^(5/3), etc.
        This is a trace-consistency check.
        """
        if self.pseudocode is None:
            return (False, "SKIP: No pseudocode available")
        required_phrases = ["dBE/dZ", "stability", "expand", "(1+delta/A)^(5/3)", "a4"]
        found = [phrase for phrase in required_phrases if phrase in self.pseudocode]
        if len(found) >= 3:
            return (True, f"Pseudocode contains key steps: {found}")
        else:
            return (False, f"Pseudocode missing many key steps; found only: {found}")

    def test_monotonicity_Z_A_with_A_perturbation(self) -> Tuple[bool, str]:
        """
        Perturb A by +1 and check that Z/A ratio decreases (i.e., Z increases
        slower than A). This uses explicit baseline and perturbed input.
        """
        if self.Z_relation is None or self.a3 is None or self.a4 is None:
            return (False, "SKIP: Z_relation, a3 or a4 not available")
        if not all(isinstance(v, (int, float)) for v in [self.a3, self.a4]):
            return (False, "SKIP: a3 or a4 not numeric")

        # Parse Z_relation
        if isinstance(self.Z_relation, str):
            try:
                Z_expr = sp.sympify(self.Z_relation, locals={
                    'A': sp.Symbol('A'),
                    'a3': sp.Symbol('a3'),
                    'a4': sp.Symbol('a4'),
                })
            except Exception:
                return (False, "SKIP: Cannot parse Z_relation")
        else:
            Z_expr = self.Z_relation

        # Baseline A=100, perturbed A=101
        A_base = 100
        A_pert = 101
        try:
            Z_base = float(Z_expr.subs({sp.Symbol('A'): A_base,
                                        sp.Symbol('a3'): self.a3,
                                        sp.Symbol('a4'): self.a4}))
            Z_pert = float(Z_expr.subs({sp.Symbol('A'): A_pert,
                                        sp.Symbol('a3'): self.a3,
                                        sp.Symbol('a4'): self.a4}))
        except Exception as e:
            return (False, f"Evaluation failed: {e}")

        ratio_base = Z_base / A_base
        ratio_pert = Z_pert / A_pert
        if ratio_pert < ratio_base:
            return (True,
                    f"Z/A decreases from {ratio_base:.4f} to {ratio_pert:.4f} when A increases from {A_base} to {A_pert} (expected).")
        else:
            return (False,
                    f"Z/A does not decrease (base={ratio_base:.4f}, pert={ratio_pert:.4f})")

    def test_monotonicity_Z_with_a3_perturbation(self) -> Tuple[bool, str]:
        """
        Perturb a3 by +1% and check that Z decreases (since a3 appears
        in denominator of Z_relation). This is a proper input-perturbation test.
        """
        if self.Z_relation is None or self.a3 is None or self.a4 is None:
            return (False, "SKIP: Z_relation, a3 or a4 not available")
        if not all(isinstance(v, (int, float)) for v in [self.a3, self.a4]):
            return (False, "SKIP: a3 or a4 not numeric")

        # Parse Z_relation
        if isinstance(self.Z_relation, str):
            try:
                Z_expr = sp.sympify(self.Z_relation, locals={
                    'A': sp.Symbol('A'),
                    'a3': sp.Symbol('a3'),
                    'a4': sp.Symbol('a4'),
                })
            except Exception:
                return (False, "SKIP: Cannot parse Z_relation")
        else:
            Z_expr = self.Z_relation

        # Baseline and perturbed a3
        a3_base = self.a3
        a3_pert = self.a3 * 1.01  # 1% increase
        A_fixed = 100  # pick a fixed A
        try:
            Z_base = float(Z_expr.subs({sp.Symbol('A'): A_fixed,
                                        sp.Symbol('a3'): a3_base,
                                        sp.Symbol('a4'): self.a4}))
            Z_pert = float(Z_expr.subs({sp.Symbol('A'): A_fixed,
                                        sp.Symbol('a3'): a3_pert,
                                        sp.Symbol('a4'): self.a4}))
        except Exception as e:
            return (False, f"Evaluation failed: {e}")

        if Z_pert < Z_base:
            return (True,
                    f"Z decreases from {Z_base:.2f} to {Z_pert:.2f} when a3 increases by 1% (expected).")
        else:
            return (False,
                    f"Z does not decrease (base={Z_base:.2f}, pert={Z_pert:.2f})")

    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: Z_relation should yield finite values, a4 should be
        in a broad plausible range.
        """
        if self.Z_relation is None and self.a4_est is None:
            return (False, "SKIP: no results to check")
        all_ok = True
        msgs = []

        # Check Z_relation finite for a few A (if it's a string)
        if self.Z_relation is not None and isinstance(self.Z_relation, str):
            try:
                Z_expr = sp.sympify(self.Z_relation, locals={'A': sp.Symbol('A')})
                for A_val in [50, 100, 150]:
                    Z_val = float(Z_expr.subs(sp.Symbol('A'), A_val))
                    if not (math.isfinite(Z_val) and 0 < Z_val < A_val):
                        all_ok = False
                        msgs.append(f"Z_relation gives out-of-range for A={A_val}")
            except Exception:
                all_ok = False
                msgs.append("Cannot evaluate Z_relation")

        # Check a4 finite and positive and < 100 MeV
        if self.a4_est is not None and isinstance(self.a4_est, (int, float)):
            if not (0 < self.a4_est < 100):
                all_ok = False
                msgs.append(f"a4 = {self.a4_est} outside broad range (0,100) MeV")
        else:
            all_ok = False
            msgs.append("a4 not numeric or missing")

        if all_ok:
            return (True, "All weak sanity checks passed.")
        else:
            return (False, "; ".join(msgs))

    # ------------------------------------------------------------------
    # Runner: discover all test_* methods and execute them
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
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)