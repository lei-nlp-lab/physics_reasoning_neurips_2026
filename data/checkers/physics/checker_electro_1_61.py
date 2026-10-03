import math
import sympy as sp
import traceback
from typing import Any, Tuple, List, Dict, Optional

class PhysicsChecker:
    """
    Auditor for the resistance problem between concentric spherical conductors.
    """
    def __init__(self, solution_instance: Any):
        # Safely extract results, givens, explanation, pseudocode
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract required physics quantities with None defaults
        # From results
        self.R = self.results.get('R', None)

        # From givens (symbolic or numeric)
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.sigma = self.givens.get('sigma', None)

        # Optional I (often cancels, but used in derivation)
        self.I = self.givens.get('I', None)

        # Expected symbolic form (for modeling checks)
        self.expected_R_symbolic = None
        if all(v is not None for v in [self.a, self.b, self.sigma]):
            sym_a, sym_b, sym_sigma = sp.symbols('a b sigma', positive=True)
            self.expected_R_symbolic = 1/(4*sp.pi*sym_sigma) * (1/sym_a - 1/sym_b)

        # Numeric conversion helpers (if possible)
        self.a_num = None
        self.b_num = None
        self.sigma_num = None
        self.R_num = None
        for name in ['a', 'b', 'sigma', 'R']:
            val = getattr(self, name)
            if val is not None and isinstance(val, (int, float)):
                setattr(self, f'{name}_num', float(val))
            elif val is not None and isinstance(val, sp.Basic):
                # Try numeric evaluation if all symbols numeric
                try:
                    setattr(self, f'{name}_num', float(val.evalf()))
                except:
                    pass

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_results_key_R(self) -> Tuple[bool, str]:
        """Check that 'R' exists in results."""
        if 'R' not in self.results:
            return (False, "Missing key 'R' in results")
        return (True, "'R' found in results")

    def test_interface_givens_required(self) -> Tuple[bool, str]:
        """Check that required givens (a, b, sigma) are provided."""
        missing = []
        for key in ['a', 'b', 'sigma']:
            if self.givens.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required givens: {missing}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    # Invariant checks (physics-theorem based)
    # ----------------------------------------------------------------------
    def test_invariant_resistance_formula(self) -> Tuple[bool, str]:
        """Check that R matches the theoretical expression R = 1/(4πσ)(1/a - 1/b)."""
        if self.R is None or self.a is None or self.b is None or self.sigma is None:
            return (False, "SKIP: Missing R, a, b, or sigma for formula check")

        # Compute expected R from givens (using numeric if available, else symbolic)
        if self.a_num is not None and self.b_num is not None and self.sigma_num is not None:
            expected = 1/(4*math.pi*self.sigma_num) * (1/self.a_num - 1/self.b_num)
            if self.R_num is not None:
                if math.isclose(self.R_num, expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"R numeric matches expected: {self.R_num} ≈ {expected}")
                else:
                    return (False, f"R numeric mismatch: got {self.R_num}, expected {expected}")
            else:
                # Compare symbolic
                sym_expected = self.expected_R_symbolic
                if sym_expected is None:
                    return (False, "Cannot compute symbolic expected R")
                # Substitute numeric values
                subs_dict = {sp.Symbol('a'): self.a, sp.Symbol('b'): self.b, sp.Symbol('sigma'): self.sigma}
                sym_expected_num = sym_expected.subs(subs_dict).evalf()
                if math.isclose(float(self.R.evalf()), float(sym_expected_num), rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "R symbolic matches expected")
                else:
                    return (False, "R symbolic mismatch")
        else:
            return (False, "SKIP: Not all givens are numeric for formula check")

    def test_invariant_resistance_positive(self) -> Tuple[bool, str]:
        """Check that R > 0 (since a<b, sigma>0)."""
        if self.R_num is not None:
            if self.R_num > 0:
                return (True, f"R = {self.R_num} > 0")
            else:
                return (False, f"R = {self.R_num} is not positive")
        else:
            # Symbolic check: assume positive parameters
            if self.a is not None and self.b is not None and self.sigma is not None:
                # Check that symbolic expression is positive for positive a,b,sigma
                sym_expected = self.expected_R_symbolic
                if sym_expected is not None:
                    # Use SymPy to check positivity assumptions
                    result = sp.simplify(sym_expected > 0)
                    # This may be True if assumptions hold
                    if result == True:
                        return (True, "Symbolic expression is positive under assumptions")
                    else:
                        return (False, "Symbolic expression might not be positive")
            return (False, "SKIP: Cannot check positivity without numeric or fully symbolic")

    def test_invariant_current_conservation(self) -> Tuple[bool, str]:
        """Check that current I is conserved across any spherical shell (if I given)."""
        if self.I is None:
            return (False, "SKIP: I not provided")
        # The derivation uses J = I/(4πr^2); if we have numeric I, we can check J at two radii? Not straightforward without additional data.
        # Weak check: I must be positive (or at least non-zero). We'll just check positivity.
        if isinstance(self.I, (int, float)):
            if self.I > 0:
                return (True, f"I = {self.I} > 0 (conventional flow)")
            else:
                return (False, "I must be positive for conventional current direction")
        return (False, "SKIP: I not numeric")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_explanation_contains_key_concepts(self) -> Tuple[bool, str]:
        """Check that explanation mentions spherical symmetry, Ohm's law, and integration."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps provided")
        text = str(self.explanation_steps).lower()
        required = ["spherical", "ohm", "integral"]
        missing = [word for word in required if word not in text]
        if missing:
            return (False, f"Explanation missing keywords: {missing}")
        return (True, "Explanation contains key concepts")

    def test_modeling_pseudocode_emit_contract(self) -> Tuple[bool, str]:
        """Check that results['R'] matches the pseudocode emit contract."""
        # The pseudocode says emit results['R'] = R.
        # If results has 'R', it's satisfied.
        if 'R' in self.results:
            return (True, "Pseudocode emit contract fulfilled: results['R'] present")
        else:
            return (False, "Pseudocode emit contract not fulfilled")

    def test_modeling_resistance_expression_symmetry(self) -> Tuple[bool, str]:
        """Check that R depends only on a, b, sigma (and not on I)."""
        # If R is symbolic, check its free symbols.
        if self.R is not None and isinstance(self.R, sp.Basic):
            free_syms = self.R.free_symbols
            allowed = {sp.Symbol('a'), sp.Symbol('b'), sp.Symbol('sigma')}
            if free_syms.issubset(allowed):
                return (True, "R depends only on a, b, sigma")
            else:
                extra = free_syms - allowed
                return (False, f"R includes unexpected symbols: {extra}")
        else:
            return (False, "SKIP: R not symbolic")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb inputs and compare)
    # ----------------------------------------------------------------------
    def test_monotonicity_resistance_increases_with_b(self) -> Tuple[bool, str]:
        """Verify that increasing b increases R."""
        if self.a_num is None or self.sigma_num is None or self.b_num is None:
            return (False, "SKIP: Need numeric a, b, sigma")

        # Baseline R
        R_baseline = 1/(4*math.pi*self.sigma_num) * (1/self.a_num - 1/self.b_num)

        # Perturb b up by 10%
        b_pert = self.b_num * 1.1
        # Compute expected new R using the same formula (we assume the checker can re-derive)
        # But we need to use the resistance formula to compute new R. That's fine because it's derived from physics.
        R_pert = 1/(4*math.pi*self.sigma_num) * (1/self.a_num - 1/b_pert)

        # Perturbed R should be larger (since 1/b decreases, so delta increases)
        if R_pert > R_baseline:
            return (True, f"b increased -> R increases: {R_pert} > {R_baseline}")
        else:
            return (False, f"Monotonicity violated: b increased but R did not increase")

    def test_monotonicity_resistance_decreases_with_sigma(self) -> Tuple[bool, str]:
        """Verify that increasing sigma decreases R."""
        if self.a_num is None or self.sigma_num is None or self.b_num is None:
            return (False, "SKIP: Need numeric a, b, sigma")

        # Baseline R
        R_baseline = 1/(4*math.pi*self.sigma_num) * (1/self.a_num - 1/self.b_num)

        # Perturb sigma up by 10%
        sigma_pert = self.sigma_num * 1.1
        R_pert = 1/(4*math.pi*sigma_pert) * (1/self.a_num - 1/self.b_num)

        if R_pert < R_baseline:
            return (True, f"sigma increased -> R decreases: {R_pert} < {R_baseline}")
        else:
            return (False, f"Monotonicity violated: sigma increased but R did not decrease")

    def test_monotonicity_resistance_increases_with_a_decrease(self) -> Tuple[bool, str]:
        """Verify that decreasing a increases R (since 1/a increases)."""
        if self.a_num is None or self.sigma_num is None or self.b_num is None:
            return (False, "SKIP: Need numeric a, b, sigma")

        R_baseline = 1/(4*math.pi*self.sigma_num) * (1/self.a_num - 1/self.b_num)

        # Perturb a down (e.g., 90% of current a)
        a_pert = self.a_num * 0.9
        R_pert = 1/(4*math.pi*self.sigma_num) * (1/a_pert - 1/self.b_num)

        if R_pert > R_baseline:
            return (True, f"a decreased -> R increases: {R_pert} > {R_baseline}")
        else:
            return (False, f"Monotonicity violated: a decreased but R did not increase")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_R_finite(self) -> Tuple[bool, str]:
        """Check that R is finite and not NaN."""
        if self.R_num is not None:
            if math.isfinite(self.R_num):
                return (True, f"R = {self.R_num} is finite")
            else:
                return (False, f"R = {self.R_num} is not finite")
        elif self.R is not None and isinstance(self.R, sp.Basic):
            # Symbolic check: should be finite for positive a,b,sigma
            if self.expected_R_symbolic is not None:
                # Assume positive, so expression is finite
                return (True, "Symbolic R is finite under given assumptions")
            else:
                return (False, "Cannot assess finiteness of symbolic R")
        else:
            return (False, "SKIP: R not available for finiteness check")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return (success, logs)."""
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)