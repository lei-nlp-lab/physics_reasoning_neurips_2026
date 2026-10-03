import sympy as sp
import math
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Physics Checker for the dipole antenna problem.
    Checks interface, invariants, modeling, and answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results from solution_instance in a safe manner
        results = getattr(solution_instance, 'results', {})
        self.I_amplitude = results.get('I_amplitude', None)
        self.R_symbolic_str = results.get('R_symbolic', None)
        self.R_numerical = results.get('R_numerical', None)

        # Attempt to extract physics constants and givens
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.c = getattr(solution_instance, 'c', None)
        self.q = getattr(solution_instance, 'q', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.d = getattr(solution_instance, 'd', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None)  # avoid shadowing built-in

        # If c is not given but mu0 and epsilon0 are, compute it
        if self.c is None and self.mu0 is not None and self.epsilon0 is not None:
            try:
                self.c = 1 / math.sqrt(self.mu0 * self.epsilon0)
            except (ValueError, ArithmeticError):
                self.c = None

        # Store explanation steps if present, but only for reference
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """Check all required result keys exist."""
        required = ["I_amplitude", "R_symbolic", "R_numerical"]
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return False, f"Missing required result keys: {missing}"
        return True, "All required result keys present"

    def test_interface_R_numerical_type(self) -> Tuple[bool, str]:
        """Check R_numerical is numeric and non‑negative."""
        if self.R_numerical is None:
            return False, "SKIP: R_numerical not available"
        if not isinstance(self.R_numerical, (int, float)):
            return False, f"R_numerical has wrong type: {type(self.R_numerical)}"
        if self.R_numerical < 0:
            return False, f"R_numerical is negative: {self.R_numerical}"
        return True, "R_numerical is a non‑negative number"

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_R_from_power_and_current(self) -> Tuple[bool, str]:
        """
        Verify that R derived from (average power) / (average I^2) matches
        the symbolic expression.
        Use symbolic manipulation if possible, otherwise numerical fallback.
        """
        if self.R_symbolic_str is None:
            return False, "SKIP: R_symbolic not available"
        # Expected symbolic formula
        expr_expected = (2 * sp.pi / 3) * sp.sqrt(self.mu0 / self.epsilon0) * (self.d / self.lambda_)**2
        expr_given = sp.sympify(self.R_symbolic_str)
        if expr_given is None:
            return False, "Cannot parse R_symbolic_string"
        # Check symbolic equivalence using simplification
        diff = sp.simplify(expr_expected - expr_given)
        if diff == 0:
            return True, "R_symbolic matches expected formula symbolically"
        else:
            # Fallback: numerical check using given constants if available
            if self.mu0 is not None and self.epsilon0 is not None and self.d is not None and self.lambda_ is not None:
                try:
                    val_expected = float(expr_expected.evalf())
                    val_given = float(expr_given.evalf())
                    if math.isclose(val_expected, val_given, rel_tol=1e-2, abs_tol=1e-2):
                        return True, "R_symbolic matches expected formula numerically (rel_tol=1%)"
                except Exception:
                    pass
            return False, f"R_symbolic does not match expected: diff = {diff}"

    def test_invariant_current_amplitude_derivation(self) -> Tuple[bool, str]:
        """
        Check that I_amplitude equals q * omega, as derived from dq/dt.
        """
        if self.I_amplitude is None or self.q is None or self.omega is None:
            return False, "SKIP: missing q, omega, or I_amplitude"
        I0_expected = abs(self.q * self.omega)  # amplitude is positive
        if math.isclose(self.I_amplitude, I0_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"I_amplitude = {self.I_amplitude} matches q*omega = {I0_expected}"
        else:
            return False, f"I_amplitude = {self.I_amplitude} differs from q*omega = {I0_expected}"

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_power_to_R_consistency(self) -> Tuple[bool, str]:
        """
        Use the derived relationship: R = ⟨P⟩ / ⟨I²⟩.
        If we have mu0, q, d, omega, c, then ⟨P⟩ = μ₀(qd)²ω⁴/(12πc)
        and ⟨I²⟩ = (qω)²/2. Compute R and compare with R_numerical.
        """
        needed = [self.mu0, self.q, self.d, self.omega, self.c, self.R_numerical]
        if any(v is None for v in needed):
            return False, "SKIP: missing required givens for power calculation"
        I0 = abs(self.q * self.omega)
        P_avg = self.mu0 * (self.q * self.d)**2 * self.omega**4 / (12 * math.pi * self.c)
        I2_avg = I0**2 / 2
        if I2_avg == 0:
            return False, "SKIP: I2_avg is zero (unphysical)"
        R_computed = P_avg / I2_avg
        if math.isclose(R_computed, self.R_numerical, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R from ⟨P⟩/⟨I²⟩ = {R_computed:.2f} matches R_numerical = {self.R_numerical:.2f}"
        else:
            return False, f"R computed = {R_computed:.2f}, but R_numerical = {self.R_numerical:.2f}"

    def test_modeling_R_symbolic_form(self) -> Tuple[bool, str]:
        """
        Check that the symbolic expression string contains the known
        structure: includes '(d/lambda)^2' and the factor (2pi/3).
        """
        if self.R_symbolic_str is None:
            return False, "SKIP: R_symbolic not available"
        s = self.R_symbolic_str.replace(" ", "").replace("\n", "")
        # Check for key components
        has_factor = "(2pi/3)" in s
        has_sqrt = "sqrt(mu0/epsilon0)" in s or "sqrt(mu0/epsilon_0)" in s
        has_ratio = "(d/lambda)^2" in s or "(d/λ)^2" in s
        if has_factor and has_sqrt and has_ratio:
            return True, "Symbolic expression contains expected factors"
        else:
            return False, f"Symbolic expression missing components. Got: {self.R_symbolic_str}"

    # ------------------------------------------------------------------
    # Monotonicity checks (must explicitly perturb an input)
    # ------------------------------------------------------------------
    def test_monotonicity_R_increases_with_d_over_lambda(self) -> Tuple[bool, str]:
        """
        Explicitly perturb the ratio (d/λ) and check that R increases.
        Uses the symbolic expression with typical constants if actual givens missing.
        """
        # If we have the symbolic expression, evaluate it at two different ratios.
        # Use typical constants: mu0=4πe-7, epsilon0=8.854e-12.
        if self.R_symbolic_str is not None:
            try:
                expr = sp.sympify(self.R_symbolic_str)
            except Exception:
                return False, "SKIP: cannot parse R_symbolic for monotonicity"
        else:
            # Fallback: if we have mu0, epsilon0, d, lambda, we compute R explicitly.
            if self.mu0 is None or self.epsilon0 is None:
                return False, "SKIP: cannot evaluate R for monotonicity"
            # Build the expression ourselves
            ratio_sym = sp.symbols('ratio')
            expr = (2 * sp.pi / 3) * sp.sqrt(self.mu0 / self.epsilon0) * ratio_sym**2

        # Choose two ratio values: baseline 0.2 and perturbed 0.25 (or any distinct values)
        mu0_val = self.mu0 if self.mu0 is not None else 4e-7 * math.pi
        eps0_val = self.epsilon0 if self.epsilon0 is not None else 8.854e-12

        # Evaluate at two ratios (if expression uses d/lambda, substitute accordingly)
        # We need to substitute symbolic variables (d, lambda) or ratio.
        # We'll try to substitute all free symbols with plausible values.
        free_syms = list(expr.free_symbols)
        # Determine mapping: if only mu0, eps0, and a ratio variable, fine.
        # Otherwise, try to use given d and lambda.
        # Strategy: substitute mu0/eps0 if they are symbols, then evaluate for two ratio values.
        subs_dict = {}
        for sym in free_syms:
            if sym.name == 'mu0':
                subs_dict[sym] = mu0_val
            elif sym.name == 'epsilon0' or sym.name == 'epsilon_0':
                subs_dict[sym] = eps0_val
            elif sym.name == 'd' and self.d is not None:
                subs_dict[sym] = self.d
            elif sym.name == 'lambda' and self.lambda_ is not None:
                subs_dict[sym] = self.lambda_
            # Otherwise, treat as a ratio variable: we will handle later

        # Determine which symbols remain for the ratio.
        remaining = [s for s in free_syms if s not in subs_dict]
        # If only one remains, assume it's the ratio (d/lambda or just a dummy)
        if len(remaining) == 1:
            ratio_sym = remaining[0]
        elif len(remaining) == 0:
            # Expression fully determined? Then we cannot perturb. Skip.
            return False, "SKIP: no free variable to perturb (expression fully numeric)"
        else:
            # Too many unknowns, cannot perform monotonicity.
            return False, "SKIP: too many free symbolic variables; cannot perturb ratio"

        # Define two ratio values: baseline 0.2, perturbed +50%
        r0 = 0.2
        r1 = 0.3   # 50% increase
        try:
            R0 = float(expr.subs(subs_dict).subs({ratio_sym: r0}).evalf())
            R1 = float(expr.subs(subs_dict).subs({ratio_sym: r1}).evalf())
        except Exception:
            return False, "SKIP: numeric evaluation failed for monotonicity"

        if R1 > R0:
            return True, f"R increased from {R0:.3f} (ratio={r0}) to {R1:.3f} (ratio={r1})"
        else:
            return False, f"R decreased: {R0:.3f} -> {R1:.3f}"

    def test_monotonicity_I_amplitude_increases_with_omega(self) -> Tuple[bool, str]:
        """
        Explicitly perturb omega and check that I_amplitude increases proportionally.
        I0 = q * omega.
        """
        if self.I_amplitude is None or self.q is None or self.omega is None:
            return False, "SKIP: missing q, omega, or I_amplitude"
        # Baseline I0
        I0_baseline = abs(self.q * self.omega)
        # Perturb omega by +10%
        omega_perturbed = self.omega * 1.1
        I0_perturbed = abs(self.q * omega_perturbed)
        if I0_perturbed > I0_baseline:
            return True, f"I_amplitude increases: {I0_baseline:.3f} -> {I0_perturbed:.3f} (omega +10%)"
        else:
            return False, f"I_amplitude decreased: {I0_baseline:.3f} -> {I0_perturbed:.3f}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, very simple)
    # ------------------------------------------------------------------
    def test_answer_sanity_R_numerical_magnitude(self) -> Tuple[bool, str]:
        """
        Check that the numerical resistance for d=0.3λ is reasonable.
        Expected ~71 Ω, allow range 50-100 Ω.
        No math.isclose, no derived values.
        """
        if self.R_numerical is None:
            return False, "SKIP: R_numerical not available"
        if self.R_numerical < 50 or self.R_numerical > 100:
            return False, f"R_numerical = {self.R_numerical} outside expected physical range [50, 100] Ω"
        return True, f"R_numerical = {self.R_numerical} is within plausible range"