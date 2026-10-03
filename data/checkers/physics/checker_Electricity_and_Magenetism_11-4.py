import math
import traceback
import re
from typing import Any, Tuple, List

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class PhysicsChecker:
    """
    Auditor for the 7-element dipole array problem.
    Checks interface completeness, theorem-based invariants, modeling quality,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results safely; default to None if missing
        self.pattern = getattr(solution_instance, 'results', {}).get('pattern', None)
        # Also extract other potentially useful data (not required but for robustness)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # target_answers not used to avoid hardcoded comparisons

        # Ensure pattern is a string; otherwise set to None
        if not isinstance(self.pattern, str):
            self.pattern = None

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_pattern_exists(self) -> Tuple[bool, str]:
        """Check that the 'pattern' key exists and is a non-empty string."""
        if self.pattern is None:
            return (False, "Missing or empty 'pattern' in results")
        return (True, "pattern exists and is a string")

    def test_interface_pattern_contains_components(self) -> Tuple[bool, str]:
        """Check that the pattern string contains the expected physical quantities."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        # Must mention sin^2θ and sin(7δ)/sin(δ) or equivalent
        has_sin2 = re.search(r'sin\s*\^?\s*2\s*[θθ]', self.pattern, re.IGNORECASE)
        has_array_factor = re.search(r'sin\s*\(7[δd]\)\s*/\s*sin\s*\([δd]\)', self.pattern, re.IGNORECASE)
        if has_sin2 and has_array_factor:
            return (True, "Pattern contains sin^2θ factor and (sin(7δ)/sin(δ)) array factor")
        else:
            return (False, f"Pattern lacks required factors: sin^2θ and array factor. Got: {self.pattern[:100]}")

    # ------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_sin2_factor(self) -> Tuple[bool, str]:
        """Verify that the pattern explicitly contains sin^2θ (or sin^2 theta)."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        found = bool(re.search(r'(sin\s*\^?\s*2\s*[θθ]|sin\s*[θθ]\s*\^?\s*2)', self.pattern, re.IGNORECASE))
        if found:
            return (True, "Pattern includes sin^2θ factor (dipole emission)")
        else:
            return (False, "Pattern does not explicitly show sin^2θ factor")

    def test_invariant_array_factor_real(self) -> Tuple[bool, str]:
        """Check that the array factor is purely real (no imaginary unit)."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        # The array factor squared should be real. Look for explicit i or j.
        # This is a weak check: ensure no 'i' or 'j' appears as a standalone symbol.
        if re.search(r'(?<![a-zA-Z])i(?![a-zA-Z])', self.pattern) or \
           re.search(r'(?<![a-zA-Z])j(?![a-zA-Z])', self.pattern):
            return (False, "Pattern contains imaginary unit i/j; array factor should be real")
        # Also check presence of trig functions
        has_trig = re.search(r'sin|cos', self.pattern, re.IGNORECASE)
        if has_trig:
            return (True, "Array factor appears real (trigonometric functions present, no imaginary unit)")
        else:
            return (False, "Cannot confirm real array factor; missing trig functions")

    # ------------------------------------------------------------
    # Modeling quality checks (using explanation_steps and physics)
    # ------------------------------------------------------------
    def test_modeling_delta_definition(self) -> Tuple[bool, str]:
        """Check that δ is defined as (π/2) sinθ cosφ."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        # Look for text like "delta = (pi/2) sintheta cosphi" or similar
        delta_def = re.search(r'δ\s*=\s*\(?\s*π\s*/\s*2\s*\)?\s*sin\s*[θθ]\s*cos\s*[φϕ]',
                              self.pattern, re.IGNORECASE)
        if delta_def:
            return (True, "Delta correctly defined as (π/2) sinθ cosφ")
        # Alternative: 'delta = (pi/2) sintheta cosphi'
        delta_def2 = re.search(r'delta\s*=\s*\(?\s*π\s*/\s*2\s*\)?\s*sin\s*[θθ]\s*cos\s*[φϕ]',
                               self.pattern, re.IGNORECASE)
        if delta_def2:
            return (True, "Delta correctly defined as (π/2) sinθ cosφ")
        return (False, "Delta definition not found in pattern string; expected 'δ = (π/2) sinθ cosφ'")

    def test_modeling_maximum_at_u_zero(self) -> Tuple[bool, str]:
        """Numerically verify that the intensity is maximized when u = sinθ cosφ = 0."""
        if not SYMPY_AVAILABLE:
            return (False, "SKIP: sympy not available for numerical evaluation")
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        expr = self._extract_expression()
        if expr is None:
            return (False, "Could not parse pattern expression")

        theta, phi = sp.symbols('theta phi', real=True)
        u = sp.sin(theta) * sp.cos(phi)
        # Evaluate at two points: one with u=0, one with u≠0
        try:
            f = sp.lambdify((theta, phi), expr, modules='math')
        except Exception as e:
            return (False, f"Lambdify failed: {e}")

        # Baseline: u=0 => θ=π/2, φ=π/2
        # Perturbed: u=1 => θ=π/2, φ=0
        try:
            val_baseline = f(math.pi/2, math.pi/2)   # u=0
            val_perturbed = f(math.pi/2, 0.0)        # u=1
        except Exception as e:
            return (False, f"Evaluation failed: {e}")

        if not (math.isfinite(val_baseline) and math.isfinite(val_perturbed)):
            return (False, "Intensity values not finite")
        if val_baseline <= 0 or val_perturbed <= 0:
            return (False, "Intensity should be positive")

        # The ratio should be large because S(0)=7, S(π/2)=1 -> ratio~49
        ratio = val_baseline / val_perturbed
        if ratio > 10:
            return (True, f"Maximum at u=0 confirmed: ratio = {ratio:.2f} (> 10)")
        else:
            return (False, f"Ratio too small: {ratio:.2f}, expected >>1")

    def test_modeling_array_factor_squared(self) -> Tuple[bool, str]:
        """Check that the pattern explicitly contains the array factor squared: (sin(7δ)/sin(δ))^2."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        # Look for squared ratio
        squared_ratio = re.search(r'\(\s*sin\s*\(7[δd]\)\s*/\s*sin\s*\([δd]\)\s*\)\s*\^?\s*2',
                                  self.pattern, re.IGNORECASE)
        if squared_ratio:
            return (True, "Array factor squared present: (sin(7δ)/sin(δ))^2")
        # Alternative: sin^2(7δ)/sin^2(δ)
        alt = re.search(r'sin\s*\^?\s*2\s*\(7[δd]\)\s*/\s*sin\s*\^?\s*2\s*\([δd]\)',
                        self.pattern, re.IGNORECASE)
        if alt:
            return (True, "Array factor squared present: sin^2(7δ)/sin^2(δ)")
        return (False, "Array factor squared not found in pattern")

    # ------------------------------------------------------------
    # Monotonicity checks (must perturb input and compare output)
    # ------------------------------------------------------------
    def test_monotonicity_versus_u(self) -> Tuple[bool, str]:
        """Check that, for fixed sinθ=1, intensity decreases as |u| increases from zero."""
        if not SYMPY_AVAILABLE:
            return (False, "SKIP: sympy not available")
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        expr = self._extract_expression()
        if expr is None:
            return (False, "Could not parse pattern expression")

        theta, phi = sp.symbols('theta phi', real=True)
        # Fix θ=π/2 so that u=cosφ
        try:
            f_pi2 = sp.lambdify(phi, expr.subs(theta, sp.pi/2), modules='math')
        except Exception as e:
            return (False, f"Lambdify failed: {e}")

        # Baseline: φ=π/2 => u=0 (maximum)
        # Perturbation 1: φ=π/2 - 0.1 => u=cos(1.47)≈0.0? Actually cos(π/2 - 0.1) ≈ 0.0998
        # Perturbation 2: φ=π/2 + 0.1 => u≈ -0.0998 (same absolute value)
        # Expect both perturbed values to be less than baseline.
        phi_base = math.pi/2.0
        phi_pert1 = phi_base - 0.1
        phi_pert2 = phi_base + 0.1

        try:
            base_val = f_pi2(phi_base)
            pert1_val = f_pi2(phi_pert1)
            pert2_val = f_pi2(phi_pert2)
        except Exception as e:
            return (False, f"Evaluation failed: {e}")

        if not (math.isfinite(base_val) and math.isfinite(pert1_val) and math.isfinite(pert2_val)):
            return (False, "Non-finite values encountered")
        if base_val <= 0:
            return (False, "Baseline intensity should be positive")

        # Check that both perturbed values are less than baseline (within tolerance)
        tol = 1e-6
        if pert1_val <= base_val + tol and pert2_val <= base_val + tol:
            return (True, f"Monotonicity: baseline = {base_val:.4f}, perturbed = {pert1_val:.4f}, {pert2_val:.4f} (< baseline)")
        else:
            return (False, f"Monotonicity failed: baseline={base_val:.4f}, pert1={pert1_val:.4f}, pert2={pert2_val:.4f}")

    # ------------------------------------------------------------
    # Weak answer sanity (no exact value comparisons, just sign/finite/range)
    # ------------------------------------------------------------
    def test_answer_sanity_intensity_nonnegative(self) -> Tuple[bool, str]:
        """Check that the pattern expression is non-negative for a few test points."""
        if self.pattern is None:
            return (False, "SKIP: pattern missing")
        if not SYMPY_AVAILABLE:
            # Fallback: look for squares
            if '^2' in self.pattern or '**2' in self.pattern:
                return (True, "Pattern contains squares – likely non-negative")
            else:
                return (False, "Cannot verify non-negativity without sympy")
        expr = self._extract_expression()
        if expr is None:
            return (False, "Cannot parse expression")
        theta, phi = sp.symbols('theta phi', real=True)
        try:
            f = sp.lambdify((theta, phi), expr, modules='math')
        except:
            return (False, "Lambdify failed")
        test_points = [(0.5, 0.2), (1.3, 2.1), (math.pi/4, math.pi/3)]
        for t, p in test_points:
            try:
                v = f(t, p)
                if not (v >= -1e-12):  # allow tiny negative due to rounding
                    return (False, f"Intensity negative at (θ={t:.2f}, φ={p:.2f}): value = {v:.4f}")
            except:
                return (False, f"Evaluation failed at ({t:.2f},{p:.2f})")
        return (True, "Intensity is non-negative for tested points")

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------
    def _extract_expression(self) -> sp.Expr:
        """
        Try to extract a sympy expression from the pattern string.
        The expected format: 'I  prop  sin^2theta * (sin(7delta)/sindelta)^2 with delta = ...'
        We'll parse the part before 'with' if present.
        """
        if self.pattern is None:
            return None
        s = self.pattern
        # Remove the 'with delta = ...' part
        s = re.split(r'\bwith\b', s, maxsplit=1, flags=re.IGNORECASE)[0]
        # Remove 'I prop' or similar
        s = re.sub(r'\bI\s*prop\b', '', s).strip()
        if not s:
            return None
        # Convert human-readable symbols to sympy
        s = s.replace('^', '**')
        s = s.replace('θ', 'theta')
        s = s.replace('φ', 'phi')
        s = s.replace('π', 'pi')
        s = s.replace('δ', 'delta')
        # Handle sin^2theta -> (sin(theta))**2
        s = re.sub(r'sin\s*\*\*\s*2\s*θ', 'sin(theta)**2', s, flags=re.IGNORECASE)
        s = re.sub(r'sin\s*\*\*\s*2\s*theta', 'sin(theta)**2', s, flags=re.IGNORECASE)
        s = re.sub(r'sindelta', 'sin(delta)', s, flags=re.IGNORECASE)
        s = re.sub(r'sin\s*\(7delta\)', 'sin(7*delta)', s, flags=re.IGNORECASE)
        s = re.sub(r'sin\s*\(delta\)', 'sin(delta)', s, flags=re.IGNORECASE)
        # Try sympy parsing
        try:
            expr = sp.parse_expr(s)
            return expr
        except Exception:
            # Fallback eval with namespace
            namespace = {'sin': sp.sin, 'cos': sp.cos, 'theta': sp.Symbol('theta'),
                         'phi': sp.Symbol('phi'), 'pi': sp.pi, 'delta': sp.Symbol('delta')}
            try:
                expr = eval(s, {"__builtins__": None}, namespace)
                return expr
            except:
                return None

    # ------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods and run them, collecting results."""
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