import math
import traceback
from typing import Any, Dict, List, Tuple, Union
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for the degenerate l=1 perturbation problem.
    Checks: interface, invariants (traceless sum, zero <L_z>), modeling (eigenstates),
    monotonicity (energy shifts w.r.t. parameters), and weak answer sanity (L_z exactly zero).
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safe extraction of results dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # Extract all required keys with None defaults
        self.eigenstates = self.results.get('eigenstates', None)
        self.Delta_E_x = self.results.get('Delta_E_x', None)
        self.Delta_E_y = self.results.get('Delta_E_y', None)
        self.Delta_E_z = self.results.get('Delta_E_z', None)
        self.diagram = self.results.get('diagram', None)
        self.L_z_x = self.results.get('L_z_x', None)
        self.L_z_y = self.results.get('L_z_y', None)
        self.L_z_z = self.results.get('L_z_z', None)

    # ---------- helper (safe sympy parsing) ----------
    def _safe_expr_eval(self, expr) -> sp.Expr:
        """Convert a string or numeric expression to sympy expression."""
        if expr is None:
            raise ValueError("Expression is None")
        if isinstance(expr, (int, float)):
            return sp.nsimplify(expr)
        if isinstance(expr, str):
            # Remove spaces, handle common symbols
            # Use locals to allow A, B, I as symbols
            return sp.sympify(expr, locals={'A': sp.Symbol('A'), 'B': sp.Symbol('B'), 'I': sp.Symbol('I')})
        if hasattr(expr, 'subs'):  # already sympy or similar
            return expr
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    # ---------- interface tests ----------
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that all expected keys are present in results."""
        required = ['eigenstates', 'Delta_E_x', 'Delta_E_y', 'Delta_E_z', 'diagram', 'L_z_x', 'L_z_y', 'L_z_z']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present")

    def test_interface_eigenstates_type(self) -> Tuple[bool, str]:
        """eigenstates must be a list of length 3."""
        if self.eigenstates is None:
            return (False, "SKIP: eigenstates not available")
        if not isinstance(self.eigenstates, list):
            return (False, "eigenstates is not a list")
        if len(self.eigenstates) != 3:
            return (False, f"eigenstates has length {len(self.eigenstates)}, expected 3")
        return (True, "eigenstates is a list of length 3")

    def test_interface_eigenstates_strings(self) -> Tuple[bool, str]:
        """Each element of eigenstates should be a string."""
        if self.eigenstates is None:
            return (False, "SKIP: eigenstates not available")
        if not all(isinstance(s, str) for s in self.eigenstates):
            return (False, "Not all eigenstates are strings")
        return (True, "All eigenstates are strings")

    def test_interface_Delta_E_available(self) -> Tuple[bool, str]:
        """Check that all Delta_E values are not None."""
        missing = []
        for name in ['Delta_E_x', 'Delta_E_y', 'Delta_E_z']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing energy shift keys: {missing}")
        return (True, "All Delta_E values present")

    def test_interface_Lz_available(self) -> Tuple[bool, str]:
        """Check L_z values exist."""
        missing = []
        for name in ['L_z_x', 'L_z_y', 'L_z_z']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing L_z keys: {missing}")
        return (True, "All L_z values present")

    # ---------- invariant tests (use isclose for numerical, sympy for symbolic) ----------
    def test_invariant_traceless_sum(self) -> Tuple[bool, str]:
        """V_pert is traceless: sum of Delta_E must be zero."""
        if None in (self.Delta_E_x, self.Delta_E_y, self.Delta_E_z):
            return (False, "SKIP: missing one or more Delta_E")
        try:
            expr_x = self._safe_expr_eval(self.Delta_E_x)
            expr_y = self._safe_expr_eval(self.Delta_E_y)
            expr_z = self._safe_expr_eval(self.Delta_E_z)
            total = sp.simplify(expr_x + expr_y + expr_z)
            # If total is a number, check isclose
            if total.is_Number:
                if math.isclose(float(total), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Sum = {float(total):.2e} ≈ 0")
                else:
                    return (False, f"Sum = {float(total):.2e}, not zero")
            # Symbolic: check if it simplifies to 0
            if total == 0:
                return (True, "Symbolic sum simplifies to zero")
            else:
                # Accept if it's an expression that is identically zero? Try substitution.
                free_syms = total.free_symbols
                if not free_syms:
                    # All numbers already handled
                    return (False, f"Sum expression does not simplify to zero: {total}")
                # Check via random numeric test
                # Use random values for free symbols (A,B,I). If sum is identically zero, should hold.
                import random
                try:
                    for _ in range(5):
                        subs = {s: random.uniform(-5,5) for s in free_syms}
                        val = float(total.subs(subs))
                        if not math.isclose(val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                            return (False, f"Sum not zero for random sample: {val}")
                    return (True, "Symbolic sum numerically zero for random samples")
                except Exception:
                    return (False, f"Could not verify sum: {total}")
        except Exception as e:
            return (False, f"Cannot evaluate sum: {e}")

    def test_invariant_Lz_zero(self) -> Tuple[bool, str]:
        """Expectation values of L_z must be zero for all three states."""
        for name, val in [('L_z_x', self.L_z_x), ('L_z_y', self.L_z_y), ('L_z_z', self.L_z_z)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            try:
                # Try numeric
                num = float(val)
                if not math.isfinite(num):
                    return (False, f"{name} is not finite: {val}")
                if not math.isclose(num, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"{name} = {num} not zero within tolerances")
            except (ValueError, TypeError):
                # Possibly symbolic, treat string "0" or symbolic 0
                if isinstance(val, str):
                    if val.strip() == '0':
                        continue
                    # Try sympy evaluation
                    try:
                        expr = sp.sympify(val)
                        if expr == 0:
                            continue
                        else:
                            return (False, f"{name} = {val}, not zero")
                    except Exception:
                        return (False, f"{name} = {val}, cannot interpret as zero")
                else:
                    return (False, f"{name} type {type(val)} not handled")
        return (True, "All L_z expectations are zero")

    # ---------- modeling tests ----------
    def test_modeling_eigenstates_content(self) -> Tuple[bool, str]:
        """The three eigenstates should be exactly |x>, |y>, |z> (order irrelevant)."""
        if self.eigenstates is None:
            return (False, "SKIP: eigenstates not available")
        expected = {'|x>', '|y>', '|z>'}
        actual = set(self.eigenstates)
        if actual == expected:
            return (True, "Eigenstates set matches expected {|x>,|y>,|z>}")
        else:
            return (False, f"Eigenstates set = {actual}, expected {expected}")

    def test_modeling_diagram_nonempty(self) -> Tuple[bool, str]:
        """The diagram description should be a non-empty string."""
        if self.diagram is None:
            return (False, "SKIP: diagram not available")
        if not isinstance(self.diagram, str) or len(self.diagram.strip()) == 0:
            return (False, "Diagram is not a non-empty string")
        return (True, "Diagram is a non-empty string")

    # ---------- monotonicity tests (explicit parameter perturbations) ----------
    def test_monotonicity_Delta_E_x_increases_with_A(self) -> Tuple[bool, str]:
        """
        For fixed B and I > 0, increasing A should increase ΔE_x.
        We evaluate the expression for ΔE_x at two different A values and check the trend.
        """
        if self.Delta_E_x is None:
            return (False, "SKIP: Delta_E_x not available")
        try:
            expr = self._safe_expr_eval(self.Delta_E_x)
            # Assume free symbols include A, B, I
            free = expr.free_symbols
            # Use numeric substitutions: set B=1, I=1 as baseline (positive)
            # Test at A=1 and A=2
            subs_base = {sp.Symbol('A'): 1, sp.Symbol('B'): 1, sp.Symbol('I'): 1}
            subs_pert = {sp.Symbol('A'): 2, sp.Symbol('B'): 1, sp.Symbol('I'): 1}
            # Only include symbols present
            subs_base = {s: v for s, v in subs_base.items() if s in free}
            subs_pert = {s: v for s, v in subs_pert.items() if s in free}
            val_base = float(expr.subs(subs_base))
            val_pert = float(expr.subs(subs_pert))
            if val_pert > val_base:
                return (True, f"ΔE_x increased: {val_base} -> {val_pert}")
            else:
                return (False, f"ΔE_x did not increase: {val_base} -> {val_pert}")
        except Exception as e:
            return (False, f"Could not evaluate monotonicity: {e}")

    def test_monotonicity_Delta_E_z_decreases_with_A(self) -> Tuple[bool, str]:
        """
        For fixed B and I > 0, increasing A should decrease ΔE_z (since coefficient is -(A+B)).
        """
        if self.Delta_E_z is None:
            return (False, "SKIP: Delta_E_z not available")
        try:
            expr = self._safe_expr_eval(self.Delta_E_z)
            free = expr.free_symbols
            subs_base = {sp.Symbol('A'): 1, sp.Symbol('B'): 1, sp.Symbol('I'): 1}
            subs_pert = {sp.Symbol('A'): 2, sp.Symbol('B'): 1, sp.Symbol('I'): 1}
            subs_base = {s: v for s, v in subs_base.items() if s in free}
            subs_pert = {s: v for s, v in subs_pert.items() if s in free}
            val_base = float(expr.subs(subs_base))
            val_pert = float(expr.subs(subs_pert))
            if val_pert < val_base:
                return (True, f"ΔE_z decreased: {val_base} -> {val_pert}")
            else:
                return (False, f"ΔE_z did not decrease: {val_base} -> {val_pert}")
        except Exception as e:
            return (False, f"Could not evaluate monotonicity: {e}")

    # ---------- weak answer sanity (no isclose, no re-derivation) ----------
    def test_answer_sanity_Lz_zero_exact(self) -> Tuple[bool, str]:
        """L_z values should be exactly zero (int/float 0 or string '0'). Not NaN."""
        for name, val in [('L_z_x', self.L_z_x), ('L_z_y', self.L_z_y), ('L_z_z', self.L_z_z)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            # Check not NaN
            if isinstance(val, float) and math.isnan(val):
                return (False, f"{name} is NaN")
            # Check exactly zero as number or string
            acceptable = (isinstance(val, (int, float)) and val == 0.0) or \
                         (isinstance(val, str) and val.strip() in ('0', '0.0'))
            if not acceptable:
                return (False, f"{name} is not exactly zero: {val!r}")
        return (True, "All L_z values are exactly zero")

    # ---------- runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all test_ methods
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)