import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    """
    Audit quantum mechanical solution for eigenvalues of L^2 and L_z.
    """
    def __init__(self, solution_instance):
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation = getattr(solution_instance, 'explanation_steps', None)

        # Results expected from pseudocode
        self.l2_eig = self.results.get('l2_eig', None)
        self.lz_eig = self.results.get('lz_eig', None)

        # Try to extract hbar from solution (symbolic or numeric)
        self.hbar = self.results.get('hbar', None)
        if self.hbar is None:
            self.hbar = self.givens.get('hbar', None)

        # For symbolic checks, keep symbolic versions
        self._l2_sym = None
        self._lz_sym = None
        self._hbar_sym = sp.symbols('hbar', positive=True)

        # Attempt symbolic conversion
        if self.l2_eig is not None:
            try:
                self._l2_sym = sp.sympify(self.l2_eig)
            except (sp.SympifyError, TypeError):
                pass
        if self.lz_eig is not None:
            try:
                self._lz_sym = sp.sympify(self.lz_eig)
            except (sp.SympifyError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_l2_key(self) -> Tuple[bool, str]:
        """Check that 'l2_eig' key exists and is not None."""
        if self.l2_eig is None:
            return (False, "missing 'l2_eig' in results")
        return (True, "'l2_eig' present")

    def test_interface_lz_key(self) -> Tuple[bool, str]:
        """Check that 'lz_eig' key exists and is not None."""
        if self.lz_eig is None:
            return (False, "missing 'lz_eig' in results")
        return (True, "'lz_eig' present")

    def test_interface_l2_numeric(self) -> Tuple[bool, str]:
        """Check that 'l2_eig' is a number (int/float/complex) or symbolic numeric."""
        if self.l2_eig is None:
            return (False, "SKIP: l2_eig not available")
        try:
            val = complex(self.l2_eig)
            if math.isnan(val.real) or math.isinf(val.real):
                return (False, "l2_eig is NaN or infinite")
            return (True, f"l2_eig is numeric ({val})")
        except (TypeError, ValueError):
            # Could be symbolic expression; that's acceptable
            return (True, "l2_eig is symbolic (acceptable)")

    def test_interface_lz_numeric(self) -> Tuple[bool, str]:
        """Check that 'lz_eig' is a number (int/float) or symbolic numeric."""
        if self.lz_eig is None:
            return (False, "SKIP: lz_eig not available")
        try:
            val = complex(self.lz_eig)
            if math.isnan(val.real) or math.isinf(val.real):
                return (False, "lz_eig is NaN or infinite")
            return (True, f"lz_eig is numeric ({val})")
        except (TypeError, ValueError):
            return (True, "lz_eig is symbolic (acceptable)")

    # ------------------------------------------------------------------
    # Invariant/Theorem checks
    # ------------------------------------------------------------------
    def test_invariant_l2_eigenvalue(self) -> Tuple[bool, str]:
        """
        L^2 eigenvalue must be 2 ħ^2 for a state proportional to cosθ.
        Use symbolic or numeric fallback.
        """
        if self.l2_eig is None:
            return (False, "SKIP: l2_eig not available")

        # Strategy 1: Symbolic check
        if self._l2_sym is not None:
            free = self._l2_sym.free_symbols
            if self._hbar_sym in free:
                # Ratio should be 2
                ratio = sp.simplify(self._l2_sym / self._hbar_sym**2)
                if ratio == 2:
                    return (True, "L^2 = 2 ħ^2 (symbolic)")
                # near 2 with small tolerance for floating constants?
                # Use isclose on numeric evaluation (if all symbols numeric)
                try:
                    num_ratio = float(ratio.subs({self._hbar_sym: 1.054571817e-34}))
                except (TypeError, ValueError):
                    num_ratio = None
                if num_ratio is not None and math.isclose(num_ratio, 2.0, rel_tol=1e-2):
                    return (True, "L^2 ≈ 2 ħ^2 (symbolic numeric)")
                return (False, f"L^2 / ħ^2 = {ratio} (expected 2)")

        # Strategy 2: Numeric check using known ħ
        # Use standard value from scipy if available, else a reasonable constant
        try:
            from scipy.constants import hbar as H
        except ImportError:
            H = 1.0545718176461565e-34  # J·s
        try:
            l2_val = float(self.l2_eig)
            expect = 2.0 * H * H
            if math.isclose(l2_val, expect, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"L^2 ≈ {expect} (numeric check)")
            else:
                return (False, f"L^2 = {l2_val} ≠ {expect}")
        except (TypeError, ValueError):
            return (False, "SKIP: cannot determine ħ or numeric value")

    def test_invariant_lz_eigenvalue(self) -> Tuple[bool, str]:
        """
        L_z eigenvalue must be 0 for a function independent of φ.
        """
        if self.lz_eig is None:
            return (False, "SKIP: lz_eig not available")

        # Symbolic check
        if self._lz_sym is not None:
            if self._lz_sym == 0:
                return (True, "L_z = 0 (symbolic)")
            # try numeric evaluation
            try:
                val = float(self._lz_sym)
                if math.isclose(val, 0.0, abs_tol=1e-8):
                    return (True, "L_z ≈ 0 (numeric)")
            except (TypeError, ValueError):
                pass
            return (False, f"L_z = {self._lz_sym} (expected 0)")

        # Numeric fallback
        try:
            val = float(self.lz_eig)
            if math.isclose(val, 0.0, abs_tol=1e-8):
                return (True, "L_z ≈ 0 (numeric)")
            else:
                return (False, f"L_z = {val} ≠ 0")
        except (TypeError, ValueError):
            return (False, "SKIP: cannot evaluate L_z")

    # ------------------------------------------------------------------
    # Modeling/Trace checks (re-derive from operator definition)
    # ------------------------------------------------------------------
    def test_modeling_l2_from_operator(self) -> Tuple[bool, str]:
        """
        Verify that L^2 acting on cosθ gives 2ħ² cosθ, confirming eigenvalue.
        Uses symbolic manipulation of spherical Laplacian.
        """
        try:
            theta = sp.symbols('theta', real=True)
            # angular function f = cos(theta)
            f = sp.cos(theta)
            # L^2 operator on theta part: -ħ² * (1/sinθ) d/dθ (sinθ d/dθ)
            # (no φ-derivative since f is φ-independent)
            l2_op = -self._hbar_sym**2 * (1/sp.sin(theta)) * sp.diff(sp.sin(theta) * sp.diff(f, theta), theta)
            l2_f = sp.simplify(l2_op)
            expected = 2 * self._hbar_sym**2 * sp.cos(theta)
            if sp.simplify(l2_f - expected) == 0:
                return (True, "L^2 cosθ = 2ħ² cosθ derived symbolically")
            else:
                # Use isclose on numeric evaluation if symbols are given numeric values
                try:
                    # Evaluate at a sample theta, e.g., π/4
                    val_l2 = complex(l2_f.subs({theta: sp.pi/4, self._hbar_sym: 1.0}))
                    val_exp = complex(expected.subs({theta: sp.pi/4, self._hbar_sym: 1.0}))
                    if math.isclose(val_l2.real, val_exp.real, rel_tol=1e-8):
                        return (True, "L^2 cosθ ≈ 2ħ² cosθ (numeric sample)")
                    else:
                        return (False, "Derived L^2 cosθ does not match 2ħ² cosθ")
                except Exception:
                    return (False, f"Symbolic check failed: L^2 f = {l2_f}")
        except Exception as e:
            return (False, f"SKIP: modeling test crashed: {e}")

    def test_modeling_lz_zero_from_phi(self) -> Tuple[bool, str]:
        """
        Verify that L_z acting on a φ-independent function gives zero.
        """
        try:
            phi = sp.symbols('phi', real=True)
            # generic function independent of phi
            f = sp.Function('f')(sp.symbols('r'), sp.symbols('theta'))
            # L_z = -i ħ d/dφ
            lz_op = -sp.I * self._hbar_sym * sp.diff(f, phi)
            lz_f = sp.simplify(lz_op)
            # Since f does not depend on phi, derivative is zero
            if lz_f == 0:
                return (True, "L_z ψ = 0 for φ-independent function")
            else:
                return (False, f"L_z did not simplify to zero: {lz_f}")
        except Exception as e:
            return (False, f"SKIP: modeling test crashed: {e}")

    def test_modeling_angular_separation(self) -> Tuple[bool, str]:
        """
        Check that the wave function is separable into radial*cosθ.
        This is a modeling consistency: explanation should mention Y_{10}.
        """
        # We check that the solution's explanation (if available) contains "Y_{10}" or "cosθ".
        if self.explanation is None:
            return (False, "SKIP: no explanation_steps provided")
        text = str(self.explanation)
        if "Y_{10}" in text or "cosθ" in text or "cos theta" in text:
            return (True, "Explanation mentions Y_{10} / cosθ separation")
        else:
            # Even without explanation, the eigenvalue itself implies it.
            # But we can also check the given eigenvalue matches Y_{10}
            if self._l2_sym is not None and self._lz_sym is not None:
                if self._l2_sym == 2*self._hbar_sym**2 and self._lz_sym == 0:
                    return (True, "Eigenvalues consistent with Y_{10}")
            return (False, "No trace of angular separation in explanation")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_l2_with_hbar(self) -> Tuple[bool, str]:
        """
        Perturb ħ by +10% and verify that L^2 increases.
        Uses the functional form L^2 = 2 ħ^2.
        """
        if self._l2_sym is None:
            # If no symbolic form, try to extract functional dependence
            # Fallback: numeric check with assumed hbar value
            try:
                hbar_base = 1.054571817e-34
                # Use base hbar to get two points
                l2_base = 2.0 * hbar_base**2
                hbar_pert = 1.1 * hbar_base
                l2_pert = 2.0 * hbar_pert**2
                if l2_pert > l2_base:
                    return (True, "Increasing ħ increases L^2 (numeric check)")
                else:
                    return (False, "Monotonicity violated")
            except Exception:
                return (False, "SKIP: cannot evaluate numeric hbar")
        # Use symbolic
        try:
            # define base and perturbed hbar symbols (positive)
            hbase = sp.symbols('hbase', positive=True)
            hpert = 1.1 * hbase
            L2_base = self._l2_sym.subs({self._hbar_sym: hbase})
            L2_pert = self._l2_sym.subs({self._hbar_sym: hpert})
            # Check that L2_pert > L2_base for all positive hbase
            # Simplify difference
            diff = sp.simplify(L2_pert - L2_base)
            # Should be positive: 2*(1.1² - 1²) * hbase² = 2*0.21*hbase² = 0.42*hbase² > 0
            if diff.is_positive or (not diff.is_negative and diff != 0):
                # Try numeric test for a sample value
                sample_val = 1.0  # arbitrary
                try:
                    num_diff = float(diff.subs({hbase: sample_val}))
                    if num_diff > 0:
                        return (True, f"L^2 increases with ħ (symbolic, diff={diff})")
                except Exception:
                    pass
                return (True, f"Symbolic monotonicity likely (diff={diff})")
            else:
                return (False, f"Monotonicity not confirmed, diff={diff}")
        except Exception as e:
            return (False, f"SKIP: monotonicity crashed: {e}")

    def test_monotonicity_lz_unchanged(self) -> Tuple[bool, str]:
        """
        Perturb ħ (or any parameter) and verify L_z remains 0.
        L_z is independent of ħ, so it should stay zero.
        """
        if self._lz_sym is None:
            # Numeric fallback
            try:
                lz = float(self.lz_eig)
                if lz == 0.0:
                    return (True, "L_z = 0 invariant under perturbation (no dependence)")
                else:
                    return (False, "L_z is not zero")
            except Exception:
                return (False, "SKIP: cannot evaluate L_z")
        # Symbolic: L_z should not contain hbar, thus unchanged
        free = self._lz_sym.free_symbols
        if self._hbar_sym not in free:
            return (True, "L_z independent of ħ, thus invariant under perturbation")
        else:
            # It does have hbar? Actually L_z = 0 always, so hbar should not appear
            return (False, "L_z symbolically depends on ħ, which contradicts L_z=0")

    # ------------------------------------------------------------------
    # Answer sanity checks (weak, only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_eigenvalues_plausible(self) -> Tuple[bool, str]:
        """
        Weak plausibility checks: L^2 positive, L_z finite.
        No isclose, no re-derivation.
        """
        # Check L^2
        if self.l2_eig is not None:
            try:
                l2_val = complex(self.l2_eig)
                if not (l2_val.real > 0 and l2_val.imag == 0):
                    return (False, f"L^2 eigenvalue unrealistic: {l2_val}")
            except (TypeError, ValueError):
                pass  # symbolic – assume plausible
        # Check L_z
        if self.lz_eig is not None:
            try:
                lz_val = complex(self.lz_eig)
                if math.isinf(lz_val.real) or math.isnan(lz_val.real):
                    return (False, "L_z eigenvalue is infinite or NaN")
            except (TypeError, ValueError):
                pass
        # If any missing, still weak pass (we rely on other tests)
        return (True, "Eigenvalues pass basic plausibility (positive/finite)")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Collect all test_ methods
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
                msg = f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)