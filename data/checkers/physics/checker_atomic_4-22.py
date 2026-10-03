import math
import sympy as sp
from typing import Tuple, List, Any, Optional, Dict

class PhysicsChecker:
    """Audits a solution for the fractional change in kinetic energy of an alpha particle
    scattered 180° by an O-16 nucleus (elastic head-on collision)."""

    def __init__(self, solution_instance: Any):
        # Extract the required result 'fract_change' (the numeric fractional change)
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.fract_change = results.get('fract_change', None)

        # Extract givens (masses) if available; assume they are in a dict 'givens' or 'inputs',
        # or as direct attributes like m_alpha, m_O.
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        inputs = getattr(solution_instance, 'inputs', {})
        if inputs is None:
            inputs = {}

        # Try to get m_alpha and m_O from givens, then inputs, then as top-level attributes
        self.m_alpha = givens.get('m_alpha', None) or inputs.get('m_alpha', None)
        if self.m_alpha is None:
            self.m_alpha = getattr(solution_instance, 'm_alpha', None)

        self.m_O = givens.get('m_O', None) or inputs.get('m_O', None)
        if self.m_O is None:
            self.m_O = getattr(solution_instance, 'm_O', None)

        # Convert any symbolic or string masses to float, if possible
        if self.m_alpha is not None and not isinstance(self.m_alpha, (int, float)):
            try:
                self.m_alpha = float(sp.N(self.m_alpha))
            except Exception:
                pass
        if self.m_O is not None and not isinstance(self.m_O, (int, float)):
            try:
                self.m_O = float(sp.N(self.m_O))
            except Exception:
                pass

        # Explanation steps for guidance (not hardcoded, but can be used for reasoning)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests: existence and type checks
    # ----------------------------------------------------------------------
    def test_interface_fract_change_present(self) -> Tuple[bool, str]:
        """Check that the result key 'fract_change' exists."""
        if self.fract_change is None:
            return False, "Missing required key 'fract_change' in results."
        return True, "Key 'fract_change' is present."

    def test_interface_fract_change_numeric(self) -> Tuple[bool, str]:
        """Check that fract_change is a numeric value (int/float)."""
        if self.fract_change is None:
            return False, "fract_change is None; test skipped."
        if not isinstance(self.fract_change, (int, float)):
            return False, f"fract_change is not numeric; got {type(self.fract_change)}."
        return True, f"fract_change is numeric: {self.fract_change}"

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks: physical sanity
    # ----------------------------------------------------------------------
    def test_invariant_fractional_change_negative(self) -> Tuple[bool, str]:
        """In an elastic head-on collision with a heavier target, alpha loses kinetic energy;
        therefore fractional change must be negative."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot check sign."
        if not isinstance(self.fract_change, (int, float)):
            return False, "fract_change not numeric; cannot check sign."
        if self.fract_change >= 0:
            return False, f"Expected negative fractional change but got {self.fract_change}."
        return True, f"Fractional change is negative ({self.fract_change})."

    def test_invariant_fractional_change_range(self) -> Tuple[bool, str]:
        """The fractional change must be between -1 and 0 (energy can't change sign or exceed initial)."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot check range."
        if not isinstance(self.fract_change, (int, float)):
            return False, "fract_change not numeric; cannot check range."
        if not (-1.0 <= self.fract_change <= 0.0):
            return False, f"Fractional change {self.fract_change} is not in [-1, 0]."
        return True, f"Fractional change within [-1, 0]."

    # ----------------------------------------------------------------------
    # Modeling checks: verify that fract_change follows from conservation laws
    # ----------------------------------------------------------------------
    def test_modeling_fract_change_formula(self) -> Tuple[bool, str]:
        """If masses are available, compute the theoretical fractional change from
        elastic head-on formula: -4 m_alpha m_O / (m_alpha + m_O)^2 and compare using isclose."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot compare."
        if self.m_alpha is None or self.m_O is None:
            return False, "Skipped: masses m_alpha and m_O not available."
        if not isinstance(self.m_alpha, (int, float)) or not isinstance(self.m_O, (int, float)):
            return False, "Masses not numeric; cannot compute formula."

        ma = self.m_alpha
        mo = self.m_O
        expected = -4.0 * ma * mo / ((ma + mo) ** 2)
        if not math.isclose(self.fract_change, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"fract_change {self.fract_change} differs from expected {expected} (m_alpha={ma}, m_O={mo})."
        return True, f"fract_change matches theoretical formula: {self.fract_change} ≈ {expected}."

    def test_modeling_fract_change_symbolic_derivation(self) -> Tuple[bool, str]:
        """Use SymPy to derive the fractional change from momentum and energy conservation
        and verify that it equals the given fract_change (if masses are known)."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot compare symbolic derivation."
        if self.m_alpha is None or self.m_O is None:
            return False, "Skipped: masses not available."
        try:
            ma = sp.symbols('ma', positive=True)
            mo = sp.symbols('mo', positive=True)
            v0 = sp.symbols('v0', positive=True)
            v1, v2 = sp.symbols('v1 v2', real=True)

            # Conservation of momentum: ma*v0 = ma*v1 + mo*v2
            eq_mom = sp.Eq(ma * v0, ma * v1 + mo * v2)
            # Conservation of KE: (1/2)ma*v0^2 = (1/2)ma*v1^2 + (1/2)mo*v2^2
            eq_ke = sp.Eq(ma * v0 ** 2, ma * v1 ** 2 + mo * v2 ** 2)

            sol = sp.solve([eq_mom, eq_ke], (v1, v2), dict=True)
            if not sol:
                return False, "Could not solve conservation equations symbolically."

            # There are two solutions: (v1, v2) = (v0, 0) and the nontrivial elastic solution
            # We take the nontrivial one (v1 != v0)
            nontrivial = None
            for s in sol:
                if s[v1].simplify() != v0:
                    nontrivial = s
                    break
            if nontrivial is None:
                return False, "Could not isolate nontrivial solution."

            # Fractional change in KE of alpha: (v1^2 - v0^2)/v0^2
            v1_expr = nontrivial[v1]
            fractional_change_expr = (v1_expr ** 2 - v0 ** 2) / v0 ** 2
            fractional_change_simplified = sp.simplify(fractional_change_expr)

            # Substitute the actual masses
            ma_val = float(self.m_alpha)
            mo_val = float(self.m_O)
            fractional_change_derived = fractional_change_simplified.subs({ma: ma_val, mo: mo_val})
            derived_float = float(sp.N(fractional_change_derived))

            if not math.isclose(self.fract_change, derived_float, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"Derived fractional change {derived_float} differs from given {self.fract_change}."
            return True, f"Symbolic derivation gives {derived_float} ≈ {self.fract_change}."
        except Exception as e:
            return False, f"Symbolic derivation failed: {e}"

    # ----------------------------------------------------------------------
    # Monotonicity test: if target mass increases, fractional change should become less negative
    # (i.e., closer to zero) for fixed alpha mass.
    # ----------------------------------------------------------------------
    def test_monotonicity_increasing_O_mass_less_negative(self) -> Tuple[bool, str]:
        """Perturb m_O upward by 10% and check that fractional change becomes less negative
        (i.e., closer to zero) compared to the baseline."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot perform monotonicity test."
        if self.m_alpha is None or self.m_O is None:
            return False, "Skipped: masses not available."
        if not isinstance(self.m_alpha, (int, float)) or not isinstance(self.m_O, (int, float)):
            return False, "Masses not numeric; cannot perturb."

        ma = self.m_alpha
        mo = self.m_O
        # Baseline fractional change (should match self.fract_change)
        baseline = -4.0 * ma * mo / ((ma + mo) ** 2)
        if not math.isclose(self.fract_change, baseline, rel_tol=1e-9, abs_tol=1e-9):
            # If baseline differs, still use the self.fract_change as baseline? No,
            # we need a consistent reference. Use the computed baseline.
            return False, f"Baseline computed {baseline} does not match fract_change {self.fract_change}; cannot reliably test monotonicity."

        # Perturb m_O upward by 10%
        mo_perturbed = mo * 1.1
        perturbed_f = -4.0 * ma * mo_perturbed / ((ma + mo_perturbed) ** 2)

        # Expect: for fixed ma, as mo increases, |f| decreases? Let's check: derivative
        # f = -4ma mo/(ma+mo)^2; derivative wrt mo >0? Actually for mo > ma, derivative is positive,
        # meaning f becomes less negative. So perturbed_f > baseline_f.
        if perturbed_f <= baseline:
            return False, f"Monotonicity violated: perturbed fraction {perturbed_f} is not greater than baseline {baseline} (expected less negative with larger target mass)."
        return True, f"Monotonicity holds: O mass increase makes fraction less negative ({baseline} -> {perturbed_f})."

    # ----------------------------------------------------------------------
    # Optional weak answer sanity (do not re-derive or use exact match)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak_magnitude(self) -> Tuple[bool, str]:
        """Weak sanity: the fractional change should be a negative number not exceeding -1 in magnitude,
        and for the standard masses (4 and 16) it should be close to -0.64, but we only check order-of-magnitude."""
        if self.fract_change is None:
            return False, "fract_change is None; cannot check sanity."
        if not isinstance(self.fract_change, (int, float)):
            return False, "fract_change not numeric."
        # Check that it's negative and roughly in the expected range (-1 to 0)
        if not (-1.0 < self.fract_change < 0.0):
            return False, f"Fractional change {self.fract_change} not in plausible range (-1, 0)."
        # Additional rough: for standard masses it should be around -0.6 to -0.7
        # but we only do a coarse check (not using exact value)
        if self.fract_change < -0.8 or self.fract_change > -0.1:
            return False, f"Fractional change {self.fract_change} seems far from typical -0.64 range (but this is only a weak check)."
        return True, f"Fractional change {self.fract_change} passes weak magnitude sanity."

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs