import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    """
    Audits solutions to the nuclear stability problem.
    Checks consistency with the semi-empirical mass formula,
    equilibrium Z, separation energy difference, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract all relevant attributes with None defaults
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None) or {}
        self.key_constraints = getattr(solution_instance, 'key_constraints', None) or []

        # Extract expected keys from results
        self.delta_S = self.results.get('delta_S', None)
        self.Z = self.results.get('Z', None)
        self.S_p = self.results.get('S_p', None)
        self.S_n = self.results.get('S_n', None)
        self.B = self.results.get('B', None)
        self.N = self.results.get('N', None)

        # Extract givens (coefficients and mass number)
        self.a_c = self.givens.get('a_c', None)
        self.a_a = self.givens.get('a_a', None)
        self.A = self.givens.get('A', None)

        # Additional parameters from physics mapping if available
        self.a_v = self.physics_mapping.get('a_v', None)
        self.a_s = self.physics_mapping.get('a_s', None)

        # Fallback: if givens missing, try to get from physics_mapping
        if self.a_c is None and 'a_c' in self.physics_mapping:
            self.a_c = self.physics_mapping['a_c']
        if self.a_a is None and 'a_a' in self.physics_mapping:
            self.a_a = self.physics_mapping['a_a']

    # ---------- Interface tests ----------
    def test_interface_delta_S_exists(self) -> Tuple[bool, str]:
        """Check that delta_S exists and is a number."""
        if self.delta_S is None:
            return (False, "SKIP: 'delta_S' not found in results")
        try:
            float(self.delta_S)
            return (True, f"delta_S = {self.delta_S} is numeric")
        except (TypeError, ValueError):
            return (False, f"delta_S is not numeric: {self.delta_S}")

    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that critical givens are present."""
        missing = []
        for key in ['a_c', 'a_a', 'A']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens (a_c, a_a, A) present")

    # ---------- Invariant / theorem-based tests ----------
    def test_invariant_equilibrium_Z(self) -> Tuple[bool, str]:
        """Using equilibrium condition from SEMF: Z = A / (2 + (a_c/(2*a_a)) * A^{2/3})"""
        if self.Z is None or self.A is None or self.a_c is None or self.a_a is None:
            return (False, "SKIP: missing Z, A, a_c, or a_a")
        try:
            A, a_c, a_a = float(self.A), float(self.a_c), float(self.a_a)
            Z_eq = A / (2 + (a_c / (2 * a_a)) * (A ** (2/3)))
            Z_given = float(self.Z)
            if math.isclose(Z_given, Z_eq, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Z={Z_given} matches equilibrium Z={Z_eq:.3f}")
            else:
                return (False, f"Z={Z_given} differs from equilibrium Z={Z_eq:.3f}")
        except Exception as e:
            return (False, f"Error in equilibrium Z check: {e}")

    def test_invariant_separation_energy_diff_formula(self) -> Tuple[bool, str]:
        """
        Check that if both S_p and S_n are provided, delta_S = S_p - S_n
        follows the theoretical expression derived from SEMF:
          delta_S ≈ -a_c*(2Z-1)/A^{1/3} + 8*a_a/A
        """
        if None in (self.S_p, self.S_n, self.Z, self.A, self.a_c, self.a_a):
            return (False, "SKIP: missing S_p, S_n, Z, A, a_c, or a_a")
        try:
            S_p = float(self.S_p)
            S_n = float(self.S_n)
            Z = float(self.Z)
            A = float(self.A)
            a_c = float(self.a_c)
            a_a = float(self.a_a)
            delta_given = S_p - S_n

            delta_theory = -a_c * (2*Z - 1) / (A ** (1/3)) + 8 * a_a / A
            if math.isclose(delta_given, delta_theory, rel_tol=0.1, abs_tol=1.0):
                return (True, f"delta_S = {delta_given:.2f} close to theory {delta_theory:.2f}")
            else:
                return (False, f"delta_S = {delta_given:.2f} not close to theory {delta_theory:.2f}")
        except Exception as e:
            return (False, f"Error in separation energy diff check: {e}")

    # ---------- Modeling / trace tests (at least two) ----------
    def test_trace_delta_S_scaling(self) -> Tuple[bool, str]:
        """
        Modeling check: verify that the leading term of delta_S scales as A^{4/3}
        and is positive. This test re-derives the leading term from the equilibrium
        condition and checks its functional form without referencing a specific constant.
        """
        if self.a_c is None or self.a_a is None or self.A is None:
            return (False, "SKIP: missing a_c, a_a, or A")
        try:
            a_c, a_a, A = sp.symbols('a_c a_a A', positive=True)
            Z_expr = A / (2 + (a_c/(2*a_a)) * A**(sp.Rational(2,3)))
            delta_expr = -a_c * (2*Z_expr - 1) / (A - 1)**(sp.Rational(1,3)) + a_a * (A - 2*Z_expr) / (A - 1)
            delta_series = sp.series(delta_expr, A, sp.oo, 3).removeO()
            delta_leading = sp.simplify(delta_series)

            # Evaluate for two different A values to check scaling
            a_c_val = float(self.a_c)
            a_a_val = float(self.a_a)
            A1 = float(self.A)
            A2 = A1 * 1.5  # 50% increase
            val1 = float(delta_leading.subs({a_c: a_c_val, a_a: a_a_val, A: A1}))
            val2 = float(delta_leading.subs({a_c: a_c_val, a_a: a_a_val, A: A2}))

            # Check positivity
            if val1 <= 0 or val2 <= 0:
                return (False, f"Leading term not positive: {val1}, {val2}")

            # Check scaling: for A^{4/3}, ratio should be (A2/A1)^{4/3}
            expected_ratio = (A2 / A1) ** (4/3)
            actual_ratio = val2 / val1
            if math.isclose(actual_ratio, expected_ratio, rel_tol=0.05):
                return (True, f"delta_S leading term scales as A^{4/3}: ratio {actual_ratio:.3f} vs expected {expected_ratio:.3f}")
            else:
                return (False, f"Scaling does not match A^{4/3}: ratio {actual_ratio:.3f} vs expected {expected_ratio:.3f}")
        except Exception as e:
            return (False, f"Symbolic trace test failed: {e}")

    def test_trace_equilibrium_Z_formula(self) -> Tuple[bool, str]:
        """
        Modeling check: re-derive Z from the equilibrium condition using
        given coefficients and compare with the provided Z (if available).
        """
        if self.A is None or self.a_c is None or self.a_a is None:
            return (False, "SKIP: missing A, a_c, or a_a")
        try:
            A = float(self.A)
            a_c = float(self.a_c)
            a_a = float(self.a_a)
            Z_eq = A / (2 + (a_c / (2 * a_a)) * (A ** (2/3)))
            if self.Z is not None:
                Z_given = float(self.Z)
                if math.isclose(Z_given, Z_eq, rel_tol=0.05, abs_tol=0.5):
                    return (True, f"Given Z={Z_given} is close to re-derived Z_eq={Z_eq:.3f}")
                else:
                    return (False, f"Given Z={Z_given} differs from re-derived Z_eq={Z_eq:.3f}")
            else:
                return (True, f"Z not given; re-derived Z_eq = {Z_eq:.3f} (modeling reference)")
        except Exception as e:
            return (False, f"Error in equilibrium Z re-derivation: {e}")

    # ---------- Monotonicity tests (each must perturb an input and compare outputs) ----------
    def test_monotonicity_delta_S_increases_with_A(self) -> Tuple[bool, str]:
        """
        Perturb A by +10% and check that the leading term of delta_S increases.
        This tests that the separation energy difference grows with mass number.
        """
        if self.a_c is None or self.a_a is None or self.A is None:
            return (False, "SKIP: missing a_c, a_a, or A")
        try:
            a_c = float(self.a_c)
            a_a = float(self.a_a)
            A_base = float(self.A)

            # Baseline output: leading term of delta_S for base A
            delta_base = (a_c**2) / (4 * a_a) * (A_base ** (4/3))

            # Perturb input: increase A by 10%
            A_pert = A_base * 1.1
            delta_pert = (a_c**2) / (4 * a_a) * (A_pert ** (4/3))

            if delta_pert > delta_base:
                return (True, f"delta_S increases: {delta_base:.4f} -> {delta_pert:.4f} for A {A_base} -> {A_pert}")
            else:
                return (False, f"Monotonicity failed: delta_S does not increase with A")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_neutron_excess_increases_with_A(self) -> Tuple[bool, str]:
        """
        Perturb A by +10% and check that the neutron excess (N-Z) derived from
        equilibrium Z increases. This tests that heavier nuclei have more neutrons.
        """
        if self.a_c is None or self.a_a is None or self.A is None:
            return (False, "SKIP: missing a_c, a_a, or A")
        try:
            a_c = float(self.a_c)
            a_a = float(self.a_a)
            A_base = float(self.A)

            # Baseline: compute equilibrium Z and N-Z for base A
            Z_base = A_base / (2 + (a_c / (2 * a_a)) * (A_base ** (2/3)))
            N_minus_Z_base = A_base - 2 * Z_base

            # Perturb input: increase A by 10%
            A_pert = A_base * 1.1
            Z_pert = A_pert / (2 + (a_c / (2 * a_a)) * (A_pert ** (2/3)))
            N_minus_Z_pert = A_pert - 2 * Z_pert

            if N_minus_Z_pert > N_minus_Z_base:
                return (True, f"Neutron excess increases: {N_minus_Z_base:.2f} -> {N_minus_Z_pert:.2f} for A {A_base:.0f} -> {A_pert:.0f}")
            else:
                return (False, f"Neutron excess does not increase: {N_minus_Z_base:.2f} -> {N_minus_Z_pert:.2f}")
        except Exception as e:
            return (False, f"Monotonicity neutron excess error: {e}")

    # ---------- Weak answer sanity test (use only one) ----------
    def test_answer_sanity_delta_S_sign(self) -> Tuple[bool, str]:
        """Check that delta_S is positive (should be for heavy nuclei)."""
        if self.delta_S is None:
            return (False, "SKIP: delta_S missing")
        try:
            ds = float(self.delta_S)
            if ds > 0:
                return (True, f"delta_S = {ds} > 0")
            else:
                return (False, f"delta_S = {ds} <= 0, unexpected")
        except (TypeError, ValueError):
            return (False, f"delta_S not numeric: {self.delta_S}")

    # ---------- Run all tests ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test methods and execute them, collecting logs."""
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