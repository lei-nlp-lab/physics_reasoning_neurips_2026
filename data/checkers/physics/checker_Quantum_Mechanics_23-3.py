import math
import sympy
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Robust physics checker for the Ramsauer effect problem (quantum finite square well).
    Checks transmission/reflection formulas, probability conservation, resonance condition,
    and the specific well depth calculation.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract required results safely; all defaults are None
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict):
            self.results = solution_instance
        # else results kept as empty dict

        self.T_formula = self.results.get('T_formula', None)
        self.R_formula = self.results.get('R_formula', None)
        self.E_n = self.results.get('E_n', None)
        self.V0_J = self.results.get('V0_J', None)
        self.V0_eV = self.results.get('V0_eV', None)

        # Try to extract givens (constants for part c) if they exist
        self.givens = {}
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.givens = solution_instance.givens
        elif 'givens' in self.results and isinstance(self.results['givens'], dict):
            self.givens = self.results['givens']

    # ------------------------------------------------------------------
    # Interface tests: check required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_T_formula_exists(self) -> Tuple[bool, str]:
        if self.T_formula is None:
            return (False, "Missing key: T_formula")
        return (True, "T_formula present")

    def test_interface_R_formula_exists(self) -> Tuple[bool, str]:
        if self.R_formula is None:
            return (False, "Missing key: R_formula")
        return (True, "R_formula present")

    def test_interface_E_n_exists(self) -> Tuple[bool, str]:
        if self.E_n is None:
            return (False, "Missing key: E_n")
        return (True, "E_n formula present")

    def test_interface_V0_J_exists(self) -> Tuple[bool, str]:
        if self.V0_J is None:
            return (False, "Missing key: V0_J")
        return (True, "V0_J present")

    def test_interface_V0_eV_exists(self) -> Tuple[bool, str]:
        if self.V0_eV is None:
            return (False, "Missing key: V0_eV")
        return (True, "V0_eV present")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_R_plus_T_equals_one_numerical(self) -> Tuple[bool, str]:
        """
        Check numerically that for some representative parameters we get R+T ≈ 1.
        Uses the symbolic formulas if possible.
        """
        if self.T_formula is None or self.R_formula is None:
            return (False, "SKIP: T_formula or R_formula missing")

        # Try to parse the formulas as sympy expressions
        try:
            T_sym = sympy.sympify(self.T_formula)
            R_sym = sympy.sympify(self.R_formula)
        except Exception:
            return (False, "Cannot parse T_formula or R_formula as sympy expressions")

        # Define symbolic variables k, q, a
        k, q, a = sympy.symbols('k q a', real=True, positive=True)
        # Try to substitute test values
        try:
            T_num = float(T_sym.subs({k: 2.0, q: 3.0, a: 1.5}).evalf())
            R_num = float(R_sym.subs({k: 2.0, q: 3.0, a: 1.5}).evalf())
            total = T_num + R_num
            if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"R+T ≈ 1 (got {total})")
            else:
                return (False, f"R+T = {total}, not 1")
        except Exception:
            return (False, "Cannot evaluate formula numerically")

    def test_invariant_probabilities_between_0_and_1(self) -> Tuple[bool, str]:
        """
        For a few test points, ensure T and R are in [0,1].
        """
        if self.T_formula is None or self.R_formula is None:
            return (False, "SKIP: T_formula or R_formula missing")
        try:
            T_sym = sympy.sympify(self.T_formula)
            R_sym = sympy.sympify(self.R_formula)
        except Exception:
            return (False, "Cannot parse formulas")

        k, q, a = sympy.symbols('k q a', real=True, positive=True)
        test_vals = [(1.0, 0.5, 0.2), (0.5, 2.5, 1.0), (3.0, 3.0, 0.5)]
        for k_val, q_val, a_val in test_vals:
            try:
                T_num = float(T_sym.subs({k: k_val, q: q_val, a: a_val}).evalf())
                R_num = float(R_sym.subs({k: k_val, q: q_val, a: a_val}).evalf())
            except Exception:
                continue
            if T_num < 0 or T_num > 1:
                return (False, f"T={T_num} not in [0,1] for (k={k_val}, q={q_val}, a={a_val})")
            if R_num < 0 or R_num > 1:
                return (False, f"R={R_num} not in [0,1] for (k={k_val}, q={q_val}, a={a_val})")
        return (True, "Probabilities in [0,1] for tested points")

    def test_invariant_resonance_condition_T_equals_one(self) -> Tuple[bool, str]:
        """
        Check that T=1 when sin(2qa)=0 (i.e., 2qa = n*pi).
        """
        if self.T_formula is None:
            return (False, "SKIP: T_formula missing")
        try:
            T_sym = sympy.sympify(self.T_formula)
        except Exception:
            return (False, "Cannot parse T_formula")

        k, q, a = sympy.symbols('k q a', real=True, positive=True)
        # Substitute sin(2qa)=0 -> 2qa = pi (n=1)
        # We need a specific numeric scenario: set k=1, choose a=1, then q=pi/2 so sin(pi)=0
        try:
            T_res = float(T_sym.subs({k: 1.0, q: math.pi/2, a: 1.0}).evalf())
            if math.isclose(T_res, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"T ≈ 1 at resonance (got {T_res})")
            else:
                return (False, f"T = {T_res} at resonance (expected ~1)")
        except Exception:
            return (False, "Cannot evaluate resonance test")

    # ------------------------------------------------------------------
    # Modeling/trace tests: check the form and correctness of emitted quantities
    # ------------------------------------------------------------------
    def test_modeling_T_formula_structure(self) -> Tuple[bool, str]:
        """
        Check that T_formula matches the expected analytic form:
        T = 4k^2q^2 / (4k^2q^2 + (k^2 - q^2)^2 sin^2(2qa))
        We compare numerically for a few random values.
        """
        if self.T_formula is None:
            return (False, "SKIP: T_formula missing")
        try:
            T_sym = sympy.sympify(self.T_formula)
        except Exception:
            return (False, "Cannot parse T_formula")

        # Build reference expression
        k, q, a = sympy.symbols('k q a', real=True, positive=True)
        ref_T = 4*k**2*q**2 / (4*k**2*q**2 + (k**2 - q**2)**2 * (sympy.sin(2*q*a))**2)

        # Compare for a few numeric triples
        import random
        for _ in range(10):
            k_val = random.uniform(0.5, 5.0)
            q_val = random.uniform(0.5, 5.0)
            a_val = random.uniform(0.1, 2.0)
            try:
                val_T = float(T_sym.subs({k: k_val, q: q_val, a: a_val}).evalf())
                val_ref = float(ref_T.subs({k: k_val, q: q_val, a: a_val}).evalf())
                if not math.isclose(val_T, val_ref, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"T deviates at (k={k_val}, q={q_val}, a={a_val}): T={val_T}, ref={val_ref}")
            except Exception:
                return (False, f"Cannot evaluate at test point")
        return (True, "T_formula matches expected analytic form")

    def test_modeling_E_n_structure(self) -> Tuple[bool, str]:
        """
        Check that E_n formula is of the form n^2 π^2 ħ^2/(8 m a^2) - V0
        We evaluate numerically with standard constants.
        """
        if self.E_n is None:
            return (False, "SKIP: E_n missing")
        try:
            E_sym = sympy.sympify(self.E_n)
        except Exception:
            return (False, "Cannot parse E_n formula")

        n, a, V0 = sympy.symbols('n a V0', real=True, positive=True)
        try:
            # Use standard constants
            hbar = 1.054571817e-34
            m_e = 9.10938356e-31
            # Substitute with specific values
            value = float(E_sym.subs({n: 1, a: 1e-10, V0: 1e-19,
                                      sympy.Symbol('hbar'): hbar,
                                      sympy.Symbol('m'): m_e,
                                      sympy.pi: math.pi}).evalf())
            expected = (math.pi**2 * hbar**2) / (8 * m_e * (1e-10)**2) - 1e-19
            if math.isclose(value, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "E_n formula numerically consistent")
            else:
                return (False, f"E_n evaluation gives {value}, expected {expected}")
        except Exception:
            return (False, "Cannot evaluate E_n with standard constants")

    def test_modeling_V0_computation(self) -> Tuple[bool, str]:
        """
        Re-derive V0 from given parameters (part c) and compare with emitted V0_eV.
        Use the resonance condition: V0 = (π²ħ²)/(8 m a²) - E
        """
        # Expect standard constants
        hbar = 1.054571817e-34
        m_e = 9.10938356e-31
        eV_J = 1.602176634e-19

        # Get given values from givens or fallback to problem constants
        E_eV = self.givens.get('E', 0.7)       # incident energy in eV
        a_nm = self.givens.get('a', 0.1)       # well half-width in nm
        # Convert to SI
        E_J = E_eV * eV_J
        a_m = a_nm * 1e-9

        # Compute V0 in Joules
        V0_J_expected = (math.pi**2 * hbar**2) / (8 * m_e * a_m**2) - E_J
        V0_eV_expected = V0_J_expected / eV_J

        if self.V0_J is not None:
            if not math.isclose(self.V0_J, V0_J_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"V0_J mismatch: got {self.V0_J}, expected ~{V0_J_expected}")
        if self.V0_eV is not None:
            if not math.isclose(self.V0_eV, V0_eV_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"V0_eV mismatch: got {self.V0_eV}, expected ~{V0_eV_expected}")
        if self.V0_J is None and self.V0_eV is None:
            return (False, "SKIP: neither V0_J nor V0_eV present")
        return (True, f"V0 consistent with E={E_eV} eV, a={a_nm} nm")

    def test_modeling_R_formula_from_T(self) -> Tuple[bool, str]:
        """
        Check that R_formula is 1 - T_formula (non-negative).
        """
        if self.T_formula is None or self.R_formula is None:
            return (False, "SKIP: T_formula or R_formula missing")
        try:
            T_sym = sympy.sympify(self.T_formula)
            R_sym = sympy.sympify(self.R_formula)
        except Exception:
            return (False, "Cannot parse formulas")

        # Check if R_sym is exactly 1 - T_sym
        try:
            diff = sympy.simplify(R_sym - (1 - T_sym))
            # If diff is identically 0, the expressions are equivalent
            if diff == 0:
                return (True, "R = 1 - T identically")
            else:
                # Otherwise check numerically
                k, q, a = sympy.symbols('k q a', real=True, positive=True)
                for _ in range(5):
                    import random
                    k_val = random.uniform(0.5, 5.0)
                    q_val = random.uniform(0.5, 5.0)
                    a_val = random.uniform(0.1, 2.0)
                    val_diff = float(diff.subs({k:k_val, q:q_val, a:a_val}).evalf())
                    if not math.isclose(val_diff, 0, abs_tol=1e-12):
                        return (False, f"R + T not 1 numerically at (k={k_val}, q={q_val}, a={a_val})")
                return (True, "R ≈ 1 - T numerically")
        except Exception:
            return (False, "Error checking R = 1 - T")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_T_decreases_off_resonance(self) -> Tuple[bool, str]:
        """
        Check that moving away from a resonance (by increasing a slightly) reduces T.
        Use a fixed k, q and baseline a that gives resonance.
        """
        if self.T_formula is None:
            return (False, "SKIP: T_formula missing")
        try:
            T_sym = sympy.sympify(self.T_formula)
        except Exception:
            return (False, "Cannot parse T_formula")

        # Use parameters: k=1, q such that 2qa = pi -> a = pi/(2q). Choose q=1, then a=pi/2
        k_val = 1.0
        q_val = 1.0
        a_res = math.pi / 2
        # Evaluate T at a_res and at a slightly larger a
        try:
            T_res = float(T_sym.subs({sympy.Symbol('k'): k_val, sympy.Symbol('q'): q_val, sympy.Symbol('a'): a_res}).evalf())
            T_pert = float(T_sym.subs({sympy.Symbol('k'): k_val, sympy.Symbol('q'): q_val, sympy.Symbol('a'): a_res*1.1}).evalf())
        except Exception:
            return (False, "Cannot evaluate T for test parameters")

        if T_pert < T_res:
            return (True, f"T decreased when a increased from {a_res:.2f} to {a_res*1.1:.2f}")
        else:
            return (False, f"T did not decrease: at resonance T={T_res}, perturbed T={T_pert}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one allowed, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_V0_positive(self) -> Tuple[bool, str]:
        """
        Weak sanity: V0 is positive and finite.
        """
        if self.V0_eV is not None:
            if self.V0_eV > 0 and not math.isnan(self.V0_eV):
                return (True, f"V0_eV = {self.V0_eV} is positive finite")
            else:
                return (False, "V0_eV is not positive or not finite")
        if self.V0_J is not None:
            if self.V0_J > 0:
                return (True, f"V0_J = {self.V0_J} is positive")
            else:
                return (False, "V0_J is not positive")
        return (False, "SKIP: No V0 value")

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)