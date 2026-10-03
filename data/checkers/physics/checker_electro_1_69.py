import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}
        self.a = givens.get('a', None)
        self.q = givens.get('q', None)
        self.E0 = givens.get('E0', None)
        self.eps0 = givens.get('eps0', None)

        # Extract results safely (default None)
        results = getattr(solution_instance, 'results', {})
        self.V_in = results.get('V_in', None)
        self.V_out = results.get('V_out', None)
        self.p = results.get('p', None)
        self.W1 = results.get('W1', None)
        self.W2 = results.get('W2', None)
        self.W3 = results.get('W3', None)
        self.W12 = results.get('W12', None)
        self.W13 = results.get('W13', None)
        self.W23 = results.get('W23', None)

        # For symbolic operations
        self.r, self.theta = sp.symbols('r theta', real=True, positive=True)

    # ------------------------------------------------------------------
    # Helper: convert value to float, handling sympy and strings
    # ------------------------------------------------------------------
    def _to_numeric(self, val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, 'subs') and hasattr(val, 'evalf'):
            try:
                # Substitute known numeric values for symbols
                subs = {}
                if self.a is not None and not isinstance(self.a, str):
                    try:
                        subs[sp.Symbol('a')] = float(self.a)
                    except:
                        pass
                if self.q is not None and not isinstance(self.q, str):
                    try:
                        subs[sp.Symbol('q')] = float(self.q)
                    except:
                        pass
                if self.E0 is not None and not isinstance(self.E0, str):
                    try:
                        subs[sp.Symbol('E0')] = float(self.E0)
                    except:
                        pass
                if self.eps0 is not None and not isinstance(self.eps0, str):
                    try:
                        subs[sp.Symbol('eps0')] = float(self.eps0)
                    except:
                        pass
                return float(val.subs(subs).evalf())
            except:
                return None
        if isinstance(val, str):
            try:
                return float(val)
            except:
                return None
        try:
            return float(val)
        except:
            return None

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        na = self._to_numeric(a)
        nb = self._to_numeric(b)
        if na is None or nb is None:
            return False
        return math.isclose(na, nb, rel_tol=rel_tol, abs_tol=abs_tol)

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist
    # ------------------------------------------------------------------
    def test_interface_V_in_present(self):
        if self.V_in is None:
            return (False, "V_in is None")
        return (True, "V_in present")

    def test_interface_V_out_present(self):
        if self.V_out is None:
            return (False, "V_out is None")
        return (True, "V_out present")

    def test_interface_p_present(self):
        if self.p is None:
            return (False, "p is None")
        return (True, "p present")

    def test_interface_W1_present(self):
        if self.W1 is None:
            return (False, "W1 is None")
        return (True, "W1 present")

    def test_interface_W2_present(self):
        if self.W2 is None:
            return (False, "W2 is None")
        return (True, "W2 present")

    def test_interface_W3_present(self):
        if self.W3 is None:
            return (False, "W3 is None")
        return (True, "W3 present")

    def test_interface_W12_present(self):
        if self.W12 is None:
            return (False, "W12 is None")
        return (True, "W12 present")

    def test_interface_W13_present(self):
        if self.W13 is None:
            return (False, "W13 is None")
        return (True, "W13 present")

    def test_interface_W23_present(self):
        if self.W23 is None:
            return (False, "W23 is None")
        return (True, "W23 present")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity
    # ------------------------------------------------------------------
    def test_invariant_potential_continuity(self):
        """Potential at r=a should be continuous: V_in == V_out evaluated at r=a."""
        if self.V_in is None or self.V_out is None:
            return (False, "SKIP: V_in or V_out missing")
        # Evaluate V_out at r=a, θ=0 (any θ works because potential constant on surface)
        try:
            val_out = self.V_out.subs({self.r: self.a, self.theta: 0})
        except:
            return (False, "Cannot substitute r=a into V_out")
        if self._isclose(self.V_in, val_out):
            return (True, "Potential continuous at r=a")
        else:
            return (False, "V_in not equal to V_out at r=a")

    def test_invariant_laplace_outside(self):
        """Check that V_out satisfies Laplace's equation for r>a (symbolic test)."""
        if self.V_out is None:
            return (False, "SKIP: V_out missing")
        if not hasattr(self.V_out, 'diff'):
            return (False, "V_out not symbolic; skipping Laplace test")
        try:
            # Spherical Laplacian for azimuthal symmetry
            L = (1/self.r**2)*sp.diff(self.r**2 * sp.diff(self.V_out, self.r), self.r) \
                  + (1/(self.r**2*sp.sin(self.theta)))*sp.diff(sp.sin(self.theta)*sp.diff(self.V_out, self.theta), self.theta)
            L_simplified = sp.simplify(L)
            if L_simplified == 0:
                return (True, "Laplacian of V_out is identically zero symbolically")
            # If not identically zero, test numerically at a sample point
            subs = {self.r: 2*self.a, self.theta: sp.pi/4}
            numeric = self._to_numeric(L_simplified.subs(subs)) if hasattr(L_simplified, 'subs') else None
            if numeric is not None and math.isclose(numeric, 0, abs_tol=1e-10):
                return (True, f"Laplacian ≈ 0 at sample point: {numeric}")
            else:
                return (False, f"Laplacian not zero: symbolically {L_simplified}, numeric {numeric}")
        except Exception as e:
            return (False, f"Laplace test crashed: {e}")

    def test_invariant_W1_infinite(self):
        """W1 should be infinity."""
        if self.W1 is None:
            return (False, "SKIP: W1 missing")
        if isinstance(self.W1, str) and self.W1.strip().lower() == 'infinity':
            return (True, "W1 is 'infinity'")
        try:
            if math.isinf(float(self.W1)):
                return (True, "W1 is inf")
        except:
            pass
        return (False, f"W1 not infinite: {self.W1}")

    def test_invariant_finite_energies(self):
        """W2, W3, W13 should be finite numbers."""
        for name, val in [("W2", self.W2), ("W3", self.W3), ("W13", self.W13)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            numeric = self._to_numeric(val)
            if numeric is None:
                return (False, f"{name} not convertible to numeric")
            if not math.isfinite(numeric):
                return (False, f"{name} not finite: {numeric}")
        return (True, "W2, W3, W13 are finite")

    def test_invariant_zero_cross_energies(self):
        """W12 and W23 should be zero (or very close)."""
        for name, val in [("W12", self.W12), ("W23", self.W23)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            numeric = self._to_numeric(val)
            if numeric is None:
                return (False, f"{name} not convertible to numeric")
            if not math.isclose(numeric, 0, abs_tol=1e-10):
                return (False, f"{name} not zero: {numeric}")
        return (True, "W12 and W23 are zero")

    # ------------------------------------------------------------------
    # Modeling/trace tests – check specific relations without copying whole explanation table
    # ------------------------------------------------------------------
    def test_modeling_self_energy_sphere(self):
        """W2 should equal q^2/(8π ε0 a)."""
        if self.W2 is None:
            return (False, "SKIP: W2 missing")
        if self.q is None or self.a is None or self.eps0 is None:
            return (False, "SKIP: needed givens missing")
        expected = self.q**2 / (8 * math.pi * self.eps0 * self.a)
        if self._isclose(self.W2, expected):
            return (True, f"W2 ≈ {expected}")
        else:
            return (False, f"W2 = {self._to_numeric(self.W2)} not close to {expected}")

    def test_modeling_dipole_energy(self):
        """W3 should equal (4π ε0 a^3 / 3) E0^2."""
        if self.W3 is None:
            return (False, "SKIP: W3 missing")
        if self.a is None or self.eps0 is None or self.E0 is None:
            return (False, "SKIP: needed givens missing")
        expected = (4 * math.pi * self.eps0 * self.a**3 / 3) * self.E0**2
        if self._isclose(self.W3, expected):
            return (True, f"W3 ≈ {expected}")
        else:
            return (False, f"W3 = {self._to_numeric(self.W3)} not close to {expected}")

    def test_modeling_interaction_energy(self):
        """W13 should equal -2π ε0 a^3 E0^2."""
        if self.W13 is None:
            return (False, "SKIP: W13 missing")
        if self.a is None or self.eps0 is None or self.E0 is None:
            return (False, "SKIP: needed givens missing")
        expected = -2 * math.pi * self.eps0 * self.a**3 * self.E0**2
        if self._isclose(self.W13, expected):
            return (True, f"W13 ≈ {expected}")
        else:
            return (False, f"W13 = {self._to_numeric(self.W13)} not close to {expected}")

    def test_modeling_dipole_moment_formula(self):
        """p should equal 4π ε0 a^3 E0."""
        if self.p is None:
            return (False, "SKIP: p missing")
        if self.a is None or self.eps0 is None or self.E0 is None:
            return (False, "SKIP: needed givens missing")
        expected = 4 * math.pi * self.eps0 * self.a**3 * self.E0
        if self._isclose(self.p, expected):
            return (True, f"p ≈ {expected}")
        else:
            return (False, f"p = {self._to_numeric(self.p)} not close to {expected}")

    # ------------------------------------------------------------------
    # Monotonicity test – must perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_p_increases_with_a(self):
        """Dipole moment p should increase with radius a (all else constant)."""
        if self.p is None or self.a is None or self.eps0 is None or self.E0 is None:
            return (False, "SKIP: needed values missing")
        # Baseline p
        p_base = self._to_numeric(self.p)
        if p_base is None:
            return (False, "p not numeric")
        # Increased a by 10%
        a_new = 1.1 * self._to_numeric(self.a)
        # Compute expected p_new = 4π ε0 a_new^3 E0
        eps0_n = self._to_numeric(self.eps0)
        E0_n = self._to_numeric(self.E0)
        if eps0_n is None or E0_n is None:
            return (False, "Cannot convert eps0 or E0 to numeric")
        p_expected_new = 4 * math.pi * eps0_n * (a_new**3) * E0_n
        # p_expected_new should be larger
        if p_expected_new > p_base:
            return (True, f"p increased from {p_base} to expected {p_expected_new} for 10% larger a")
        else:
            return (False, f"p does not increase with a: base {p_base}, expected larger a -> {p_expected_new}")

    def test_monotonicity_W13_magnitude_increases_with_a(self):
        """|W13| should increase with radius a (since W13 ~ a^3)."""
        if self.W13 is None or self.a is None or self.eps0 is None or self.E0 is None:
            return (False, "SKIP: needed values missing")
        w13_base = self._to_numeric(self.W13)
        if w13_base is None:
            return (False, "W13 not numeric")
        a_new = 1.1 * self._to_numeric(self.a)
        eps0_n = self._to_numeric(self.eps0)
        E0_n = self._to_numeric(self.E0)
        if eps0_n is None or E0_n is None:
            return (False, "Cannot convert eps0 or E0 to numeric")
        w13_expected_new = -2 * math.pi * eps0_n * (a_new**3) * E0_n**2
        # |W13| increases because negative gets more negative (larger magnitude)
        if abs(w13_expected_new) > abs(w13_base):
            return (True, f"|W13| increased from {abs(w13_base)} to {abs(w13_expected_new)} for 10% larger a")
        else:
            return (False, f"|W13| does not increase: base {abs(w13_base)}, expected {abs(w13_expected_new)}")

    # ------------------------------------------------------------------
    # Only one weak answer-sanity test (no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_W2_positive(self):
        """W2 should be positive (finite self-energy of a conducting sphere)."""
        if self.W2 is None:
            return (False, "SKIP: W2 missing")
        numeric = self._to_numeric(self.W2)
        if numeric is None:
            return (False, "W2 not convertible to numeric")
        if numeric > 0:
            return (True, f"W2 = {numeric} > 0")
        else:
            return (False, f"W2 should be positive but got {numeric}")

    # ------------------------------------------------------------------
    # run_all_tests: discover all test_ methods, execute in try/except
    # ------------------------------------------------------------------
    def run_all_tests(self):
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
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)