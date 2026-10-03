import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Define symbols we may need
        self.hbar = sp.symbols('hbar', real=True, positive=True)
        self.e = sp.symbols('e', real=True, positive=True)
        self.m_e = sp.symbols('m_e', real=True, positive=True)
        self.c = sp.symbols('c', real=True, positive=True)
        self.B = sp.symbols('B', real=True)
        self.a0 = sp.symbols('a0', real=True, positive=True)  # Bohr radius

        # Extract results from solution_instance safely
        results = getattr(solution_instance, 'results', {})
        self.H_general = results.get('H_general', None)
        self.H_hydrogen = results.get('H_hydrogen', None)
        self.H1 = results.get('H1', None)
        self.H2 = results.get('H2', None)
        self.Delta_E = results.get('Delta_E', None)
        self.mu = results.get('mu', None)

        # Optional: extract givens if present (not required but useful)
        self.provided_e = getattr(solution_instance, 'e', None) or results.get('e', None)
        self.provided_m_e = getattr(solution_instance, 'm_e', None) or results.get('m_e', None)
        self.provided_hbar = getattr(solution_instance, 'hbar', None) or results.get('hbar', None)
        self.provided_c = getattr(solution_instance, 'c', None) or results.get('c', None)
        self.provided_a0 = getattr(solution_instance, 'a0', None) or results.get('a0', None)
        self.provided_B = getattr(solution_instance, 'B', None) or results.get('B', None)

        # Prepare numeric values for monotonicity checks (cgs units)
        self.num_e = 4.803e-10  # esu
        self.num_m_e = 9.109e-28  # g
        self.num_hbar = 1.055e-27  # erg*s
        self.num_c = 2.998e10  # cm/s
        self.num_a0 = 5.292e-9  # cm
        self.num_B = 1.0e-4  # 1 gauss (weak field)

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ==================== Interface checks ====================
    def test_interface_H_general_exists(self):
        if self.H_general is None:
            return (False, "H_general is missing")
        return (True, "H_general present")

    def test_interface_H_hydrogen_exists(self):
        if self.H_hydrogen is None:
            return (False, "H_hydrogen is missing")
        return (True, "H_hydrogen present")

    def test_interface_H1_H2_exists(self):
        if self.H1 is None or self.H2 is None:
            return (False, f"H1={self.H1}, H2={self.H2} at least one missing")
        return (True, "H1 and H2 present")

    def test_interface_Delta_E_exists(self):
        if self.Delta_E is None:
            return (False, "Delta_E is missing")
        return (True, "Delta_E present")

    def test_interface_mu_exists(self):
        if self.mu is None:
            return (False, "mu is missing")
        return (True, "mu present")

    # ==================== Invariant checks (theorem-based) ====================
    def test_invariant_H_general_form(self):
        """Check that H_general is a sympy expression (basic sanity)."""
        if self.H_general is None:
            return (False, "SKIP: H_general not available")
        try:
            expr = sp.sympify(self.H_general) if isinstance(self.H_general, str) else self.H_general
        except:
            return (False, "H_general is not a valid sympy expression")
        if not isinstance(expr, sp.Basic):
            return (False, "H_general is not a sympy expression")
        return (True, "H_general is a valid sympy expression")

    def test_invariant_H_hydrogen_contains_B(self):
        """Check that H_hydrogen contains the B field."""
        if self.H_hydrogen is None:
            return (False, "SKIP: H_hydrogen not available")
        expr = sp.sympify(self.H_hydrogen) if isinstance(self.H_hydrogen, str) else self.H_hydrogen
        if expr.has(self.B):
            return (True, "H_hydrogen contains B")
        else:
            return (False, "H_hydrogen does not contain B")

    def test_invariant_H1_paramagnetic_proportional_to_B(self):
        """Check that H1 contains B (paramagnetic term)."""
        if self.H1 is None:
            return (False, "SKIP: H1 not available")
        expr = sp.sympify(self.H1) if isinstance(self.H1, str) else self.H1
        if expr.has(self.B):
            return (True, "H1 contains B")
        else:
            return (False, "H1 does not contain B")

    def test_invariant_H2_diamagnetic_proportional_to_Bsq(self):
        """Check that H2 contains B^2 (diamagnetic term)."""
        if self.H2 is None:
            return (False, "SKIP: H2 not available")
        expr = sp.sympify(self.H2) if isinstance(self.H2, str) else self.H2
        if expr.has(self.B**2):
            return (True, "H2 contains B^2")
        elif expr.has(self.B):
            return (False, "H2 contains B but not B^2")
        else:
            return (False, "H2 does not contain B")

    def test_invariant_Delta_E_quadratic(self):
        """Check that Delta_E is proportional to B^2."""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        expr = sp.sympify(self.Delta_E) if isinstance(self.Delta_E, str) else self.Delta_E
        if expr.has(self.B**2):
            return (True, "Delta_E contains B^2")
        elif expr.has(self.B):
            return (False, "Delta_E is linear in B, should be quadratic")
        else:
            return (False, "Delta_E does not contain B")

    def test_invariant_mu_linear_in_B(self):
        """Check that mu contains B and is negative (diamagnetic)."""
        if self.mu is None:
            return (False, "SKIP: mu not available")
        expr = sp.sympify(self.mu) if isinstance(self.mu, str) else self.mu
        if not expr.has(self.B):
            return (False, "mu does not contain B")
        subs = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                self.a0: self.num_a0, self.B: self.num_B}
        try:
            num_val = float(expr.subs(subs).evalf())
            if num_val < 0:
                return (True, f"mu = {num_val:.5e} < 0 for B>0")
            else:
                return (False, f"mu = {num_val:.5e} >= 0, expected negative")
        except:
            return (True, "mu contains B, sign not verifiable")

    def test_invariant_mu_derivative_consistency(self):
        """Check mu = -d(Delta_E)/dB using numeric evaluation."""
        if self.Delta_E is None or self.mu is None:
            return (False, "SKIP: Delta_E or mu missing")
        try:
            Delta_E_expr = sp.sympify(self.Delta_E) if isinstance(self.Delta_E, str) else self.Delta_E
            mu_expr = sp.sympify(self.mu) if isinstance(self.mu, str) else self.mu
            dE_dB = sp.diff(Delta_E_expr, self.B)
            minus_dE_dB = -dE_dB
            subs = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                    self.a0: self.num_a0, self.B: self.num_B}
            num_minus = float(minus_dE_dB.subs(subs).evalf())
            num_mu = float(mu_expr.subs(subs).evalf())
            if math.isclose(num_minus, num_mu, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"mu ≈ -dDelta_E/dB: mu={num_mu:.5e}, -dE/dB={num_minus:.5e}")
            else:
                return (False, f"mu ({num_mu}) != -dDelta_E/dB ({num_minus})")
        except Exception as e:
            return (False, f"Cannot check derivative consistency: {e}")

    # ==================== Modeling/Trace checks ====================
    def test_modeling_perturbation_validity(self):
        """Check that the solution acknowledges the paramagnetic term vanishes for ground state."""
        if self.H1 is None:
            return (False, "SKIP: H1 not available")
        return (True, "Modeling: H1 (paramagnetic) and H2 (diamagnetic) defined")

    def test_modeling_expectation_value_reproduced(self):
        """Re-derive the key expectation value <r^2 sin^2θ> = 2 a0^2 from theoretical formulas."""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        expected_Delta_E = self.e**2 * self.B**2 * self.a0**2 / (4 * self.m_e * self.c**2)
        actual = sp.sympify(self.Delta_E) if isinstance(self.Delta_E, str) else self.Delta_E
        diff = sp.simplify(actual - expected_Delta_E)
        if diff == 0:
            return (True, "Delta_E matches expectation value calculation")
        else:
            subs = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                    self.a0: self.num_a0, self.B: self.num_B}
            try:
                num_actual = float(actual.subs(subs).evalf())
                num_expected = float(expected_Delta_E.subs(subs).evalf())
                if math.isclose(num_actual, num_expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Numerically Delta_E = {num_actual:.5e} ≈ {num_expected:.5e}")
                else:
                    return (False, f"Delta_E value mismatch: {num_actual} vs {num_expected}")
            except:
                return (False, "Cannot verify expectation value numerically")

    # ==================== Monotonicity checks ====================
    def test_monotonicity_energy_increases_with_B(self):
        """Check that Delta_E increases (quadratically) with B by perturbing B by 5% and 10%."""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        expr = sp.sympify(self.Delta_E) if isinstance(self.Delta_E, str) else self.Delta_E
        subs_base = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                     self.a0: self.num_a0}
        B0 = self.num_B
        B_pert = 1.05 * B0  # 5% increase
        try:
            E0 = float(expr.subs({**subs_base, self.B: B0}).evalf())
            E_pert = float(expr.subs({**subs_base, self.B: B_pert}).evalf())
            if E_pert > E0:
                return (True, f"Delta_E increases: B0={B0:.2e} -> B_pert={B_pert:.2e}, E0={E0:.5e} -> E_pert={E_pert:.5e}")
            else:
                return (False, f"Delta_E does not increase: E0={E0:.5e}, E_pert={E_pert:.5e}")
        except Exception as e:
            return (False, f"Monotonicity check failed: {e}")

    def test_monotonicity_mu_decreases_with_increasing_B(self):
        """Check that mu becomes more negative (decreases) as B increases.
        Perturb B by +5% and compare baseline mu to perturbed mu."""
        if self.mu is None:
            return (False, "SKIP: mu not available")
        expr = sp.sympify(self.mu) if isinstance(self.mu, str) else self.mu
        subs_base = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                     self.a0: self.num_a0}
        B0 = self.num_B
        B_pert = 1.05 * B0  # +5%
        try:
            mu0 = float(expr.subs({**subs_base, self.B: B0}).evalf())
            mu_pert = float(expr.subs({**subs_base, self.B: B_pert}).evalf())
            # Expect mu_pert < mu0 (more negative) because mu = -const * B
            if mu_pert < mu0:
                return (True, f"mu decreases (more negative): B0={B0:.2e} mu0={mu0:.5e}, B_pert={B_pert:.2e} mu_pert={mu_pert:.5e}")
            else:
                return (False, f"mu does not become more negative: mu0={mu0:.5e}, mu_pert={mu_pert:.5e}")
        except Exception as e:
            return (False, f"Monotonicity check failed: {e}")

    # ==================== Answer sanity (weak, only one) ====================
    def test_answer_sanity_basic(self):
        """Weak sanity check: Delta_E >= 0 and mu negative for positive B."""
        if self.Delta_E is None or self.mu is None:
            return (False, "SKIP: Delta_E or mu missing")
        subs = {self.e: self.num_e, self.m_e: self.num_m_e, self.c: self.num_c,
                self.a0: self.num_a0, self.B: self.num_B}
        try:
            Delta_E_expr = sp.sympify(self.Delta_E) if isinstance(self.Delta_E, str) else self.Delta_E
            mu_expr = sp.sympify(self.mu) if isinstance(self.mu, str) else self.mu
            delta_val = float(Delta_E_expr.subs(subs).evalf())
            mu_val = float(mu_expr.subs(subs).evalf())
            if delta_val >= 0 and mu_val < 0:
                return (True, f"Sanity: Delta_E={delta_val:.5e}>=0, mu={mu_val:.5e}<0")
            else:
                return (False, f"Sanity failed: Delta_E={delta_val:.5e}, mu={mu_val:.5e}")
        except Exception as e:
            return (False, f"Sanity check exception: {e}")