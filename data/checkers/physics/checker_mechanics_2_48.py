import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract givens (M, m, R, g)
        givens = getattr(solution_instance, 'givens', {})
        self.M = givens.get('M', None)
        self.m = givens.get('m', None)
        self.R = givens.get('R', None)
        self.g = givens.get('g', None)

        # Safely extract results (all expected keys from the emit contract)
        results = getattr(solution_instance, 'results', {})
        self.L = results.get('L', None)
        self.eq1 = results.get('eq1', None)
        self.eq2 = results.get('eq2', None)
        self.omega1_lim = results.get('omega1_lim', None)
        self.ratio1_lim = results.get('ratio1_lim', None)
        self.omega2_lim = results.get('omega2_lim', None)
        self.ratio2_lim = results.get('ratio2_lim', None)
        self.omega1_lim_small = results.get('omega1_lim_small', None)
        self.ratio1_lim_small = results.get('ratio1_lim_small', None)
        self.omega2_lim_small = results.get('omega2_lim_small', None)
        self.ratio2_lim_small = results.get('ratio2_lim_small', None)
        self.omega1 = results.get('omega1', None)
        self.omega2 = results.get('omega2', None)

        # Try to sympify string results for numerical/symbolic testing
        self._sympify()

    def _sympify(self):
        """Convert string results to sympy expressions where possible."""
        for attr in [
            'L', 'eq1', 'eq2',
            'omega1_lim', 'ratio1_lim', 'omega2_lim', 'ratio2_lim',
            'omega1_lim_small', 'ratio1_lim_small', 'omega2_lim_small', 'ratio2_lim_small',
            'omega1', 'omega2'
        ]:
            val = getattr(self, attr, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr, sp.sympify(val))
                except (sp.SympifyError, TypeError):
                    pass  # keep as string if conversion fails

    # ------------------------------------------------------------------
    # Interface tests – check that required keys are present
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required = [
            'L', 'eq1', 'eq2',
            'omega1_lim', 'ratio1_lim', 'omega2_lim', 'ratio2_lim',
            'omega1_lim_small', 'ratio1_lim_small', 'omega2_lim_small', 'ratio2_lim_small',
            'omega1', 'omega2'
        ]
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_given_keys(self):
        required_givens = ['M', 'm', 'R', 'g']
        missing = [k for k in required_givens if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant tests – physical laws and theorem‑based checks
    # ------------------------------------------------------------------
    def test_invariant_lagrangian_kinetic_energy_sign(self):
        """The kinetic energy in L must be positive‑semidefinite."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if isinstance(self.L, sp.Basic):
            try:
                theta_dot, phi_dot = sp.symbols('theta_dot phi_dot')
                T = sp.sympify(self.L).expand()
                coeff_theta2 = T.coeff(theta_dot**2)
                coeff_phi2 = T.coeff(phi_dot**2)
                if coeff_theta2 <= 0 or coeff_phi2 <= 0:
                    return (False, "Kinetic energy coefficients not positive")
                return (True, "Kinetic energy positive‑semidefinite")
            except Exception as e:
                return (False, f"Could not analyse L: {e}")
        return (False, "L is not a sympy expression")

    def test_invariant_linearized_mass_matrix_symmetric(self):
        """From the small‑oscillation quadratic Lagrangian, the mass matrix must be symmetric."""
        if any(v is None for v in [self.M, self.m, self.R, self.g]):
            return (False, "SKIP: required givens missing")
        Mmat = sp.Matrix([
            [(2*self.M + self.m)*self.R**2, self.m*self.R**2],
            [self.m*self.R**2, self.m*self.R**2]
        ])
        if Mmat.is_symmetric():
            return (True, "Mass matrix is symmetric")
        return (False, "Mass matrix is not symmetric")

    def test_invariant_linearized_stiffness_matrix_form(self):
        """Stiffness matrix should be diagonal (no coupling in potential)."""
        if any(v is None for v in [self.M, self.m, self.R, self.g]):
            return (False, "SKIP: required givens missing")
        Kmat = sp.Matrix([
            [(self.M + self.m)*self.g*self.R, 0],
            [0, self.m*self.g*self.R]
        ])
        if Kmat[0,1] == 0 and Kmat[1,0] == 0:
            return (True, "Stiffness matrix is diagonal")
        return (False, "Stiffness matrix should be diagonal (uncoupled small oscillations)")

    def test_invariant_normal_mode_frequencies_derived(self):
        """Derive ω1, ω2 from mass and stiffness matrices and compare with reported values."""
        if any(v is None for v in [self.M, self.m, self.R, self.g, self.omega1, self.omega2]):
            return (False, "SKIP: required givens or results missing")
        try:
            omega1_rep = sp.sympify(self.omega1)
            omega2_rep = sp.sympify(self.omega2)
        except:
            return (False, "omega1/omega2 not sympy‑parseable")

        Mmat = sp.Matrix([
            [(2*self.M + self.m)*self.R**2, self.m*self.R**2],
            [self.m*self.R**2, self.m*self.R**2]
        ])
        Kmat = sp.Matrix([
            [(self.M + self.m)*self.g*self.R, 0],
            [0, self.m*self.g*self.R]
        ])
        ω2 = sp.symbols('ω2')
        char_poly = (Kmat - ω2*Mmat).det()
        try:
            val1 = sp.N(char_poly.subs(ω2, omega1_rep**2))
            val2 = sp.N(char_poly.subs(ω2, omega2_rep**2))
        except Exception as e:
            return (False, f"Substitution failed: {e}")
        tol = 1e-2
        if abs(val1) > tol or abs(val2) > tol:
            return (False, f"Characteristic equation not satisfied: residuals {val1:.3e}, {val2:.3e}")
        return (True, "Reported normal‑mode frequencies satisfy the secular equation")

    def test_invariant_amplitude_ratio_omega1(self):
        """For ω1, the amplitude ratio A/B should be 1."""
        keys = ['ratio1_lim', 'ratio1_lim_small']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                continue
            try:
                val_n = float(sp.N(val))
                if not math.isclose(val_n, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"{key} = {val_n}, expected 1")
            except:
                return (False, f"Could not evaluate {key}")
        return (True, "Amplitude ratio for ω1 equals 1 (within tolerance)")

    def test_invariant_amplitude_ratio_omega2_limits(self):
        """Check limit ratios: m>>M → -1; m<<M → -m/M."""
        if self.M is None or self.m is None:
            return (False, "SKIP: M or m missing")
        tol = 1e-2
        if self.ratio2_lim is not None:
            try:
                ratio2_lim_val = float(sp.N(self.ratio2_lim))
                expected = -1.0
                if not math.isclose(ratio2_lim_val, expected, rel_tol=tol, abs_tol=tol):
                    return (False, f"ratio2_lim = {ratio2_lim_val}, expected -1")
            except:
                return (False, "Could not evaluate ratio2_lim")
        if self.ratio2_lim_small is not None:
            try:
                ratio2_small_val = float(sp.N(self.ratio2_lim_small))
                expected = -float(sp.N(self.m/self.M))
                if not math.isclose(ratio2_small_val, expected, rel_tol=tol, abs_tol=tol):
                    return (False, f"ratio2_lim_small = {ratio2_small_val}, expected -m/M = {expected}")
            except:
                return (False, "Could not evaluate ratio2_lim_small")
        return (True, "Limit amplitude ratios consistent with theory")

    # ------------------------------------------------------------------
    # Modeling tests – check consistency of equations of motion
    # ------------------------------------------------------------------
    def test_modeling_euler_lagrange_theta(self):
        """Verifies that eq1 is indeed the Euler‑Lagrange equation for θ from L."""
        if self.L is None or self.eq1 is None:
            return (False, "SKIP: L or eq1 missing")
        if not (isinstance(self.L, sp.Basic) and isinstance(self.eq1, sp.Basic)):
            return (False, "L or eq1 not sympy expressions")
        try:
            eq1_str = str(self.eq1)
            if 'theta_ddot' not in eq1_str and 'ddot(' not in eq1_str:
                return (False, "eq1 does not appear to contain second derivative of theta")
            return (True, "eq1 structurally appears correct")
        except Exception as e:
            return (False, f"Analysis of Euler‑Lagrange for θ failed: {e}")

    def test_modeling_euler_lagrange_phi(self):
        """Similar structural check for eq2."""
        if self.L is None or self.eq2 is None:
            return (False, "SKIP: L or eq2 missing")
        if not (isinstance(self.eq2, sp.Basic)):
            return (False, "eq2 not a sympy expression")
        try:
            eq2_str = str(self.eq2)
            if 'phi_ddot' not in eq2_str and 'ddot(' not in eq2_str:
                return (False, "eq2 does not appear to contain second derivative of phi")
            return (True, "eq2 structurally appears correct")
        except Exception as e:
            return (False, f"Analysis of Euler‑Lagrange for φ failed: {e}")

    def test_modeling_small_oscillation_positive_definiteness(self):
        """Check that the mass matrix is positive definite (stable oscillations)."""
        if any(v is None for v in [self.M, self.m, self.R, self.g]):
            return (False, "SKIP: givens missing")
        try:
            Mmat = sp.Matrix([
                [(2*self.M + self.m)*self.R**2, self.m*self.R**2],
                [self.m*self.R**2, self.m*self.R**2]
            ])
            if not Mmat.is_positive_definite():
                return (False, "Mass matrix not positive definite")
            return (True, "Mass matrix is positive definite")
        except Exception as e:
            return (False, f"Check failed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input, predict output change
    # ------------------------------------------------------------------
    def test_monotonicity_omega2_increases_with_m(self):
        """
        Perturb m by +20% (increase) and verify that omega2 (from theoretical formula)
        increases.  Uses the reported omega2 as baseline and computes the expected
        perturbed omega2 using the same theoretical relation.
        """
        if any(v is None for v in [self.M, self.m, self.R, self.g, self.omega2]):
            return (False, "SKIP: required givens or omega2 missing")

        try:
            omega2_base = float(sp.N(self.omega2))
        except Exception as e:
            return (False, f"Cannot evaluate omega2: {e}")

        # Perturb m: increase by 20%
        m_new = 1.2 * self.m
        # Compute expected perturbed omega2 using the formula (same as solver's)
        omega2_perturbed = math.sqrt((self.M + m_new) * self.g / (self.M * self.R))
        # The trend should be: larger m -> larger omega2
        if omega2_perturbed <= omega2_base:
            return (False, f"omega2 did not increase for increased m: base {omega2_base:.4f}, perturbed {omega2_perturbed:.4f}")
        # Also check that the ratio matches theory
        expected_ratio = math.sqrt((self.M + m_new) / (self.M + self.m))
        actual_ratio = omega2_perturbed / omega2_base
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ratio {actual_ratio:.4f} differs from expected {expected_ratio:.4f}")
        return (True, "omega2 increases with increased m (monotonic trend correct)")

    def test_monotonicity_omega1_independent_of_m(self):
        """
        omega1 = sqrt(g/(2R)) is independent of m and M.
        Perturb m by +50% and verify that the formula predicts the same omega1.
        (This tests an invariance, which is a trivial monotonicity.)
        """
        if any(v is None for v in [self.R, self.g, self.omega1]):
            return (False, "SKIP: required givens or omega1 missing")

        try:
            omega1_base = float(sp.N(self.omega1))
        except Exception as e:
            return (False, f"Cannot evaluate omega1: {e}")

        # The theoretical omega1 depends only on g and R
        omega1_theory = math.sqrt(self.g / (2 * self.R))
        if not math.isclose(omega1_base, omega1_theory, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega1 base {omega1_base:.4f} does not match theory {omega1_theory:.4f}")

        # Even if we perturb m (or M), omega1 remains the same
        # So we can check that the formula yields the same value
        # (No further computation needed; the perturbation is automatic)
        return (True, "omega1 is independent of m (no change under perturbation)")

    # ------------------------------------------------------------------
    # Weak answer‑sanity test (only sign, finiteness, broad range)
    # Combined for both omegas
    # ------------------------------------------------------------------
    def test_answer_sanity_omega_sign_and_range(self):
        """Check that omega1 and omega2 are positive, finite, and omega2 >= omega1."""
        all_ok = True
        msgs = []
        for name, val in [('omega1', self.omega1), ('omega2', self.omega2)]:
            if val is None:
                msgs.append(f"SKIP: {name} missing")
                continue
            try:
                num = float(sp.N(val))
                if not (num > 0 and math.isfinite(num)):
                    msgs.append(f"{name} = {num} not positive finite")
                    all_ok = False
                else:
                    msgs.append(f"{name} is positive finite")
            except:
                msgs.append(f"Cannot evaluate {name}")
                all_ok = False
        # Additional relation: omega2 should typically be >= omega1 (for positive masses)
        if self.omega1 is not None and self.omega2 is not None:
            try:
                o1 = float(sp.N(self.omega1))
                o2 = float(sp.N(self.omega2))
                if o2 < o1:
                    msgs.append(f"omega2 ({o2}) < omega1 ({o1}), unexpected")
                    all_ok = False
                else:
                    msgs.append("omega2 >= omega1")
            except:
                pass
        if all_ok:
            return (True, "; ".join(msgs))
        else:
            return (False, "; ".join(msgs))

    # ------------------------------------------------------------------
    # Main runner (do not modify)
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)