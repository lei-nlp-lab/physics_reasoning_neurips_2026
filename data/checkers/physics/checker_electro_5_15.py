import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all needed information with safe defaults
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.pseudocode = getattr(solution_instance, 'pseudocode', '')

        # Givens (use None if missing)
        self.R = self.givens.get('R', None)
        self.v = self.givens.get('v', None)
        self.B = self.givens.get('B', None)
        self.epsilon0 = self.givens.get('epsilon0', None)

        # Result key(s)
        self.sigma_raw = self.results.get('sigma', None)

        # Convert to sympy expression if it's a string or sympy expression
        if isinstance(self.sigma_raw, str):
            try:
                self.sigma = sp.sympify(self.sigma_raw)
            except Exception:
                self.sigma = None
        else:
            self.sigma = self.sigma_raw  # may already be a sympy expression or None

        # Define common symbols
        self.theta = sp.Symbol('theta', real=True)
        self.r = sp.Symbol('r', real=True)
        # Symbolic constants for internal use
        self.eps0_sym = sp.Symbol('epsilon0', positive=True)
        self.E0_sym = sp.Symbol('E0', positive=True)

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

    # --------------------------------------------
    # Interface checks
    # --------------------------------------------
    def test_interface_sigma_key(self):
        """Check that sigma key exists and is not None."""
        if self.sigma_raw is None:
            return (False, "Missing 'sigma' in results")
        return (True, "sigma key present")

    def test_interface_givens_basic(self):
        """Check that essential givens are provided."""
        missing = []
        for key in ['R', 'v', 'B']:
            if self.givens.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All basic givens present (R, v, B)")

    # --------------------------------------------
    # Invariant / Theorem checks
    # --------------------------------------------
    def test_invariant_total_charge_zero(self):
        """Total induced charge must be zero: ∫ sigma dA = 0."""
        if self.sigma is None:
            return (False, "SKIP: sigma expression not available")

        # Attempt symbolic integration
        try:
            integrand = self.sigma * self.R**2 * sp.sin(self.theta)
            # Use symbolic R if needed
            R_expr = self.R if self.R is not None else sp.Symbol('R')
            integrand_sub = integrand.subs(self.R, R_expr) if self.R is None else integrand
            int_theta = sp.integrate(integrand_sub, (self.theta, 0, sp.pi))
            total = 2 * sp.pi * int_theta
            if sp.simplify(total) == 0:
                return (True, "Symbolic integration shows zero total charge")
        except Exception:
            pass  # Fall to numerical

        # Numerical fallback: need numeric R and appropriate parameters
        if self.R is None:
            return (False, "SKIP: need numeric R for numerical integration")
        try:
            free_syms = self.sigma.free_symbols
            sub_dict = {}
            if sp.Symbol('v') in free_syms and self.v is not None:
                sub_dict[sp.Symbol('v')] = self.v
            if sp.Symbol('B') in free_syms and self.B is not None:
                sub_dict[sp.Symbol('B')] = self.B
            if sp.Symbol('epsilon0') in free_syms:
                sub_dict[sp.Symbol('epsilon0')] = 8.854187817e-12
            if isinstance(self.R, (int, float)):
                sub_dict[sp.Symbol('R')] = self.R
            for sym in free_syms:
                if sym not in sub_dict:
                    sub_dict[sym] = 1.0  # fallback for unknown symbols
            sigma_num_expr = self.sigma.subs(sub_dict)
            f_num = sp.lambdify(self.theta, sigma_num_expr, 'numpy')
            import numpy as np
            N = 200
            dtheta = np.pi / N
            theta_vals = np.linspace(0, np.pi, N+1)[:-1] + dtheta/2
            sigma_vals = f_num(theta_vals)
            integrand_num = sigma_vals * np.sin(theta_vals) * dtheta
            total_num = 2 * np.pi * np.sum(integrand_num)
            if isinstance(self.R, (int, float)):
                total_num *= self.R**2
            if math.isclose(total_num, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numerical integration gives zero total charge ({total_num:.2e})")
            else:
                return (False, f"Total charge not zero: {total_num:.2e}")
        except Exception as e:
            return (False, f"Numerical integration failed: {e}")

    def test_invariant_dipole_moment(self):
        """Induced dipole moment p_z = ∫ z σ dA should equal 4π ε0 R^3 E0."""
        if self.sigma is None:
            return (False, "SKIP: sigma expression not available")

        # Compute dipole moment analytically from sigma
        try:
            # p_z = ∫ (r cosθ) σ dA, with dA = R^2 sinθ dθ dφ, evaluated at r=R
            p_expr = sp.integrate(
                sp.integrate(
                    self.R * sp.cos(self.theta) * self.sigma * self.R**2 * sp.sin(self.theta),
                    (self.theta, 0, sp.pi)
                ),
                (sp.Symbol('phi'), 0, 2*sp.pi)
            )
            p_expr = sp.simplify(p_expr)
            # Expected: 4π ε0 R^3 E0, where E0 = v B
            expected = 4 * sp.pi * self.eps0_sym * self.R**3 * self.E0_sym
            if self.v is not None and self.B is not None:
                expected_sub = expected.subs({self.E0_sym: self.v * self.B})
                diff = sp.simplify(p_expr - expected_sub)
                if diff == 0:
                    return (True, "Symbolic dipole moment matches expected 4π ε0 R^3 v B")
            # Numeric fallback
            if self.v is not None and self.B is not None and isinstance(self.R, (int, float)):
                sub_dict = {
                    sp.Symbol('epsilon0'): 8.854187817e-12,
                    sp.Symbol('v'): self.v,
                    sp.Symbol('B'): self.B,
                    sp.Symbol('R'): self.R,
                }
                p_num = float(p_expr.subs(sub_dict).evalf())
                expected_num = 4 * math.pi * 8.854187817e-12 * (self.R**3) * self.v * self.B
                if math.isclose(p_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Numerical dipole moment {p_num:.2e} matches expected {expected_num:.2e}")
                else:
                    return (False, f"Dipole moment mismatch: {p_num:.2e} vs {expected_num:.2e}")
            else:
                return (False, "Cannot compute numeric dipole moment (need numeric R, v, B)")
        except Exception as e:
            return (False, f"Dipole moment test failed: {e}")

    # --------------------------------------------
    # Modeling / Trace checks
    # --------------------------------------------
    def test_modeling_cos_dependence(self):
        """Surface charge density should be proportional to cosθ."""
        if self.sigma is None:
            return (False, "SKIP: sigma expression not available")
        # Check that sigma(θ+π) = -sigma(θ)
        sym_shift = sp.pi
        sigma_shifted = self.sigma.subs(self.theta, self.theta + sym_shift)
        sigma_sum = sp.simplify(self.sigma + sigma_shifted)
        if sigma_sum == 0:
            return (True, "Symbolic check: sigma(θ) is odd in cosθ")
        # Numeric fallback
        try:
            sub_dict = {}
            if self.v is not None:
                sub_dict[sp.Symbol('v')] = self.v
            if self.B is not None:
                sub_dict[sp.Symbol('B')] = self.B
            if self.epsilon0 is not None:
                sub_dict[sp.Symbol('epsilon0')] = self.epsilon0
            if not sub_dict:
                return (False, "SKIP: cannot evaluate numerically without numeric v, B, epsilon0")
            sigma_num = self.sigma.subs(sub_dict)
            f_num = sp.lambdify(self.theta, sigma_num, 'numpy')
            import numpy as np
            theta_test = np.array([0.0, np.pi])
            vals = f_num(theta_test)
            if math.isclose(vals[0], -vals[1], rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numerical check: sigma(0)={vals[0]:.3e}, sigma(π)={vals[1]:.3e} (oddness holds)")
            else:
                return (False, f"Oddness fails: sigma(0)={vals[0]:.3e}, sigma(π)={vals[1]:.3e}")
        except Exception as e:
            return (False, f"Numeric oddness check failed: {e}")

    def test_modeling_scaling_with_E0(self):
        """Coefficient of cosθ should equal 3 ε0 v B."""
        if self.sigma is None:
            return (False, "SKIP: sigma expression not available")
        try:
            coeff = sp.simplify(sp.expand(self.sigma) / sp.cos(self.theta))
            expected = 3 * self.eps0_sym * sp.Symbol('v') * sp.Symbol('B')
            diff = sp.simplify(coeff - expected)
            if diff == 0:
                return (True, "Coefficient matches 3 ε0 v B")
            # Numeric fallback
            sub_dict = {}
            if self.v is not None:
                sub_dict[sp.Symbol('v')] = self.v
            if self.B is not None:
                sub_dict[sp.Symbol('B')] = self.B
            if self.epsilon0 is not None:
                sub_dict[sp.Symbol('epsilon0')] = self.epsilon0
            if not sub_dict:
                return (False, f"Symbolic coefficient {coeff} does not simplify to expected")
            coeff_num = float(coeff.subs(sub_dict).evalf())
            expected_num = 3 * self.epsilon0 * self.v * self.B if all(x is not None for x in [self.epsilon0, self.v, self.B]) else None
            if expected_num is None:
                return (False, "Cannot compute numeric expected value (epsilon0, v, B needed)")
            if math.isclose(coeff_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numeric coefficient {coeff_num:.3e} close to expected {expected_num:.3e}")
            else:
                return (False, f"Coefficient mismatch: {coeff_num:.3e} vs {expected_num:.3e}")
        except Exception as e:
            return (False, f"Coefficient check failed: {e}")

    # --------------------------------------------
    # Monotonicity tests
    # --------------------------------------------
    def test_monotonicity_sigma_increases_with_v(self):
        """If v increases by +10%, sigma (at θ=0) should increase (direction) and scale approximately linearly."""
        if self.v is None or self.B is None or self.epsilon0 is None:
            return (False, "SKIP: need numeric v, B, epsilon0")
        if self.sigma is None:
            return (False, "SKIP: no sigma expression")
        try:
            # Baseline at given v
            sub_dict = {
                sp.Symbol('v'): self.v,
                sp.Symbol('B'): self.B,
                sp.Symbol('epsilon0'): self.epsilon0,
                self.theta: 0.0
            }
            sigma0 = float(self.sigma.subs(sub_dict).evalf())
            # Perturb v upward by 10%
            sub_dict[sp.Symbol('v')] = self.v * 1.1
            sigma1 = float(self.sigma.subs(sub_dict).evalf())
            # Direction check: sigma1 should be greater than sigma0
            if sigma1 <= sigma0:
                return (False, f"sigma did not increase: sigma0={sigma0:.3e}, sigma1={sigma1:.3e}")
            # Optional: check approximate linearity
            if sigma0 != 0:
                ratio = sigma1 / sigma0
                expected_ratio = 1.1
                if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"sigma increased (ratio {ratio:.5f} close to 1.1)")
                else:
                    return (False, f"sigma increased but ratio {ratio:.5f} not close to 1.1")
            else:
                return (False, "sigma0 is zero; cannot test scaling")
        except Exception as e:
            return (False, f"Monotonicity test failed: {e}")

    def test_monotonicity_sigma_increases_with_B(self):
        """If B increases by +10%, sigma (at θ=0) should increase (direction) and scale approximately linearly."""
        if self.v is None or self.B is None or self.epsilon0 is None:
            return (False, "SKIP: need numeric v, B, epsilon0")
        if self.sigma is None:
            return (False, "SKIP: no sigma expression")
        try:
            # Baseline at given B
            sub_dict = {
                sp.Symbol('v'): self.v,
                sp.Symbol('B'): self.B,
                sp.Symbol('epsilon0'): self.epsilon0,
                self.theta: 0.0
            }
            sigma0 = float(self.sigma.subs(sub_dict).evalf())
            # Perturb B upward by 10%
            sub_dict[sp.Symbol('B')] = self.B * 1.1
            sigma1 = float(self.sigma.subs(sub_dict).evalf())
            # Direction check: sigma1 should be greater than sigma0
            if sigma1 <= sigma0:
                return (False, f"sigma did not increase: sigma0={sigma0:.3e}, sigma1={sigma1:.3e}")
            # Optional: check approximate linearity
            if sigma0 != 0:
                ratio = sigma1 / sigma0
                expected_ratio = 1.1
                if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"sigma increased (ratio {ratio:.5f} close to 1.1)")
                else:
                    return (False, f"sigma increased but ratio {ratio:.5f} not close to 1.1")
            else:
                return (False, "sigma0 is zero; cannot test scaling")
        except Exception as e:
            return (False, f"Monotonicity test failed: {e}")

    # --------------------------------------------
    # Answer sanity (weak)
    # --------------------------------------------
    def test_answer_sanity_sign(self):
        """Weak sanity: for v>0, B>0, cosθ>0, sigma should be positive."""
        if not (self.v is not None and self.B is not None and self.epsilon0 is not None):
            return (False, "SKIP: need numeric v, B, epsilon0")
        if self.sigma is None:
            return (False, "SKIP: no sigma expression")
        try:
            sub_dict = {
                sp.Symbol('v'): self.v,
                sp.Symbol('B'): self.B,
                sp.Symbol('epsilon0'): self.epsilon0,
                self.theta: 0.0
            }
            sigma_val = float(self.sigma.subs(sub_dict).evalf())
            if sigma_val > 0:
                return (True, f"sigma(0) = {sigma_val:.3e} > 0 as expected")
            else:
                return (False, f"sigma(0) = {sigma_val:.3e} should be positive")
        except Exception as e:
            return (False, f"Sign check failed: {e}")