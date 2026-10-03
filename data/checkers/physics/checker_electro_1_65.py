import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for a perfectly conducting sphere in a uniform electric field.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and a weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract all required members; defaults are None.
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Givens (expected as attributes or inside results)
        self.E0 = getattr(solution_instance, 'E0', None) or self.results.get('E0', None)
        self.a = getattr(solution_instance, 'a', None) or self.results.get('a', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None) or self.results.get('epsilon0', None)

        # Emitted results
        self.sigma = self.results.get('sigma', None)
        self.p_z = self.results.get('p_z', None)

        # For angle-dependent checks we may need theta; assume it's a symbol or numeric.
        self.theta = getattr(solution_instance, 'theta', None) or self.results.get('theta', sp.Symbol('theta'))

    # ------------------------------------------------------------------
    #  Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the two required result keys are present and not None."""
        if self.sigma is None:
            return (False, "Missing required result key 'sigma'")
        if self.p_z is None:
            return (False, "Missing required result key 'p_z'")
        return (True, "Both 'sigma' and 'p_z' are present")

    def test_interface_sigma_type(self):
        """Check that sigma is either a symbolic expression or a number."""
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        # Accept sympy expression, int, float, etc.
        if isinstance(self.sigma, sp.Basic) or isinstance(self.sigma, (int, float, complex)):
            return (True, "sigma has an acceptable type")
        return (False, f"sigma has unexpected type {type(self.sigma)}")

    def test_interface_pz_type(self):
        """Check that p_z is a number or a sympy expression."""
        if self.p_z is None:
            return (False, "SKIP: p_z not available")
        if isinstance(self.p_z, (int, float, complex)) or isinstance(self.p_z, sp.Basic):
            return (True, "p_z has an acceptable type")
        return (False, f"p_z has unexpected type {type(self.p_z)}")

    # ------------------------------------------------------------------
    #  Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_total_charge_zero(self):
        """
        The surface charge density integrated over the sphere must be zero
        because the sphere is isolated (no net charge).
        """
        if self.sigma is None or self.a is None:
            return (False, "SKIP: sigma or radius missing")
        # Perform symbolic integral if sigma is a sympy expression
        if isinstance(self.sigma, sp.Basic):
            theta_sym = sp.Symbol('theta', real=True)
            # sigma might already contain theta; try substitution
            sigma_expr = self.sigma.subs(sp.Symbol('theta'), theta_sym) if self.sigma.has(sp.Symbol('theta')) else self.sigma
            integrand = sigma_expr * sp.sin(theta_sym)
            integral = sp.integrate(integrand, (theta_sym, 0, sp.pi))
            if integral == 0:
                return (True, "Total charge is zero (symbolic integration)")

            # If symbolic integration gives nonzero, maybe it simplifies to zero with assumptions
            integral_simplified = sp.simplify(integral)
            if integral_simplified == 0:
                return (True, "Total charge is zero (simplified symbolic integration)")

            # Fallback numerical check using manual trapezoidal rule (no numpy)
            try:
                # We need numeric values for epsilon0, E0, a, etc. If missing, use 1.0 for scaling.
                E0_val = float(self.E0) if self.E0 is not None else 1.0
                a_val = float(self.a) if self.a is not None else 1.0
                eps0_val = float(self.epsilon0) if self.epsilon0 is not None else 8.854e-12
                # Build numeric sigma function using lambdify
                sigma_func = sp.lambdify(theta_sym, sigma_expr, 'numpy')
                import numpy as np
                thetas = np.linspace(0, np.pi, 10001)
                values = sigma_func(thetas) * np.sin(thetas)
                integ_num = np.trapz(values, thetas) * (2*np.pi)  # full sphere surface integral
                if math.isclose(integ_num, 0.0, abs_tol=1e-12):
                    return (True, "Total charge is zero (numerical integration)")
                else:
                    return (False, f"Total charge not zero numerically: {integ_num}")
            except Exception:
                return (False, "Could not verify total charge zero")
        else:
            # Numeric sigma: we need a value of sigma for all angles? Not possible.
            return (False, "SKIP: total charge zero check requires symbolic sigma or an angular distribution")

    def test_invariant_sigma_antisymmetric(self):
        """
        Physics: σ(θ) = -σ(π-θ) because the field is uniform along z and
        the induced charge is antisymmetric.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        theta = sp.Symbol('theta', real=True)
        if isinstance(self.sigma, sp.Basic):
            # Replace theta symbol
            sigma_expr = self.sigma.subs(sp.Symbol('theta'), theta) if self.sigma.has(sp.Symbol('theta')) else self.sigma
            # sigma(pi-theta)
            sigma_pi_theta = sigma_expr.subs(theta, sp.pi - theta)
            # Check symmetry: sigma + sigma(pi-theta) = 0
            sym_check = sp.simplify(sigma_expr + sigma_pi_theta)
            if sym_check == 0:
                return (True, "σ(θ) = -σ(π-θ) holds symbolically")
            # Fallback numeric with a few angles
            try:
                # Pick a few theta values
                for th in [0.3, 0.8, 1.2, math.pi-0.3]:
                    val1 = float(sigma_expr.subs(theta, th))
                    val2 = float(sigma_expr.subs(theta, math.pi - th))
                    if not math.isclose(val1, -val2, rel_tol=1e-9, abs_tol=1e-12):
                        return (False, f"Antisymmetry fails at theta={th}: {val1} vs {-val2}")
                return (True, "σ(θ) = -σ(π-θ) confirmed numerically")
            except Exception:
                return (False, "Could not verify antisymmetry")
        else:
            # Numeric sigma is not angle-dependent? Assume it's a constant? That would be wrong physically.
            return (False, "SKIP: sigma is numeric, cannot test angular antisymmetry")

    def test_invariant_potential_bcs(self):
        """
        Check that sigma is proportional to cosθ, which follows from the boundary condition
        V(r=a)=constant and Laplace's equation.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        theta = sp.Symbol('theta', real=True)
        if isinstance(self.sigma, sp.Basic):
            sigma_expr = self.sigma.subs(sp.Symbol('theta'), theta) if self.sigma.has(sp.Symbol('theta')) else self.sigma
            # Check that sigma = constant * cosθ
            # Divide by cosθ and see if result is constant
            ratio = sp.simplify(sigma_expr / sp.cos(theta))
            # If ratio is not a function of theta, then sigma = const * cosθ
            if not ratio.has(theta):
                return (True, "sigma is proportional to cosθ")
            # Fallback numeric test: sigma(θ1)/cos(θ1) should equal sigma(θ2)/cos(θ2)
            try:
                th1, th2 = 0.2, 0.7
                v1 = float(sigma_expr.subs(theta, th1)) / math.cos(th1)
                v2 = float(sigma_expr.subs(theta, th2)) / math.cos(th2)
                if math.isclose(v1, v2, rel_tol=1e-5):
                    return (True, "sigma/cosθ is constant (numerical consistency)")
                else:
                    return (False, f"sigma/cosθ not constant: {v1} vs {v2}")
            except Exception:
                return (False, "Could not verify cosθ proportionality")
        else:
            return (False, "SKIP: sigma is numeric, cannot check angular dependence")

    # ------------------------------------------------------------------
    #  Modeling quality tests (check forms against physics)
    # ------------------------------------------------------------------
    def test_modeling_sigma_correct_form(self):
        """Check that sigma = 3 ε0 E0 cosθ symbolically or numerically."""
        if self.sigma is None or self.E0 is None or self.epsilon0 is None:
            return (False, "SKIP: missing sigma, E0, or epsilon0")
        theta = sp.Symbol('theta', real=True)
        expected_sigma = 3 * self.epsilon0 * self.E0 * sp.cos(theta)
        if isinstance(self.sigma, sp.Basic):
            # Replace any theta symbol in sigma with our symbol
            sigma_expr = self.sigma.subs(sp.Symbol('theta'), theta) if self.sigma.has(sp.Symbol('theta')) else self.sigma
            diff = sp.simplify(sigma_expr - expected_sigma)
            if diff == 0:
                return (True, "sigma exactly matches 3 ε0 E0 cosθ")
            else:
                # Numeric fallback with sample values
                try:
                    E0_val = float(self.E0) if not isinstance(self.E0, sp.Basic) else float(self.E0)
                    eps0_val = float(self.epsilon0) if not isinstance(self.epsilon0, sp.Basic) else float(self.epsilon0)
                    # Evaluate both expressions at a few angles
                    for th in [0.1, 0.5, 1.0]:
                        expected_num = 3 * eps0_val * E0_val * math.cos(th)
                        # Evaluate sigma expression
                        sigma_num = float(sigma_expr.subs(theta, th).evalf())
                        if not math.isclose(sigma_num, expected_num, rel_tol=1e-2, abs_tol=1e-12):
                            return (False, f"sigma mismatch at θ={th}: {sigma_num} vs {expected_num}")
                    return (True, "sigma matches 3 ε0 E0 cosθ numerically")
                except Exception:
                    return (False, "Could not verify sigma form numerically")
        else:
            # Numeric sigma: if it's a number, it cannot be a function of θ; it's wrong unless angle fixed
            return (False, "SKIP: sigma is a number, cannot confirm cosθ form")

    def test_modeling_pz_correct_form(self):
        """Check that p_z = 4π ε0 a^3 E0."""
        if self.p_z is None:
            return (False, "SKIP: p_z not available")
        if self.epsilon0 is None or self.a is None or self.E0 is None:
            return (False, "SKIP: missing epsilon0, a, or E0")
        expected_pz = 4 * math.pi * self.epsilon0 * self.a**3 * self.E0
        if isinstance(self.p_z, sp.Basic):
            pz_expr = self.p_z
            diff = sp.simplify(pz_expr - expected_pz)
            if diff == 0:
                return (True, "p_z exactly matches 4π ε0 a^3 E0")
            else:
                # Numeric comparison
                try:
                    a_val = float(self.a) if not isinstance(self.a, sp.Basic) else float(self.a)
                    E0_val = float(self.E0) if not isinstance(self.E0, sp.Basic) else float(self.E0)
                    eps0_val = float(self.epsilon0) if not isinstance(self.epsilon0, sp.Basic) else float(self.epsilon0)
                    expected_num = 4 * math.pi * eps0_val * a_val**3 * E0_val
                    pz_num = float(pz_expr.evalf())
                    if math.isclose(pz_num, expected_num, rel_tol=1e-5):
                        return (True, "p_z matches 4π ε0 a^3 E0 numerically")
                    else:
                        return (False, f"p_z numerical mismatch: {pz_num} vs {expected_num}")
                except Exception:
                    return (False, "Could not verify p_z form")
        else:
            # Numeric p_z
            pz_val = float(self.p_z)
            a_val = float(self.a)
            E0_val = float(self.E0)
            eps0_val = float(self.epsilon0)
            expected_num = 4 * math.pi * eps0_val * a_val**3 * E0_val
            if math.isclose(pz_val, expected_num, rel_tol=1e-5):
                return (True, "p_z matches 4π ε0 a^3 E0")
            else:
                return (False, f"p_z mismatch: {pz_val} vs {expected_num}")

    # ------------------------------------------------------------------
    #  Monotonicity / trend tests
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_increases_with_E0(self):
        """
        If we increase E0 by 10%, sigma should increase proportionally (same angle).
        We evaluate sigma at a fixed angle for two values of E0.
        """
        if self.sigma is None or self.E0 is None or self.epsilon0 is None or self.a is None:
            return (False, "SKIP: missing sigma, E0, epsilon0, or a")
        if not isinstance(self.sigma, sp.Basic):
            return (False, "SKIP: sigma is numeric, cannot vary E0")
        theta = sp.Symbol('theta', real=True)
        sigma_expr = self.sigma.subs(sp.Symbol('theta'), theta) if self.sigma.has(sp.Symbol('theta')) else self.sigma

        # Use a fixed angle theta=π/4 to avoid zero
        th_test = math.pi/4
        # Substitutions for baseline
        subs_base = {theta: th_test, sp.Symbol('E0'): self.E0, sp.Symbol('epsilon0'): self.epsilon0, sp.Symbol('a'): self.a}
        # Substitutions for perturbed E0 (10% increase)
        E0_new = 1.1 * self.E0
        subs_pert = {theta: th_test, sp.Symbol('E0'): E0_new, sp.Symbol('epsilon0'): self.epsilon0, sp.Symbol('a'): self.a}

        try:
            sigma_base = float(sigma_expr.subs(subs_base).evalf())
            sigma_pert = float(sigma_expr.subs(subs_pert).evalf())
            # Expected ratio = 1.1
            if sigma_base == 0:
                # If sigma_base is zero at this angle, choose another angle (unlikely)
                # Fallback to theta=π/6
                th_alt = math.pi/6
                subs_base_alt = {theta: th_alt, sp.Symbol('E0'): self.E0, sp.Symbol('epsilon0'): self.epsilon0, sp.Symbol('a'): self.a}
                subs_pert_alt = {theta: th_alt, sp.Symbol('E0'): E0_new, sp.Symbol('epsilon0'): self.epsilon0, sp.Symbol('a'): self.a}
                sigma_base_alt = float(sigma_expr.subs(subs_base_alt).evalf())
                sigma_pert_alt = float(sigma_expr.subs(subs_pert_alt).evalf())
                if sigma_base_alt == 0:
                    return (False, "sigma is zero for all test angles, cannot test monotonicity")
                ratio = sigma_pert_alt / sigma_base_alt
            else:
                ratio = sigma_pert / sigma_base
            if math.isclose(ratio, 1.1, rel_tol=1e-5):
                return (True, "sigma scales linearly with E0 (verified for θ=π/4)")
            else:
                return (False, f"sigma does not scale with E0: ratio {ratio} vs expected 1.1")
        except Exception:
            return (False, "Could not verify monotonicity for sigma with E0")

    def test_monotonicity_pz_scales_with_a_cubed(self):
        """
        The dipole moment scales as a^3. Check that if we increase radius by 5%,
        the dipole moment increases by (1.05)^3 ≈ 1.157625.
        """
        if self.p_z is None or self.a is None or self.E0 is None or self.epsilon0 is None:
            return (False, "SKIP: missing p_z, a, E0, or epsilon0")
        if not isinstance(self.p_z, sp.Basic):
            # Numeric p_z: compute expected scaling directly from given values
            try:
                a_val = float(self.a)
                E0_val = float(self.E0)
                eps0_val = float(self.epsilon0)
                pz_val = float(self.p_z)
                # Perturb a by 5%
                a_new = 1.05 * a_val
                expected_pz_new = 4 * math.pi * eps0_val * a_new**3 * E0_val
                # Baseline p_z should match the original formula
                expected_pz_base = 4 * math.pi * eps0_val * a_val**3 * E0_val
                if not math.isclose(pz_val, expected_pz_base, rel_tol=1e-5):
                    return (False, "p_z does not match expected baseline formula")
                ratio_actual = expected_pz_new / pz_val
                expected_ratio = 1.05**3
                if math.isclose(ratio_actual, expected_ratio, rel_tol=1e-5):
                    return (True, "p_z scales as a^3 (verified numerically)")
                else:
                    return (False, f"p_z scaling ratio {ratio_actual} vs expected {expected_ratio}")
            except Exception:
                return (False, "Could not verify p_z scaling numerically")
        else:
            # Symbolic p_z
            pz_expr = self.p_z
            a_sym = sp.Symbol('a')
            # Substitute given a
            pz_base = pz_expr.subs(a_sym, self.a).evalf()
            a_new = 1.05 * self.a
            pz_new = pz_expr.subs(a_sym, a_new).evalf()
            if pz_base == 0:
                return (False, "p_z is zero, cannot test scaling")
            ratio = float(pz_new / pz_base)
            if math.isclose(ratio, 1.05**3, rel_tol=1e-5):
                return (True, "p_z scales as a^3 (symbolic verification)")
            else:
                return (False, f"p_z scaling ratio {ratio} vs expected {1.05**3}")

    # ------------------------------------------------------------------
    #  Weak answer sanity (secondary, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: check that p_z is finite, and sign consistent with E0 if known.
        """
        if self.p_z is None:
            return (False, "SKIP: p_z not available")
        # Convert to float if possible
        if isinstance(self.p_z, sp.Basic):
            try:
                pz_val = float(self.p_z.evalf())
            except:
                pz_val = None
        else:
            try:
                pz_val = float(self.p_z)
            except:
                pz_val = None
        if pz_val is None:
            return (False, "Cannot convert p_z to float")
        if math.isnan(pz_val) or math.isinf(pz_val):
            return (False, f"p_z is NaN or infinite: {pz_val}")
        # Check sign: if E0 > 0 and a > 0, p_z should be positive
        if self.E0 is not None and self.a is not None:
            try:
                E0_val = float(self.E0)
                a_val = float(self.a)
                if E0_val > 0 and a_val > 0 and pz_val <= 0:
                    return (False, f"p_z should be positive when E0>0, a>0 but got {pz_val}")
                if E0_val < 0 and a_val > 0 and pz_val >= 0:
                    return (False, f"p_z should be negative when E0<0, a>0 but got {pz_val}")
            except:
                pass  # skip if cannot convert
        return (True, "p_z finite, sign consistent with E0 if known")

    # ------------------------------------------------------------------
    #  Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
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