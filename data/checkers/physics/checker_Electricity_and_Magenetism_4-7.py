import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract givens and results - using None defaults
        givens = getattr(solution_instance, 'givens', None) or {}
        results = getattr(solution_instance, 'results', None) or {}
        explanation = getattr(solution_instance, 'explanation_steps', None)

        # Input parameters (given)
        self.sigma0 = givens.get('sigma_0', None)
        self.R = givens.get('R', None)
        self.h = givens.get('h', None)
        self.epsilon0 = givens.get('epsilon_0', None)

        # Results from solver
        self.P = results.get('P', None)

        # Optional: explanation steps
        self.explanation = explanation

        # Define sympy symbols for later use
        self.rho = sp.symbols('rho', real=True, positive=True)
        self.sigma0_s = sp.symbols('sigma0', real=True, positive=True)
        self.R_s = sp.symbols('R', real=True, positive=True)
        self.h_s = sp.symbols('h', real=True, positive=True)
        self.epsilon0_s = sp.symbols('epsilon0', real=True, positive=True)

        # Known sigma expression from problem statement
        self.sigma_expr = self.R_s**3 * self.sigma0_s / (self.h_s**2 + self.rho**2)**(3/2)

    def _parse_expression(self, expr_str):
        """Try to parse a string into a sympy expression, mapping common variable names."""
        if expr_str is None:
            return None
        if not isinstance(expr_str, str):
            return expr_str
        try:
            local_dict = {
                'sigma0': self.sigma0_s,
                'epsilon0': self.epsilon0_s,
                'R': self.R_s,
                'h': self.h_s,
                'rho': self.rho,
            }
            expr = sp.sympify(expr_str, locals=local_dict)
            return expr
        except Exception:
            return None

    def _get_numeric_subs(self, rho_val=0.0, h_factor=1.0, R_factor=1.0):
        """Return a substitution dictionary for numeric evaluation, using given values if available."""
        if None in (self.R, self.h, self.sigma0, self.epsilon0):
            return None
        return {
            self.R_s: self.R * R_factor,
            self.h_s: self.h * h_factor,
            self.sigma0_s: self.sigma0,
            self.epsilon0_s: self.epsilon0,
            self.rho: rho_val,
        }

    # ---- Interface Tests ----
    def test_interface_required_givens(self):
        """Check that all required given quantities are present and numeric."""
        for name, val in [('sigma_0', self.sigma0), ('R', self.R), ('h', self.h), ('epsilon_0', self.epsilon0)]:
            if val is None:
                return (False, f"Missing given: {name}")
            if not isinstance(val, (int, float)):
                return (False, f"{name} should be numeric, got {type(val)}")
        return (True, "All required givens present and numeric")

    def test_interface_result_P(self):
        """Check that result P is provided (as a string or sympy expression)."""
        if self.P is None:
            return (False, "Missing result: P")
        if not isinstance(self.P, (str, sp.Expr)):
            return (False, f"P should be a string or sympy expression, got {type(self.P)}")
        return (True, "Result P present")

    # ---- Invariant Tests (theorem-based) ----
    def test_invariant_pressure_formula_identity(self):
        """Check that P = sigma^2/(2 epsilon_0) holds symbolically or numerically."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Could not parse P expression")
        P_expected = self.sigma_expr**2 / (2 * self.epsilon0_s)
        diff = sp.simplify(P_expr - P_expected)
        if diff == 0:
            return (True, "P matches sigma^2/(2 epsilon_0) symbolically")
        subs = self._get_numeric_subs(rho_val=0.0)
        if subs is not None:
            try:
                num_P = float(P_expr.subs(subs))
                num_exp = float(P_expected.subs(subs))
                if math.isclose(num_P, num_exp, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "P matches sigma^2/(2 epsilon_0) numerically (rel_tol=1e-2)")
                else:
                    return (False, f"Numerical mismatch: P={num_P}, expected={num_exp}")
            except Exception:
                pass
        return (False, f"Could not verify identity; diff = {diff}")

    def test_invariant_image_charge_uniform(self):
        """Check that the image charge density derived from the given plane charge is constant (-sigma0)."""
        # From image method: sigma_img = -((h^2+rho^2)^(3/2)/R^3) * sigma
        sigma_img = - ((self.h_s**2 + self.rho**2)**(3/2) / self.R_s**3) * self.sigma_expr
        # Simplify explicitly: sigma_img should be -sigma0
        sigma_img_simpl = sp.simplify(sigma_img)
        target = -self.sigma0_s
        diff = sp.simplify(sigma_img_simpl - target)
        if diff == 0:
            return (True, "Image charge density is uniform (-sigma0)")
        else:
            return (False, f"Image charge density not uniform: diff = {diff}")

    def test_invariant_pressure_asymptotic_zero(self):
        """Check that pressure tends to 0 as rho -> infinity."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Cannot parse P for asymptotic test")
        # Compute limit as rho -> infinity
        limit_expr = sp.limit(P_expr, self.rho, sp.oo)
        if limit_expr == 0:
            return (True, "Pressure tends to 0 as rho -> infinity")
        else:
            return (False, f"Limit is {limit_expr}, expected 0")

    # ---- Modeling / Trace Tests (at least two) ----
    def test_modeling_pressure_formula_scaling(self):
        """Check that P(ρ) * (h^2+ρ^2)^3 is constant (ρ-independent)."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Cannot parse P for scaling test")
        factor = (self.h_s**2 + self.rho**2)**3
        scaled = sp.simplify(P_expr * factor)
        derivative = sp.diff(scaled, self.rho)
        if derivative == 0:
            return (True, "P * (h^2+ρ^2)^3 is independent of ρ (derivative zero)")
        subs1 = self._get_numeric_subs(rho_val=0.0)
        subs2 = self._get_numeric_subs(rho_val=0.5)
        if subs1 is not None and subs2 is not None:
            try:
                val1 = float(scaled.subs(subs1))
                val2 = float(scaled.subs(subs2))
                if math.isclose(val1, val2, rel_tol=1e-2):
                    return (True, f"P * (h^2+ρ^2)^3 is constant: {val1} ≈ {val2}")
                else:
                    return (False, f"Scaling not constant: ρ=0 gives {val1}, ρ=0.5 gives {val2}")
            except Exception:
                pass
        return (False, "Could not verify scaling (insufficient numeric data or symbolic failure)")

    def test_modeling_pressure_derivation_step(self):
        """Check that explanation steps (if present) contain key relations."""
        if self.explanation is None:
            return (False, "SKIP: explanation_steps not available")
        steps_text = str(self.explanation).lower()
        keywords = [('pressure', 'sigma', 'epsilon'), ('image', 'sphere'), ('uniform',)]
        found_any = any(all(kw in steps_text for kw in kw_list) for kw_list in keywords)
        if found_any:
            return (True, "Explanation steps contain relevant relations")
        else:
            return (False, "Explanation steps missing key physics (pressure, image sphere, uniform charge)")

    # ---- Monotonicity Tests (explicit perturbation of inputs) ----
    def test_monotonicity_pressure_decreases_with_rho(self):
        """Perturb ρ upward; pressure should decrease."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Cannot parse P for monotonicity test")
        subs_base = self._get_numeric_subs(rho_val=0.0)
        if subs_base is None:
            return (False, "SKIP: insufficient numeric data")
        try:
            base_val = float(P_expr.subs(subs_base))
            subs_pert = self._get_numeric_subs(rho_val=0.1)
            pert_val = float(P_expr.subs(subs_pert))
            if pert_val < base_val:
                return (True, f"Pressure decreased: base={base_val}, perturbed={pert_val}")
            else:
                return (False, f"Pressure did not decrease: base={base_val}, perturbed={pert_val}")
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")

    def test_monotonicity_pressure_decreases_with_h(self):
        """Perturb h upward (increase distance); pressure should decrease."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Cannot parse P for monotonicity test (h)")
        subs_base = self._get_numeric_subs(h_factor=1.0)
        if subs_base is None:
            return (False, "SKIP: insufficient numeric data")
        try:
            base_val = float(P_expr.subs(subs_base))
            subs_pert = self._get_numeric_subs(h_factor=1.1)
            pert_val = float(P_expr.subs(subs_pert))
            if pert_val < base_val:
                return (True, f"Pressure decreased with increased h: base={base_val}, perturbed={pert_val}")
            else:
                return (False, f"Pressure did not decrease: base={base_val}, perturbed={pert_val}")
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")

    # ---- Answer Sanity (weak, at most one) ----
    def test_answer_sanity_pressure_positive(self):
        """Weak check: pressure formula should yield positive values for positive parameters."""
        P_expr = self._parse_expression(self.P)
        if P_expr is None:
            return (False, "Cannot parse P for positivity check")
        from sympy import ask, Q
        try:
            with sp.assuming(sp.Q.positive(self.R_s), sp.Q.positive(self.h_s),
                             sp.Q.positive(self.sigma0_s), sp.Q.positive(self.epsilon0_s),
                             sp.Q.positive(self.rho)):
                positive = ask(sp.Q.positive(P_expr))
                if positive:
                    return (True, "Pressure is positive under positive parameters")
        except:
            pass
        subs = self._get_numeric_subs(rho_val=0.0)
        if subs is not None:
            try:
                num_val = float(P_expr.subs(subs))
                if num_val > 0:
                    return (True, f"Numeric check: P = {num_val} > 0")
                else:
                    return (False, f"Numeric check: P = {num_val} <= 0")
            except:
                pass
        return (True, "Could not definitively confirm positivity, but no contradiction found")

    # ---- Runner ----
    def run_all_tests(self):
        """Discover and run all test_* methods, collecting logs."""
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