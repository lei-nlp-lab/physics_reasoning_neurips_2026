import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        
        # Extract givens dictionary safely (optional)
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        
        # Extract expected keys safely
        self.q = self.results.get('q', None)
        self.L = self.results.get('L', None)
        self.x_eq = self.results.get('x_eq', None)
        self.z_eq = self.results.get('z_eq', None)
        self.omega_sq = self.results.get('omega_sq', None)
        self.omega = self.results.get('omega', None)
        self.x_sol = self.results.get('x_sol', None)
        
        # Extract givens safely
        self.m = self.givens.get('m', None) if isinstance(self.givens, dict) else None
        self.g = self.givens.get('g', None) if isinstance(self.givens, dict) else None
        self.a = self.givens.get('a', None) if isinstance(self.givens, dict) else None
        
        # Constants
        self.expected_omega_sq = None
        if self.g is not None and self.a is not None and self.a != 0:
            self.expected_omega_sq = 2.0 * self.g / self.a
        
    # ---------- Helper ----------
    def _parse_latex_expression(self, expr_str):
        """Try to parse a LaTeX string into a sympy expression; return None on failure."""
        try:
            # Remove display math delimiters if present
            if expr_str.startswith('$$') and expr_str.endswith('$$'):
                expr_str = expr_str[2:-2]
            elif expr_str.startswith('$') and expr_str.endswith('$'):
                expr_str = expr_str[1:-1]
            # Use sympy parsing (may need to handle \dot{x} etc.)
            # For simplicity, we keep the raw string and check pattern matching
            # because sympy's LaTeX parser may not handle \dot.
            return sp.sympify(expr_str.replace('\\dot{x}', 'xdot').replace('\\ddot{x}', 'xddot'))
        except:
            return None
    
    # ---------- Interface Tests ----------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and are not None."""
        required_keys = ['q', 'L', 'x_eq', 'z_eq', 'omega_sq', 'omega', 'x_sol']
        missing = [k for k in required_keys if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        # Also check types: q should be a string, others may be strings or numbers
        if not isinstance(self.q, str):
            return (False, "q should be a string (the generalized coordinate name)")
        return (True, "All required keys present and valid")
    
    # ---------- Invariant Tests (use math.isclose) ----------
    def test_invariant_equilibrium_x(self):
        """Check equilibrium x_eq = 0."""
        if self.x_eq is None:
            return (False, "SKIP: x_eq not available")
        try:
            x_eq_val = float(self.x_eq)
            if math.isclose(x_eq_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"x_eq = {x_eq_val} is approximately 0")
            else:
                return (False, f"x_eq = {x_eq_val} is not near 0")
        except (ValueError, TypeError):
            return (False, f"x_eq cannot be converted to float: {self.x_eq}")
    
    def test_invariant_equilibrium_z(self):
        """Check equilibrium z_eq = 0."""
        if self.z_eq is None:
            return (False, "SKIP: z_eq not available")
        try:
            z_eq_val = float(self.z_eq)
            if math.isclose(z_eq_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"z_eq = {z_eq_val} is approximately 0")
            else:
                return (False, f"z_eq = {z_eq_val} is not near 0")
        except (ValueError, TypeError):
            return (False, f"z_eq cannot be converted to float: {self.z_eq}")
    
    def test_invariant_omega_squared_formula(self):
        """Check omega_sq = 2*g/a (if g and a are given)."""
        if self.omega_sq is None:
            return (False, "SKIP: omega_sq not available")
        if self.expected_omega_sq is None:
            return (False, "SKIP: cannot compute expected omega_sq because g or a missing")
        try:
            omega_sq_val = float(self.omega_sq)
            if math.isclose(omega_sq_val, self.expected_omega_sq, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"omega_sq = {omega_sq_val} matches expected {self.expected_omega_sq}")
            else:
                return (False, f"omega_sq = {omega_sq_val} != expected {self.expected_omega_sq}")
        except (ValueError, TypeError):
            return (False, f"omega_sq cannot be converted to float: {self.omega_sq}")
    
    # ---------- Modeling / Trace Tests ----------
    def test_modeling_lagrangian_structure(self):
        """Check that Lagrangian L contains expected pattern: kinetic minus potential, with terms like xdot^2, x^2, m, g, a."""
        if self.L is None:
            return (False, "SKIP: L not available")
        L_str = str(self.L)
        required_patterns = ['xdot', 'x^2', 'm', 'g', 'a']
        missing = [p for p in required_patterns if p not in L_str]
        if missing:
            return (False, f"Lagrangian string missing pattern(s): {missing}")
        # Check that it contains both "xdot^2" (or similar) and "x^2"
        if 'xdot^2' not in L_str and 'xdot**2' not in L_str:
            return (False, "Lagrangian should contain xdot^2 term")
        if 'x^2' not in L_str and 'x**2' not in L_str:
            return (False, "Lagrangian should contain x^2 term")
        return (True, "Lagrangian contains expected structural patterns")
    
    def test_modeling_generalized_coordinate(self):
        """Check that the generalized coordinate is 'x'."""
        if self.q is None:
            return (False, "SKIP: q not available")
        if self.q.strip().lower() == 'x':
            return (True, f"Generalized coordinate is x as expected")
        else:
            return (False, f"Generalized coordinate is '{self.q}', expected 'x'")
    
    def test_trace_linearized_equation(self):
        """Check that the linearized equation is consistent: xddot + omega_sq*x = 0."""
        # This is tricky; we can check that omega_sq is positive and that the solution x_sol contains cos/sin with frequency sqrt(omega_sq)
        if self.omega_sq is None:
            return (False, "SKIP: omega_sq not available")
        try:
            omega_sq_val = float(self.omega_sq)
            if omega_sq_val <= 0:
                return (False, f"omega_sq must be positive, got {omega_sq_val}")
        except:
            return (False, "omega_sq not numeric")
        if self.omega is None:
            # If omega not explicitly given, we can skip
            pass
        else:
            try:
                omega_val = float(self.omega)
                expected_omega = math.sqrt(omega_sq_val)
                if math.isclose(omega_val, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
                    pass
                else:
                    return (False, f"omega = {omega_val} does not match sqrt(omega_sq) = {expected_omega}")
            except:
                pass
        if self.x_sol is None:
            return (False, "SKIP: x_sol not available")
        x_sol_str = str(self.x_sol)
        # Check that solution contains omega or sqrt(2g/a) or cos/sin
        if 'cos' not in x_sol_str and 'sin' not in x_sol_str:
            return (False, "x_sol should contain a cosine or sine function")
        if 'omega' not in x_sol_str and 'sqrt' not in x_sol_str:
            return (False, "x_sol should contain reference to frequency (omega or sqrt form)")
        return (True, "Linearized equation and solution appear consistent")
    
    # ---------- Monotonicity Tests ----------
    def test_monotonicity_omega_with_a(self):
        """If a increases, omega should decrease (inverse square root)."""
        if self.g is None or self.a is None or self.a <= 0 or self.omega is None:
            return (False, "SKIP: insufficient data for monotonicity test")
        try:
            omega_base = float(self.omega)
            # Perturb a by +10%
            a_pert = self.a * 1.1
            omega_pert = math.sqrt(2.0 * self.g / a_pert)
            # Check that omega_base > omega_pert (since a increased, frequency decreases)
            if omega_base > omega_pert:
                return (True, f"omega decreases when a increases: base={omega_base}, perturbed={omega_pert}")
            else:
                return (False, f"omega did not decrease as a increased: base={omega_base}, perturbed={omega_pert}")
        except:
            return (False, "Error in monotonicity calculation")
    
    # ---------- Weak Answer Sanity ----------
    def test_answer_sanity_physical_positivity(self):
        """Check that key physical quantities are positive and finite."""
        checks = []
        if self.omega_sq is not None:
            try:
                v = float(self.omega_sq)
                if v <= 0:
                    checks.append(f"omega_sq={v} not positive")
            except:
                checks.append("omega_sq not numeric")
        if self.omega is not None:
            try:
                v = float(self.omega)
                if v <= 0:
                    checks.append(f"omega={v} not positive")
            except:
                checks.append("omega not numeric")
        if self.x_eq is not None:
            try:
                v = float(self.x_eq)
                if abs(v) > 1e-6:
                    checks.append(f"x_eq={v} not near zero")
            except:
                pass
        if self.z_eq is not None:
            try:
                v = float(self.z_eq)
                if abs(v) > 1e-6:
                    checks.append(f"z_eq={v} not near zero")
            except:
                pass
        if checks:
            return (False, "Physical positivity checks failed: " + "; ".join(checks))
        return (True, "All physical quantities have reasonable signs and values")
    
    # ---------- Main runner ----------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)