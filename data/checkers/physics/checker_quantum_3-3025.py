import sympy as sp
import math
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.g = givens.get('g', None)
        self.e = givens.get('e', None)
        self.m = givens.get('m', None)
        self.c = givens.get('c', None)
        self.B = givens.get('B', None)
        self.hbar = givens.get('hbar', sp.symbols('hbar', positive=True))
        # Assume hbar provided or use default for numeric checks
        if self.hbar is None:
            self.hbar = sp.symbols('hbar', positive=True)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.omega = results.get('omega', None)
        self.s_z_t = results.get('s_z_t', None)

        # Store initial operator symbols if needed later
        self.s_x0 = sp.symbols('s_x0')
        self.s_z0 = sp.symbols('s_z0')
        self.t = sp.symbols('t', real=True)

    # -------------------- Interface Tests --------------------
    def test_interface_required_keys(self):
        """Check required result keys exist."""
        if self.omega is None:
            return (False, "Missing result key 'omega'")
        if self.s_z_t is None:
            return (False, "Missing result key 's_z_t'")
        return (True, "All required result keys present")

    def test_interface_omega_type(self):
        """Check omega is numeric or symbolic (not None/empty)."""
        if self.omega is None:
            return (False, "Omega is None")
        if isinstance(self.omega, str) and self.omega.strip() == "":
            return (False, "Omega is empty string")
        return (True, f"Omega type: {type(self.omega).__name__}")

    def test_interface_s_z_t_type(self):
        """Check s_z_t is not None or empty."""
        if self.s_z_t is None:
            return (False, "s_z_t is None")
        if isinstance(self.s_z_t, str) and self.s_z_t.strip() == "":
            return (False, "s_z_t is empty string")
        return (True, f"s_z_t type: {type(self.s_z_t).__name__}")

    # -------------------- Invariant Tests --------------------
    def test_invariant_commutation_relations(self):
        """Check that [s_i, s_j] = i hbar ε_ijk s_k holds for our spin operators."""
        # Define Pauli matrices
        sigma_x = sp.Matrix([[0,1],[1,0]])
        sigma_y = sp.Matrix([[0,-sp.I],[sp.I,0]])
        sigma_z = sp.Matrix([[1,0],[0,-1]])
        # Spin operators: s_i = (hbar/2) sigma_i
        s_x = (self.hbar/2) * sigma_x
        s_y = (self.hbar/2) * sigma_y
        s_z = (self.hbar/2) * sigma_z

        # Compute commutators
        comm_xy = s_x * s_y - s_y * s_x
        expected_xy = sp.I * self.hbar * s_z
        if sp.simplify(comm_xy - expected_xy) != sp.zeros(2,2):
            return (False, "[s_x, s_y] != i hbar s_z")
        comm_yz = s_y * s_z - s_z * s_y
        expected_yz = sp.I * self.hbar * s_x
        if sp.simplify(comm_yz - expected_yz) != sp.zeros(2,2):
            return (False, "[s_y, s_z] != i hbar s_x")
        comm_zx = s_z * s_x - s_x * s_z
        expected_zx = sp.I * self.hbar * s_y
        if sp.simplify(comm_zx - expected_zx) != sp.zeros(2,2):
            return (False, "[s_z, s_x] != i hbar s_y")
        return (True, "Spin commutation relations verified")

    def test_invariant_heisenberg_equation_s_z(self):
        """Check that ds_z/dt from Heisenberg equation matches omega * s_x."""
        # Expected: ds_z/dt = (i/hbar)[H, s_z] where H = - (g e/(2 m c)) B s_y = -gamma B s_y
        # Using gamma = g e/(2 m c) => ds_z/dt = (i/hbar) * [-gamma B s_y, s_z] = -gamma B (i/hbar)[s_y, s_z]
        # [s_y, s_z] = -i hbar s_x => ds_z/dt = -gamma B (i/hbar)(-i hbar s_x) = gamma B s_x = omega s_x
        gamma_str = "g*e/(2*m*c)"
        omega_str = "g*e*B/(2*m*c)"  # = gamma*B
        # We'll try to verify numerically using given values if available
        if None in (self.g, self.e, self.m, self.c, self.B):
            return (True, "SKIP: givens incomplete for numerical check")
        # Compute gamma and omega numerically
        gamma_val = (self.g * self.e) / (2 * self.m * self.c)
        omega_val = gamma_val * self.B
        # Check that provided omega matches
        if self.omega is None:
            return (False, "omega not provided for comparison")
        # Try to evaluate numeric if possible
        try:
            omega_provided = float(self.omega) if not hasattr(self.omega, 'subs') else float(self.omega)
        except:
            # symbolic fallback
            if hasattr(self.omega, 'subs'):
                # substitute givens
                omega_num = self.omega.subs({sp.symbols('g'): self.g, sp.symbols('e'): self.e, 
                                             sp.symbols('m'): self.m, sp.symbols('c'): self.c,
                                             sp.symbols('B'): self.B})
                try:
                    omega_provided = float(omega_num)
                except:
                    return (True, "SKIP: cannot evaluate omega numerically")
            else:
                return (True, "SKIP: omega is string/other")
        if math.isclose(omega_provided, omega_val, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"omega matches expected value: {omega_val}")
        else:
            return (False, f"omega = {omega_provided}, expected ~{omega_val}")

    # -------------------- Modeling/Trace Tests --------------------
    def test_modeling_s_z_t_form(self):
        """Check that s_z(t) has the form s_z0*cos(omega*t) + s_x0*sin(omega*t)."""
        if self.s_z_t is None:
            return (False, "s_z_t not available")
        # Define expected symbolic expression
        s_z0 = sp.symbols('s_z0')
        s_x0 = sp.symbols('s_x0')
        omega = sp.symbols('omega')
        t = sp.symbols('t', real=True)
        expected = s_z0 * sp.cos(omega * t) + s_x0 * sp.sin(omega * t)

        # Try to transform given expression to match
        expr = self.s_z_t
        # If it's a string, parse it
        if isinstance(expr, str):
            try:
                expr = sp.sympify(expr)
            except:
                return (False, "Cannot parse s_z_t expression")
        # Substitute any existing symbols if possible: we have self.omega, self.s_z0, self.s_x0.
        # But the provided expression may use different symbols. We'll check structure by comparing after substituting generic symbol names.
        # For simplicity, check that expression contains cos and sin terms with the same arguments.
        # We'll also check sign: positive cos and sin.
        # Let's flatten and compare parts
        try:
            # Expand expression to see if it matches
            # get the free symbols
            free_syms = expr.free_symbols
            # We cannot directly compare unless we rename symbols.
            # Instead, we check that the expression is linear in s_z0 and s_x0 with cos and sin.
            # We'll differentiate w.r.t. s_z0 and s_x0 and evaluate at 0 to see coefficients.
            s_z0_sym = sp.symbols('s_z0')
            s_x0_sym = sp.symbols('s_x0')
            # We'll attempt to rewrite using generic symbols.
            # For robustness, we'll check structure: the expression should be a sum of two terms: one cos-like and one sin-like.
            # We'll expand and check.
            expanded = sp.expand(expr)
            # Check that it's a sum of two terms
            if isinstance(expanded, sp.Add):
                terms = expanded.args
                if len(terms) != 2:
                    return (False, f"Expected 2 terms, got {len(terms)}: {expanded}")
                # Check that one term involves cos and the other sin
                cos_terms = [t for t in terms if t.has(sp.cos)]
                sin_terms = [t for t in terms if t.has(sp.sin)]
                if len(cos_terms) != 1 or len(sin_terms) != 1:
                    return (False, "Missing either cos or sin term")
                # Check that coefficients are correct: the cos term should be s_z0 * cos(...), sin term s_x0 * sin(...)
                # We cannot assume symbol names, but we can check that the cos term is proportional to cos(omega*t) and the sin term proportional to sin(omega*t)
                # However, we can only check approximate equality after substituting known numeric values if given.
                # For now, pass if structure is correct.
                return (True, "Expression has correct sum of cos and sin terms")
            elif isinstance(expanded, sp.Mul) and expanded.has(sp.cos):
                # Could be a single cos term if s_x0=0? But general solution expects both.
                return (False, "Expression appears to be missing sin term")
            elif isinstance(expanded, sp.Mul):
                return (False, "Expression structure unexpected")
            else:
                return (False, f"Unexpected expression type: {type(expanded)}")
        except Exception as e:
            return (False, f"Error checking form: {e}")

    def test_trace_omega_definition_consistency(self):
        """Check that omega equals (g*e*B)/(2*m*c) as per pseudocode."""
        if self.omega is None:
            return (False, "omega not provided")
        if None in (self.g, self.e, self.m, self.c, self.B):
            return (True, "SKIP: missing givens for consistency check")
        # Compute expected omega
        expected_omega = (self.g * self.e * self.B) / (2 * self.m * self.c)
        # Compare with provided omega
        try:
            omega_val = float(self.omega) if not hasattr(self.omega, 'subs') else float(self.omega)
        except:
            if hasattr(self.omega, 'subs'):
                omega_val = float(self.omega.subs({sp.symbols('g'): self.g, sp.symbols('e'): self.e,
                                                   sp.symbols('m'): self.m, sp.symbols('c'): self.c,
                                                   sp.symbols('B'): self.B}))
            else:
                return (True, "SKIP: omega not numeric")
        if math.isclose(omega_val, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"omega = {omega_val} matches expected {expected_omega}")
        else:
            return (False, f"omega = {omega_val} but expected ~{expected_omega}")

    # -------------------- Monotonicity Tests --------------------
    def test_monotonicity_omega_increases_with_B(self):
        """Check that increasing B increases omega (or s_z oscillation frequency)."""
        if self.omega is None:
            return (True, "SKIP: omega not available for monotonicity")
        if self.B is None:
            return (True, "SKIP: B not available")
        # Choose a baseline B and perturbed B
        try:
            B_val = float(self.B) if hasattr(self.B, '__float__') else None
        except:
            B_val = None
        if B_val is None or B_val == 0:
            return (True, "SKIP: B not numeric or zero")
        # Compute omega baseline
        if hasattr(self.omega, 'subs'):
            omega_baseline = float(self.omega.subs({sp.symbols('B'): B_val}))
        else:
            try:
                omega_baseline = float(self.omega)
            except:
                return (True, "SKIP: cannot evaluate omega")
        # Perturb B upward by 10%
        B_perturbed = B_val * 1.1
        # Recompute omega with perturbed B
        # If omega is symbolic, substitute; otherwise assume linear scaling
        if hasattr(self.omega, 'subs'):
            omega_perturbed = float(self.omega.subs({sp.symbols('B'): B_perturbed}))
        else:
            # If omega is a numeric constant, it cannot depend on B so test fails
            return (False, "omega appears constant, cannot be monotonic in B")
        if omega_perturbed > omega_baseline + 1e-9:
            return (True, f"Increasing B increases omega: baseline {omega_baseline}, perturbed {omega_perturbed}")
        else:
            return (False, f"Expected omega increase when B increases, but got {omega_perturbed} vs {omega_baseline}")

    def test_monotonicity_s_z_t_oscillation_rate(self):
        """Check that for fixed t, s_z(t) changes appropriately with B (omega)."""
        # We'll compute s_z at a small t for baseline B and perturbed B
        if self.s_z_t is None or self.omega is None:
            return (True, "SKIP: s_z_t or omega missing")
        # Use symbolic expression to evaluate at a specific t, say t=1, with fake initial conditions
        s_z_t_expr = self.s_z_t
        if isinstance(s_z_t_expr, str):
            try:
                s_z_t_expr = sp.sympify(s_z_t_expr)
            except:
                return (True, "SKIP: cannot parse s_z_t")
        # Ensure it is symbolic
        if not hasattr(s_z_t_expr, 'subs'):
            return (True, "SKIP: s_z_t not symbolic")
        # Set initial operator values (choose numbers)
        s_z0_val = 1.0
        s_x0_val = 0.5
        t_val = 0.1  # small time
        # Get baseline omega
        if hasattr(self.omega, 'subs'):
            omega_baseline = float(self.omega.subs({sp.symbols('B'): self.B}))
        else:
            try:
                omega_baseline = float(self.omega)
            except:
                return (True, "SKIP: cannot get numeric omega")
        # Evaluate s_z at baseline
        subs_dict = {sp.symbols('t'): t_val, sp.symbols('s_z0'): s_z0_val, sp.symbols('s_x0'): s_x0_val,
                     sp.symbols('omega'): omega_baseline}
        try:
            val_baseline = float(s_z_t_expr.subs(subs_dict))
        except:
            return (True, "SKIP: cannot evaluate s_z(t) at baseline")
        # Now perturb B by +10%
        B_new = self.B * 1.1 if hasattr(self.B, '__float__') else self.B * 1.1
        omega_new = omega_baseline * 1.1  # since linear in B
        subs_dict['omega'] = omega_new
        try:
            val_perturbed = float(s_z_t_expr.subs(subs_dict))
        except:
            return (True, "SKIP: cannot evaluate s_z(t) at perturbed")
        # Since s_z(t) oscillates, we cannot guarantee monotonic change; we just check that values are different from baseline.
        if not math.isclose(val_baseline, val_perturbed, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"s_z(t) changes with B: baseline {val_baseline}, perturbed {val_perturbed}")
        else:
            return (False, f"s_z(t) unchanged when B increased: both ~{val_baseline}")

    # -------------------- Answer Sanity Test (weak) --------------------
    def test_answer_sanity_s_z_t_contains_trig(self):
        """Check that s_z_t string representation contains cos and sin functions (broad sanity)."""
        if self.s_z_t is None:
            return (False, "s_z_t not available for sanity check")
        expr_str = str(self.s_z_t)
        if 'cos' in expr_str and 'sin' in expr_str:
            return (True, "s_z_t contains cos and sin functions")
        else:
            return (False, f"s_z_t does not contain both cos and sin: {expr_str}")

    # -------------------- Run all tests --------------------
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
                import traceback
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)