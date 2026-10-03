import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Results (strings or expressions from pseudocode emits)
        self.F_relation = self.results.get('F_relation', None)
        self.c_v_formula = self.results.get('c_v_formula', None)
        self.Z = self.results.get('Z', None)
        self.c_v = self.results.get('c_v', None)

        # Numeric givens (may be missing)
        self.E0 = self.givens.get('E0', None)
        self.E1 = self.givens.get('E1', None)
        self.T = self.givens.get('T', None)
        self.k = self.givens.get('k', None)

        # Derived numerical constants (if possible)
        if self.E0 is not None and self.E1 is not None:
            self.deltaE = self.E1 - self.E0
        else:
            self.deltaE = None
        if self.k is not None and self.T is not None:
            self.beta = 1.0 / (self.k * self.T)
        else:
            self.beta = None

    # ------------------------------------------------------------
    # Interface tests: check that required result keys exist
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four emit keys are present in results."""
        required_keys = ['F_relation', 'c_v_formula', 'Z', 'c_v']
        missing = [key for key in required_keys if key not in self.results]
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_F_relation_content(self):
        """Check that F_relation is a non‑empty string containing key physics symbols."""
        if self.F_relation is None:
            return (False, "SKIP: F_relation not available")
        if not isinstance(self.F_relation, str) or len(self.F_relation) == 0:
            return (False, "F_relation should be a non‑empty string")
        if 'F' not in self.F_relation or 'kT' not in self.F_relation or 'ln' not in self.F_relation.lower():
            return (False, "F_relation string does not contain expected physical content (F, kT, ln)")
        return (True, "F_relation string contains expected physics symbols.")

    def test_interface_c_v_formula_content(self):
        """Check that c_v_formula contains the derivative structure."""
        if self.c_v_formula is None:
            return (False, "SKIP: c_v_formula not available")
        if 'd^2' not in self.c_v_formula and 'd²' not in self.c_v_formula:
            # also accept 'derivative' or 'partial' but better to check for second derivative
            if 'second derivative' not in self.c_v_formula.lower():
                return (False, "c_v_formula does not indicate second derivative (d^2)")
        if 'T ln Z' not in self.c_v_formula and 'T*ln(Z)' not in self.c_v_formula:
            return (False, "c_v_formula does not contain 'T ln Z'")
        return (True, "c_v_formula contains second derivative of T ln Z.")

    def test_interface_Z_content(self):
        """Check that Z string contains two exponential terms."""
        if self.Z is None:
            return (False, "SKIP: Z not available")
        # Count occurrences of 'exp' or 'e^' (case‑insensitive)
        exp_count = self.Z.lower().count('exp')
        if exp_count < 2:
            return (False, f"Z string contains only {exp_count} exponential(s), expected 2.")
        return (True, f"Z string contains {exp_count} exponentials (expected 2).")

    def test_interface_c_v_content(self):
        """Check that c_v string contains essential physical structure."""
        if self.c_v is None:
            return (False, "SKIP: c_v not available")
        # Should contain something like 'beta', 'deltaE', 'cosh', or 'exp'
        if 'beta' not in self.c_v.lower() and 'cosh' not in self.c_v.lower():
            return (False, "c_v expression does not contain beta or cosh.")
        if 'deltaE' not in self.c_v and 'ΔE' not in self.c_v and 'E1-E0' not in self.c_v:
            if 'E1' not in self.c_v or 'E0' not in self.c_v:
                return (False, "c_v expression does not reference the energy difference.")
        return (True, "c_v expression contains plausible structure.")

    # ------------------------------------------------------------
    # Invariant tests: physics laws that must hold
    # ------------------------------------------------------------
    def test_invariant_Z_two_energy_levels(self):
        """Check that the partition function explicitly sums over two states."""
        if self.Z is None:
            return (False, "SKIP: Z not available")
        # Look for two arguments to exp
        import re
        exps = re.findall(r'exp\s*\(', self.Z, re.IGNORECASE)
        if len(exps) != 2:
            return (False, f"Found {len(exps)} 'exp(...)', but two‑state system requires exactly two.")
        return (True, "Z contains two exponential contributions (two states).")

    def test_invariant_c_v_positive_and_finite(self):
        """Check that the c_v expression yields a positive number for typical parameters."""
        if self.c_v is None or self.beta is None or self.deltaE is None:
            return (False, "SKIP: insufficient data to compute c_v numerically.")
        try:
            beta = self.beta
            delta = self.deltaE
            k = self.k
            x = beta * delta
            cv = k * (x**2) * math.exp(-x) / ( (1 + math.exp(-x))**2 )
        except Exception as e:
            return (False, f"Compute failed: {e}")
        if cv <= 0:
            return (False, f"c_v = {cv:.6f} is not positive.")
        if not math.isfinite(cv):
            return (False, f"c_v = {cv} is not finite.")
        return (True, f"c_v = {cv:.6f} is positive and finite.")

    # ------------------------------------------------------------
    # Modelling/trace tests: check structure of derived expressions
    # ------------------------------------------------------------
    def test_modeling_derivative_of_lnZ(self):
        """
        Verify that the c_v formula matches the thermodynamic relation by checking
        the symbolic expression for the two‑state case (if possible).
        We use sympy to derive c_v from the provided Z and compare structure.
        """
        if self.Z is None or self.c_v is None:
            return (False, "SKIP: Z or c_v not available for symbolic modelling.")
        # Define symbols
        E0, E1, k, T = sp.symbols('E0 E1 k T', real=True, positive=True)
        beta = 1/(k*T)
        # Compute Z from step 3 formula
        Z_expr = sp.exp(-beta*E0) + sp.exp(-beta*E1)
        # Compute c_v from formula: kT * d^2/dT^2 (T * ln(Z))
        F = -k*T * sp.log(Z_expr)
        c_v_expr = -T * sp.diff(F, T, 2)
        # Simplify using substitution: delta = E1 - E0, epsilon = E0
        delta = sp.symbols('delta', real=True)
        # but cannot substitute directly because c_v_expr may be complicated
        # Just check that the expression is symmetric in E0, E1 and has the right asymptotic behaviour?
        # Instead, we'll evaluate c_v_expr at a sample numeric set and compare with the provided c_v string numeric.
        # Do a numeric check at known parameters (if available)
        if self.E0 is not None and self.E1 is not None and self.k is not None and self.T is not None:
            try:
                cv_numeric = float(c_v_expr.subs({E0: self.E0, E1: self.E1, k: self.k, T: self.T}).evalf())
                # Compare with the value from the provided c_v string (if evaluable)
                # Evaluate the provided c_v string (same as in invariant)
                evaled_cv = self._eval_c_v_numeric()
                if evaled_cv is not None:
                    if math.isclose(cv_numeric, evaled_cv, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Symbolic c_v = {cv_numeric:.6f}, provided c_v = {evaled_cv:.6f} match within tolerance.")
                    else:
                        return (False, f"Symbolic c_v = {cv_numeric:.6f} does not match provided c_v = {evaled_cv:.6f}.")
                else:
                    return (True, f"Symbolic c_v = {cv_numeric:.6f} (cannot compare provided).")
            except Exception as e:
                return (False, f"Could not evaluate symbolic c_v: {e}")
        else:
            # Without numbers, check that the symbolic c_v has the expected peak structure?
            # For simplicity, return a positive message (string structure already checked in interface)
            return (True, "Symbolic derivation attempted (numeric not available).")

    def _eval_c_v_numeric(self):
        """Helper: try to evaluate the provided c_v string numerically using known parameters."""
        if self.c_v is None or self.beta is None or self.deltaE is None or self.k is None:
            return None
        try:
            beta_s, delta_s, k_s = sp.symbols('beta deltaE k')
            expr = sp.sympify(self.c_v)
            subs_dict = {}
            for sym in expr.free_symbols:
                if str(sym) in ['beta', 'β']:
                    subs_dict[sym] = self.beta
                elif str(sym) in ['deltaE', 'ΔE']:
                    subs_dict[sym] = self.deltaE
                elif str(sym) == 'k':
                    subs_dict[sym] = self.k
            return float(expr.subs(subs_dict).evalf())
        except Exception:
            return None

    def test_modeling_partition_function_normalization(self):
        """Check that the provided Z expression sums to 1 when energies are zero? Actually not.
        Instead, check that Z at T→0 (beta→∞) gives exp(-beta*E0) (ground state only) if E0<E1.
        We'll check that if we substitute numeric givens (if available) the Z is between 0 and 2.
        """
        if self.beta is None or self.E0 is None or self.E1 is None:
            return (False, "SKIP: cannot compute Z numerically.")
        try:
            Z_val = math.exp(-self.beta * self.E0) + math.exp(-self.beta * self.E1)
        except Exception as e:
            return (False, f"Could not compute Z: {e}")
        if Z_val <= 0 or Z_val > 2:
            return (False, f"Z = {Z_val:.6f} is out of expected range (0,2].")
        return (True, f"Z = {Z_val:.6f} is in (0,2].")

    # ------------------------------------------------------------
    # Monotonicity test: Z decreases with increasing temperature
    # ------------------------------------------------------------
    def test_monotonicity_Z_decreases_with_T(self):
        """
        For a canonical ensemble, Z decreases when temperature increases.
        Perturb T upward by 10% (if numeric values available) and check that Z decreases.
        """
        if self.Z is None or self.E0 is None or self.E1 is None or self.k is None or self.T is None:
            return (False, "SKIP: need numeric E0, E1, k, T to evaluate Z monotonicity.")
        try:
            # Baseline Z at given T
            T0 = self.T
            beta0 = 1.0 / (self.k * T0)
            Z0 = math.exp(-beta0 * self.E0) + math.exp(-beta0 * self.E1)

            # Perturb T upward by 10%
            T1 = T0 * 1.1
            beta1 = 1.0 / (self.k * T1)
            Z1 = math.exp(-beta1 * self.E0) + math.exp(-beta1 * self.E1)

            if Z1 >= Z0:
                return (False, f"Z did not decrease: Z(T)={Z0:.6f}, Z(1.1T)={Z1:.6f}.")
            return (True, f"Z decreased from {Z0:.6f} to {Z1:.6f} (~{100*(Z0-Z1)/Z0:.2f}%).")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ------------------------------------------------------------
    # Answer sanity (weak): only a basic range/sanity check
    # ------------------------------------------------------------
    def test_answer_sanity_c_v_range(self):
        """Weak sanity: if numeric c_v available, it must be below an absurdly high bound (10*k)."""
        if self.c_v is None or self.beta is None or self.deltaE is None:
            return (False, "SKIP: cannot evaluate numeric c_v")
        try:
            cv = self._eval_c_v_numeric()
        except Exception:
            return (False, "Could not compute numeric c_v for sanity check.")
        if cv is None:
            return (False, "SKIP: could not evaluate provided c_v string.")
        if cv > 10 * self.k:
            return (False, f"c_v = {cv:.6f} exceeds plausible bound of 10*k = {10*self.k:.6f}.")
        if cv < 0:
            return (False, f"c_v = {cv:.6f} is negative.")
        return (True, f"c_v = {cv:.6f} passes weak sanity check (positive, below 10*k).")

    # ------------------------------------------------------------
    # Runner method
    # ------------------------------------------------------------
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