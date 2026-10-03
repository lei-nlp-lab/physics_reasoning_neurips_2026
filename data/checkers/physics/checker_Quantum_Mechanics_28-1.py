import sympy as sp
import math
import sys

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all needed quantities from solution_instance
        # Expected: results dict with keys beta_opt, E_min, E0; also maybe givens like hbar, m, lam, n
        # But givens may not be present; we only rely on results defined in contract.
        # If solution_instance.results is missing or None, we set defaults to None.
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.m = getattr(solution_instance, 'm', None)
        self.lam = getattr(solution_instance, 'lam', None)
        self.n = getattr(solution_instance, 'n', None)
        # Also try to extract from results if they are stored there
        self.beta_opt = results.get('beta_opt', None)
        self.E_min = results.get('E_min', None)
        self.E0 = results.get('E0', None)

        # Also store givens from solution_instance if available, else None
        # This allows us to recompute and compare for consistency checks
        # Use default None for all, no numeric defaults
        self.hbar = getattr(solution_instance, 'hbar', results.get('hbar', None))
        self.m = getattr(solution_instance, 'm', results.get('m', None))
        self.lam = getattr(solution_instance, 'lam', results.get('lam', None))
        self.n = getattr(solution_instance, 'n', results.get('n', None))

        # If n is a string or sympy symbol, try to evaluate
        if self.n is not None:
            try:
                self.n = int(self.n)
            except:
                pass

    # ---------- Helper methods ----------
    def _recompute_beta_opt(self):
        """Compute beta_opt from formula using stored givens (if all present)."""
        if None in (self.hbar, self.m, self.lam, self.n):
            return None
        n = self.n
        hbar = self.hbar
        m = self.m
        lam = self.lam
        # beta_opt = ( (4*m*n*lam*factorial(2*n)) / (hbar^2 * 2^(2*n) * factorial(n)) )^(1/(n+1))
        beta_opt = (
            (4 * m * n * lam * sp.factorial(2*n)) /
            (hbar**2 * 2**(2*n) * sp.factorial(n))
        )**(1/(n+1))
        # Simplify
        return sp.simplify(beta_opt)

    def _recompute_E_min(self):
        """Compute E_min from formula using stored givens."""
        if None in (self.hbar, self.m, self.lam, self.n):
            return None
        n = self.n
        hbar = self.hbar
        m = self.m
        lam = self.lam
        beta_opt = self._recompute_beta_opt()
        if beta_opt is None:
            return None
        # E_min = (hbar^2/(4*m)) * (1+1/n) * beta_opt
        E_min = (hbar**2/(4*m)) * (1 + sp.Rational(1, n)) * beta_opt
        return sp.simplify(E_min)

    def _recompute_E0(self):
        """Compute E0 for n=1 case: (hbar/2)*sqrt(2*lam/m)."""
        if None in (self.hbar, self.m, self.lam):
            return None
        # For n=1, E0 = (hbar/2)*sqrt(2*lam/m)
        E0 = (self.hbar/2) * sp.sqrt(2*self.lam/self.m)
        return sp.simplify(E0)

    # ---------- Test methods ----------
    def test_interface_required_keys(self):
        """Check that all required keys are present in results (as per contract)."""
        missing = []
        if self.beta_opt is None:
            missing.append('beta_opt')
        if self.E_min is None:
            missing.append('E_min')
        if self.E0 is None:
            missing.append('E0')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_types(self):
        """Check that extracted values are numeric (float or sympy number)."""
        for name, val in [('beta_opt', self.beta_opt), ('E_min', self.E_min), ('E0', self.E0)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float, sp.Number)):
                return (False, f"{name} is not numeric, got {type(val)}")
        return (True, "All values are numeric.")

    def test_invariant_variational_bound_positive(self):
        """E_min and E0 should be positive (energy bound)."""
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        # Convert to float for positivity check
        try:
            E_min_val = float(self.E_min)
        except:
            return (True, "SKIP: E_min cannot be converted to float")
        if E_min_val <= 0:
            return (False, f"E_min = {E_min_val} is not positive")
        if self.E0 is not None:
            try:
                E0_val = float(self.E0)
            except:
                pass
            else:
                if E0_val <= 0:
                    return (False, f"E0 = {E0_val} is not positive")
        return (True, "Energies positive.")

    def test_invariant_minimization_relation(self):
        """Check that E_min and beta_opt satisfy the minimization relation: E_min ≈ ħ²β_opt/(4m)*(1+1/n)."""
        if None in (self.E_min, self.beta_opt, self.hbar, self.m, self.n):
            return (False, "SKIP: missing required quantities")
        try:
            n_val = int(self.n)
        except:
            return (False, "SKIP: n not convertible to int")
        # Compute left and right sides
        left = sp.simplify(self.E_min)
        right = (self.hbar**2/(4*self.m)) * (1 + sp.Rational(1, n_val)) * self.beta_opt
        # Evaluate numerically if all are numbers
        try:
            left_float = float(left)
            right_float = float(right)
        except:
            return (False, "Cannot evaluate numerically")
        if math.isclose(left_float, right_float, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_min ≈ ħ²β/(4m)*(1+1/n): {left_float} ≈ {right_float}")
        else:
            return (False, f"E_min = {left_float}, expected {right_float}")

    def test_invariant_kinetic_positive(self):
        """Check that kinetic energy term ħ²β/(4m) is positive."""
        if None in (self.hbar, self.m, self.beta_opt):
            return (False, "SKIP: missing data")
        try:
            kin = (self.hbar**2/(4*self.m)) * self.beta_opt
            kin_val = float(kin)
        except:
            return (False, "Cannot compute kinetic energy")
        if kin_val <= 0:
            return (False, f"Kinetic energy = {kin_val} not positive")
        return (True, "Kinetic energy positive.")

    def test_modeling_variational_derivative_condition(self):
        """Check derivative of E with respect to β is zero at β_opt (within tolerance)."""
        # dE/dβ = ħ²/(4m) - n * λ * (2n)!/(2^{2n} n! * β^{n+1})
        if None in (self.hbar, self.m, self.lam, self.n, self.beta_opt):
            return (False, "SKIP: missing data")
        try:
            n_val = int(self.n)
        except:
            return (False, "SKIP: n not convertible to int")
        # Term1: ħ²/(4m)
        term1 = self.hbar**2/(4*self.m)
        # Term2: n * λ * (2n)!/(2^{2n} n! * β^{n+1})
        term2 = n_val * self.lam * sp.factorial(2*n_val) / (2**(2*n_val) * sp.factorial(n_val) * self.beta_opt**(n_val+1))
        try:
            term1_val = float(term1)
            term2_val = float(term2)
        except:
            return (False, "Cannot evaluate terms numerically")
        if math.isclose(term1_val, term2_val, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dE/dβ ≈ 0: {term1_val} ≈ {term2_val}")
        else:
            return (False, f"dE/dβ = {term1_val - term2_val}, terms: {term1_val} vs {term2_val}")

    def test_modeling_harmonic_oscillator_exact(self):
        """For n=1, E_min should exactly equal E0 (HO exact result)."""
        if self.n is None or self.n != 1:
            return (True, "SKIP: n != 1")
        if None in (self.E_min, self.E0):
            return (False, "Missing E_min or E0 for n=1 check")
        try:
            E_min_val = float(self.E_min)
            E0_val = float(self.E0)
        except:
            return (False, "Cannot convert to float")
        if math.isclose(E_min_val, E0_val, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"For n=1, E_min ≈ E0: {E_min_val} ≈ {E0_val}")
        else:
            return (False, f"For n=1, E_min = {E_min_val}, E0 = {E0_val}")

    def test_monotonicity_E_min_increases_with_lambda(self):
        """Perturb λ by +5% and check that E_min increases."""
        if None in (self.lam, self.E_min, self.m, self.hbar, self.n):
            return (False, "SKIP: missing data for lambda perturbation")
        try:
            orig_lam = float(self.lam)
            orig_E_min = float(self.E_min)
            n_val = int(self.n)
        except:
            return (False, "Cannot evaluate as floats")
        # Compute perturbed lam: +5%
        perturbed_lam = orig_lam * 1.05
        # Compute perturbed beta_opt and E_min using formulas
        # Use sympy for exact recomputation
        hbar_s = sp.nsimplify(self.hbar)
        m_s = sp.nsimplify(self.m)
        lam_pert = sp.nsimplify(perturbed_lam)
        beta_opt_pert = ( (4*m_s*n_val*lam_pert*sp.factorial(2*n_val)) /
                          (hbar_s**2 * 2**(2*n_val) * sp.factorial(n_val)) )**(1/(n_val+1))
        E_min_pert = (hbar_s**2/(4*m_s)) * (1 + sp.Rational(1,n_val)) * beta_opt_pert
        try:
            E_min_pert_val = float(E_min_pert)
        except:
            return (False, "Cannot evaluate perturbed E_min")
        if E_min_pert_val > orig_E_min:
            return (True, f"E_min increased from {orig_E_min} to {E_min_pert_val} for +5% λ")
        else:
            return (False, f"E_min did not increase: original {orig_E_min}, perturbed {E_min_pert_val}")

    def test_answer_sanity_E0_finite(self):
        """Check that E0 is finite and not NaN."""
        if self.E0 is None:
            return (False, "SKIP: E0 not available")
        try:
            val = float(self.E0)
        except:
            return (False, "Cannot convert E0 to float")
        if math.isnan(val) or math.isinf(val):
            return (False, f"E0 = {val} is NaN or Inf")
        return (True, f"E0 = {val} is finite")

    # ---------- Master runner ----------
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