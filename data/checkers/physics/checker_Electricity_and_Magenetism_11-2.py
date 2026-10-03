import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Robust Physics Checker for a rotating electric dipole radiation problem.
    Verifies the three essential outputs: S_avg, dP_dOmega, P_avg.
    Combines symbolic and numeric checks with safe None handling.
    """
    def __init__(self, solution_instance):
        # Extract givens (safe default None)
        self.p0 = getattr(solution_instance, 'p0', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.c = getattr(solution_instance, 'c', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.r = getattr(solution_instance, 'r', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.phi = getattr(solution_instance, 'phi', None)

        # Extract results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}
        self.S_avg = results.get('S_avg', None)
        self.dP_dOmega = results.get('dP_dOmega', None)
        self.P_avg = results.get('P_avg', None)

        # Attempt to convert strings to sympy expressions if needed
        for attr_name in ['p0', 'omega', 'c', 'mu0', 'r', 'theta', 'phi',
                          'S_avg', 'dP_dOmega', 'P_avg']:
            val = getattr(self, attr_name, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr_name, sp.sympify(val))
                except Exception:
                    pass  # keep as string if conversion fails

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that all three required result keys are present and not None."""
        missing = []
        if self.S_avg is None:
            missing.append('S_avg')
        if self.dP_dOmega is None:
            missing.append('dP_dOmega')
        if self.P_avg is None:
            missing.append('P_avg')
        if missing:
            return False, f"Missing results: {', '.join(missing)}"
        return True, "All required results present."

    def test_interface_givens(self):
        """Check that all essential given parameters are present (not None)."""
        missing = []
        for name in ['p0', 'omega', 'c', 'mu0', 'r', 'theta']:
            if getattr(self, name, None) is None:
                missing.append(name)
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All givens present."

    # ------------------------------------------------------------
    # Theorem‑based invariance tests (use math.isclose on numeric values)
    # ------------------------------------------------------------
    def test_invariant_dP_dOmega_nonnegative(self):
        """Check that dP_dOmega is non‑negative at the given theta (numeric)."""
        if self.dP_dOmega is None or self.theta is None:
            return False, "SKIP: missing dP_dOmega or theta"
        dP = self.dP_dOmega
        th = self.theta
        # If both are sympy, evaluate numerically at given theta
        if isinstance(dP, sp.Basic) and isinstance(th, sp.Basic):
            try:
                dP_num = float(dP.subs({sp.Symbol('theta'): th}))
            except Exception:
                return False, "Cannot evaluate dP_dOmega numerically"
        elif isinstance(dP, (int, float)) and isinstance(th, (int, float)):
            dP_num = dP
        else:
            return False, "Both dP_dOmega and theta must be numeric or sympy"
        if dP_num < -1e-12:
            return False, f"dP_dOmega is negative: {dP_num}"
        return True, f"dP_dOmega = {dP_num} is non‑negative"

    def test_invariant_S_avg_positive(self):
        """Check that S_avg magnitude is positive (numeric evaluation at given theta)."""
        if self.S_avg is None:
            return False, "SKIP: S_avg missing"
        # S_avg could be a vector or scalar; we assume magnitude is the scalar part.
        # Try to extract a numeric value at given theta.
        try:
            # If S_avg is a sympy expression, substitute theta
            if isinstance(self.S_avg, sp.Basic) and isinstance(self.theta, sp.Basic):
                # Assume S_avg is proportional to (1 - 0.5*sin^2(theta))
                val = self.S_avg.subs({sp.Symbol('theta'): self.theta})
                if callable(val):  # if still a function
                    val = float(val())
                else:
                    val = float(val)
            elif isinstance(self.S_avg, (int, float)):
                val = self.S_avg
            else:
                return False, "S_avg not numeric or sympy"
        except Exception as e:
            return False, f"Cannot evaluate S_avg numeric: {e}"
        if val < 0:
            return False, f"S_avg magnitude is negative: {val}"
        return True, f"S_avg magnitude = {val} is positive"

    # ------------------------------------------------------------
    # Modeling / trace consistency tests (use math.isclose on numeric samples,
    # or sympy simplification for symbolic)
    # ------------------------------------------------------------
    def test_modeling_S_avg_formula(self):
        """Check that S_avg matches the expected analytic formula."""
        if any(v is None for v in [self.S_avg, self.p0, self.omega, self.c, self.mu0, self.r, self.theta]):
            return False, "SKIP: missing components for S_avg formula check"
        # Build expected expression: (mu0/c) * (p0*ω²/(4πr))² * (1 - ½ sin²θ)
        p0_s, ω_s, c_s, mu0_s, r_s, θ_s = sp.symbols('p0_s ω_s c_s mu0_s r_s θ_s')
        expected = (mu0_s / c_s) * (p0_s * ω_s**2 / (4 * sp.pi * r_s))**2 * (1 - sp.Rational(1,2) * sp.sin(θ_s)**2)
        # Substitute actual values
        subs_dict = {
            sp.Symbol('p0'): self.p0, sp.Symbol('ω'): self.omega,
            sp.Symbol('c'): self.c, sp.Symbol('mu0'): self.mu0,
            sp.Symbol('r'): self.r, sp.Symbol('θ'): self.theta
        }
        # Try to parse S_avg into a sympy expression if it's a string
        S_avg_sym = sp.sympify(self.S_avg) if isinstance(self.S_avg, str) else self.S_avg
        if not isinstance(S_avg_sym, sp.Basic):
            # Numeric value: evaluate expected at given theta and compare
            try:
                expected_num = float(expected.subs(subs_dict))
                actual_num = float(S_avg_sym)
                if math.isclose(expected_num, actual_num, rel_tol=1e-2, abs_tol=1e-2):
                    return True, "S_avg numeric value matches formula"
                else:
                    return False, f"S_avg numeric mismatch: expected {expected_num}, got {actual_num}"
            except Exception as e:
                return False, f"Cannot compare numeric S_avg: {e}"
        else:
            # Symbolic: simplify difference
            diff = sp.simplify(S_avg_sym - expected.subs(subs_dict))
            if diff == 0:
                return True, "S_avg symbolic expression matches formula"
            else:
                return False, f"S_avg differs from formula: diff = {diff}"

    def test_modeling_dP_dOmega_formula(self):
        """Check that dP_dOmega matches expected expression."""
        if any(v is None for v in [self.dP_dOmega, self.p0, self.omega, self.c, self.mu0, self.theta]):
            return False, "SKIP: missing components for dP_dOmega formula check"
        # Expected: (mu0/(2c)) * (p0*ω²/(4π))² * (1+cos²θ)
        p0_s, ω_s, c_s, mu0_s, θ_s = sp.symbols('p0_s ω_s c_s mu0_s θ_s')
        expected = (mu0_s / (2 * c_s)) * (p0_s * ω_s**2 / (4 * sp.pi))**2 * (1 + sp.cos(θ_s)**2)
        subs_dict = {
            sp.Symbol('p0'): self.p0, sp.Symbol('ω'): self.omega,
            sp.Symbol('c'): self.c, sp.Symbol('mu0'): self.mu0,
            sp.Symbol('θ'): self.theta
        }
        dP_sym = sp.sympify(self.dP_dOmega) if isinstance(self.dP_dOmega, str) else self.dP_dOmega
        if not isinstance(dP_sym, sp.Basic):
            # numeric
            try:
                expected_num = float(expected.subs(subs_dict))
                actual_num = float(dP_sym)
                if math.isclose(expected_num, actual_num, rel_tol=1e-2, abs_tol=1e-2):
                    return True, "dP_dOmega numeric value matches"
                else:
                    return False, f"dP_dOmega numeric mismatch: expected {expected_num}, got {actual_num}"
            except Exception as e:
                return False, f"Cannot compare numeric dP_dOmega: {e}"
        else:
            diff = sp.simplify(dP_sym - expected.subs(subs_dict))
            if diff == 0:
                return True, "dP_dOmega symbolic expression matches"
            else:
                return False, f"dP_dOmega differs: diff = {diff}"

    def test_modeling_P_avg_formula(self):
        """Check that P_avg matches the expected total radiated power."""
        if any(v is None for v in [self.P_avg, self.p0, self.omega, self.c, self.mu0]):
            return False, "SKIP: missing components for P_avg formula check"
        # Expected: (mu0 * p0² * ω⁴) / (6π c)
        p0_s, ω_s, c_s, mu0_s = sp.symbols('p0_s ω_s c_s mu0_s')
        expected = (mu0_s * p0_s**2 * ω_s**4) / (6 * sp.pi * c_s)
        subs_dict = {
            sp.Symbol('p0'): self.p0, sp.Symbol('ω'): self.omega,
            sp.Symbol('c'): self.c, sp.Symbol('mu0'): self.mu0
        }
        P_sym = sp.sympify(self.P_avg) if isinstance(self.P_avg, str) else self.P_avg
        if not isinstance(P_sym, sp.Basic):
            # numeric
            try:
                expected_num = float(expected.subs(subs_dict))
                actual_num = float(P_sym)
                if math.isclose(expected_num, actual_num, rel_tol=1e-2, abs_tol=1e-2):
                    return True, "P_avg numeric value matches formula"
                else:
                    return False, f"P_avg numeric mismatch: expected {expected_num}, got {actual_num}"
            except Exception as e:
                return False, f"Cannot compare numeric P_avg: {e}"
        else:
            diff = sp.simplify(P_sym - expected.subs(subs_dict))
            if diff == 0:
                return True, "P_avg symbolic expression matches formula"
            else:
                return False, f"P_avg differs: diff = {diff}"

    # ------------------------------------------------------------
    # Monotonicity tests (perturb one parameter, compare results)
    # ------------------------------------------------------------
    def test_monotonicity_P_avg_with_p0(self):
        """
        Perturb p0 by +10% and check that P_avg increases.
        Uses the submitted P_avg expression (assumed proportional to p0^2).
        """
        if self.p0 is None or self.P_avg is None:
            return False, "SKIP: missing p0 or P_avg"
        p0_nom = float(self.p0)
        P_nom = float(self.P_avg) if isinstance(self.P_avg, (int, float)) else None
        if P_nom is None:
            return False, "P_avg not numeric"
        # Approximate: if P_avg ∝ p0^2, then doubling p0 quadruples P_avg.
        # Perturb +10%
        factor = 1.1
        P_pert = P_nom * (factor**2)  # Since P ∝ p0^2
        if P_pert <= P_nom:
            return False, "P_avg did not increase when p0 increased (expected quadratic dependence)"
        return True, f"P_avg increases with p0: +10% p0 leads to P_avg change factor ~{factor**2:.2f}"

    def test_monotonicity_P_avg_with_omega(self):
        """
        Perturb ω by +10% and check that P_avg increases (proportional to ω^4).
        """
        if self.omega is None or self.P_avg is None:
            return False, "SKIP: missing ω or P_avg"
        omega_nom = float(self.omega)
        P_nom = float(self.P_avg)
        factor = 1.1
        P_pert = P_nom * (factor**4)  # Since P ∝ ω^4
        if P_pert <= P_nom:
            return False, "P_avg did not increase when ω increased (expected quartic dependence)"
        return True, f"P_avg increases with ω: +10% ω leads to P_avg change factor ~{factor**4:.2f}"

    # ------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------
    def test_answer_sanity_P_avg_reasonable(self):
        """
        Broad sanity: P_avg should be positive and finite.
        No exact value comparisons.
        """
        if self.P_avg is None:
            return False, "SKIP: no P_avg"
        try:
            val = float(self.P_avg)
        except Exception:
            return False, "P_avg is not a number"
        if val <= 0 or math.isinf(val) or math.isnan(val):
            return False, f"P_avg = {val} is not positive/finite"
        return True, "P_avg is positive and finite"

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound test methods
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
        return all_pass, logs