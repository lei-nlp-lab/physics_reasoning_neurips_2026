import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the two-atom partition function problem.
    Extracts results and givens safely, then runs focused tests.
    """

    def __init__(self, solution_instance):
        # Extract results (expected keys from emit contract)
        self.Z_dist = getattr(solution_instance, 'Z_dist', None) or \
                      solution_instance.results.get('Z_dist', None)
        self.Z_cl_indist = getattr(solution_instance, 'Z_cl_indist', None) or \
                           solution_instance.results.get('Z_cl_indist', None)
        self.Z_FD = getattr(solution_instance, 'Z_FD', None) or \
                    solution_instance.results.get('Z_FD', None)
        self.Z_BE = getattr(solution_instance, 'Z_BE', None) or \
                    solution_instance.results.get('Z_BE', None)

        # Givens (may be in solution_instance directly or in a givens dict)
        self.beta = getattr(solution_instance, 'beta', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        if self.beta is None and hasattr(solution_instance, 'givens'):
            self.beta = solution_instance.givens.get('beta', None)
        if self.epsilon is None and hasattr(solution_instance, 'givens'):
            self.epsilon = solution_instance.givens.get('epsilon', None)

        # If givens not provided, use symbolic defaults for internal checks
        self._beta_sym = self.beta if self.beta is not None else sp.symbols('beta', positive=True)
        self._epsilon_sym = self.epsilon if self.epsilon is not None else sp.symbols('epsilon', positive=True)

        # Helper: numeric substitution point (beta=1, epsilon=1)
        self._num_subs = {self._beta_sym: 1.0, self._epsilon_sym: 1.0}

        # Precompute single-particle partition function Z1 for internal use
        self.Z1 = 1 + sp.exp(-self._beta_sym * self._epsilon_sym) + sp.exp(-2 * self._beta_sym * self._epsilon_sym)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_Z_dist_exists(self):
        if self.Z_dist is None:
            return (False, "Z_dist is missing")
        return (True, "Z_dist present")

    def test_interface_Z_cl_indist_exists(self):
        if self.Z_cl_indist is None:
            return (False, "Z_cl_indist is missing")
        return (True, "Z_cl_indist present")

    def test_interface_Z_FD_exists(self):
        if self.Z_FD is None:
            return (False, "Z_FD is missing")
        return (True, "Z_FD present")

    def test_interface_Z_BE_exists(self):
        if self.Z_BE is None:
            return (False, "Z_BE is missing")
        return (True, "Z_BE present")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose on numeric values)
    # ----------------------------------------------------------------------
    def test_invariant_classical_indist_is_half_of_dist(self):
        if any(v is None for v in (self.Z_dist, self.Z_cl_indist)):
            return (False, "SKIP: missing Z_dist or Z_cl_indist")
        try:
            # Evaluate at numeric point
            num_dist = complex(self.Z_dist.subs(self._num_subs).evalf())  # complex in case of sympy re/im? should be real
            num_indist = complex(self.Z_cl_indist.subs(self._num_subs).evalf())
            # Should be real (imag part ~0)
            if abs(num_dist.imag) > 1e-10 or abs(num_indist.imag) > 1e-10:
                return (False, "Expressions gave complex results unexpectedly")
            val_dist = num_dist.real
            val_indist = num_indist.real
            if math.isclose(val_indist, val_dist / 2.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Z_cl_indist ≈ Z_dist/2")
            else:
                return (False, f"Expected Z_cl_indist ≈ {val_dist/2}, got {val_indist}")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

    def test_invariant_FD_less_than_BE(self):
        if any(v is None for v in (self.Z_FD, self.Z_BE)):
            return (False, "SKIP: missing Z_FD or Z_BE")
        try:
            num_FD = complex(self.Z_FD.subs(self._num_subs).evalf())
            num_BE = complex(self.Z_BE.subs(self._num_subs).evalf())
            if abs(num_FD.imag) > 1e-10 or abs(num_BE.imag) > 1e-10:
                return (False, "Expressions gave complex results unexpectedly")
            val_FD = num_FD.real
            val_BE = num_BE.real
            if val_FD <= val_BE:
                return (True, f"Z_FD ({val_FD}) <= Z_BE ({val_BE}) holds")
            else:
                return (False, f"Z_FD ({val_FD}) > Z_BE ({val_BE}) violated")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

    def test_invariant_dist_geq_classical_indist(self):
        if any(v is None for v in (self.Z_dist, self.Z_cl_indist)):
            return (False, "SKIP: missing Z_dist or Z_cl_indist")
        try:
            num_dist = complex(self.Z_dist.subs(self._num_subs).evalf()).real
            num_indist = complex(self.Z_cl_indist.subs(self._num_subs).evalf()).real
            if num_dist >= num_indist:
                return (True, f"Z_dist ({num_dist}) >= Z_cl_indist ({num_indist}) holds")
            else:
                return (False, f"Z_dist < Z_cl_indist: {num_dist} < {num_indist}")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

    # ----------------------------------------------------------------------
    # Modeling checks (trace consistency, use symbolic relations)
    # ----------------------------------------------------------------------
    def test_modeling_dist_is_Z1_squared(self):
        if self.Z_dist is None:
            return (False, "SKIP: Z_dist missing")
        try:
            # Expected: Z1^2
            expected = self.Z1 ** 2
            diff = sp.simplify(self.Z_dist - expected)
            # Check if diff is zero symbolically
            if diff == 0:
                return (True, "Z_dist exactly equals Z1^2")
            # Fallback: numeric check
            num_diff = complex(diff.subs(self._num_subs).evalf())
            if abs(num_diff.real) < 1e-10 and abs(num_diff.imag) < 1e-10:
                return (True, "Z_dist numerically equals Z1^2")
            else:
                return (False, f"Difference not zero: {num_diff}")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_modeling_FD_equals_Z1_times_exp(self):
        if self.Z_FD is None:
            return (False, "SKIP: Z_FD missing")
        try:
            # Expected: Z1 * exp(-beta*epsilon)
            expected = self.Z1 * sp.exp(-self._beta_sym * self._epsilon_sym)
            diff = sp.simplify(self.Z_FD - expected)
            if diff == 0:
                return (True, "Z_FD exactly equals Z1 * exp(-beta*epsilon)")
            num_diff = complex(diff.subs(self._num_subs).evalf())
            if abs(num_diff.real) < 1e-10 and abs(num_diff.imag) < 1e-10:
                return (True, "Z_FD numerically equals Z1 * exp(-beta*epsilon)")
            else:
                return (False, f"Difference not zero: {num_diff}")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_modeling_BE_equals_Z1_times_1_plus_exp_minus2(self):
        if self.Z_BE is None:
            return (False, "SKIP: Z_BE missing")
        try:
            # Expected: Z1 * (1 + exp(-2*beta*epsilon))
            expected = self.Z1 * (1 + sp.exp(-2 * self._beta_sym * self._epsilon_sym))
            diff = sp.simplify(self.Z_BE - expected)
            if diff == 0:
                return (True, "Z_BE exactly equals Z1*(1+exp(-2*beta*epsilon))")
            num_diff = complex(diff.subs(self._num_subs).evalf())
            if abs(num_diff.real) < 1e-10 and abs(num_diff.imag) < 1e-10:
                return (True, "Z_BE numerically equals Z1*(1+exp(-2*beta*epsilon))")
            else:
                return (False, f"Difference not zero: {num_diff}")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_modeling_classical_indist_is_half_of_dist(self):
        # Same as invariant but modeling check uses symbolic relation
        if any(v is None for v in (self.Z_dist, self.Z_cl_indist)):
            return (False, "SKIP: missing Z_dist or Z_cl_indist")
        try:
            expected = self.Z_dist / 2
            diff = sp.simplify(self.Z_cl_indist - expected)
            if diff == 0:
                return (True, "Z_cl_indist exactly equals Z_dist/2")
            num_diff = complex(diff.subs(self._num_subs).evalf())
            if abs(num_diff.real) < 1e-10 and abs(num_diff.imag) < 1e-10:
                return (True, "Z_cl_indist numerically equals Z_dist/2")
            else:
                return (False, f"Difference not zero: {num_diff}")
        except Exception as e:
            return (False, f"Error: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb beta)
    # ----------------------------------------------------------------------
    def _evaluate_numeric(self, expr, beta_scale=1.0):
        """Evaluate expr with beta scaled by beta_scale (e.g., 0.9 for lower temperature)"""
        subs = self._num_subs.copy()
        # Scale beta: new_beta = beta * scale (beta larger -> lower T)
        # But we want to check monotonicity with temperature: higher T -> larger Z.
        # Higher T => smaller beta. So we can scale beta down to simulate higher T.
        subs[self._beta_sym] = self._num_subs[self._beta_sym] * beta_scale
        val = complex(expr.subs(subs).evalf())
        if abs(val.imag) > 1e-10:
            raise ValueError("Complex result")
        return val.real

    def test_monotonicity_Z_dist_increases_with_temperature(self):
        if self.Z_dist is None:
            return (False, "SKIP: Z_dist missing")
        try:
            # Baseline at beta=1
            base = self._evaluate_numeric(self.Z_dist, beta_scale=1.0)
            # Increase temperature: smaller beta, e.g., scale=0.8
            perturbed = self._evaluate_numeric(self.Z_dist, beta_scale=0.8)
            if perturbed > base:
                return (True, f"Z_dist increases when beta decreases: {base} -> {perturbed}")
            else:
                return (False, f"Monotonicity violated: base={base}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_monotonicity_Z_BE_increases_with_temperature(self):
        if self.Z_BE is None:
            return (False, "SKIP: Z_BE missing")
        try:
            base = self._evaluate_numeric(self.Z_BE, beta_scale=1.0)
            perturbed = self._evaluate_numeric(self.Z_BE, beta_scale=0.8)
            if perturbed > base:
                return (True, f"Z_BE increases when beta decreases: {base} -> {perturbed}")
            else:
                return (False, f"Monotonicity violated: base={base}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Error: {e}")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no numeric matching, only broad checks)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self):
        """Check that all partition functions are finite and positive at numerical point."""
        results = {
            'Z_dist': self.Z_dist,
            'Z_cl_indist': self.Z_cl_indist,
            'Z_FD': self.Z_FD,
            'Z_BE': self.Z_BE
        }
        failed = []
        for name, expr in results.items():
            if expr is None:
                failed.append(f"{name} missing")
                continue
            try:
                val = complex(expr.subs(self._num_subs).evalf())
                real_val = val.real
                if abs(val.imag) > 1e-10:
                    failed.append(f"{name} is complex")
                elif not math.isfinite(real_val):
                    failed.append(f"{name} is not finite")
                elif real_val <= 0:
                    failed.append(f"{name} is not positive (got {real_val})")
            except Exception as e:
                failed.append(f"{name} evaluation error: {e}")
        if failed:
            return (False, "; ".join(failed))
        return (True, "All Z values finite and positive")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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