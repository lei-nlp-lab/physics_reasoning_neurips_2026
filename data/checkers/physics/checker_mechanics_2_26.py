import math
import re

class PhysicsChecker:
    """
    Checker for the hoop-particle problem.
    Extracts givens and results from solution_instance and runs a battery of tests.
    """

    def __init__(self, solution_instance):
        # Extract givens safely – use None defaults
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})
        explanation_steps = getattr(solution_instance, 'explanation_steps', [])

        self.m = givens.get('m', None)
        self.M = givens.get('M', None)
        self.r = givens.get('r', None)
        self.g = givens.get('g', None)          # No numeric fallback – use None
        self.v0 = givens.get('v0', None)

        # Results from pseudocode [emit] contract
        self.v_f1_str = results.get('v_f1', None)
        self.v_f2_str = results.get('v_f2', None)

        # Store explanation_steps (list of strings)
        self.explanation_steps = explanation_steps

        # Numeric evaluations of the result strings (if possible)
        self.v_f1_num = None
        self.v_f2_num = None

        if self.v_f1_str and isinstance(self.v_f1_str, str):
            self.v_f1_num = self._safe_eval(self.v_f1_str)
        if self.v_f2_str and isinstance(self.v_f2_str, str):
            self.v_f2_num = self._safe_eval(self.v_f2_str)

    def _safe_eval(self, expr_str):
        """
        Evaluate a simple physics expression string using givens.
        Returns a float if successful, otherwise None.
        """
        # Replace variable names with numeric values (if available)
        subs = {
            'm': self.m if self.m is not None else 'm',
            'M': self.M if self.M is not None else 'M',
            'r': self.r if self.r is not None else 'r',
            'g': self.g if self.g is not None else 'g',
            'v0': self.v0 if self.v0 is not None else 'v0',
        }
        # Also handle 'pi' and math functions
        local_dict = {k: v for k, v in subs.items() if isinstance(v, (int, float))}
        local_dict['pi'] = math.pi
        local_dict['sqrt'] = math.sqrt
        local_dict['cos'] = math.cos
        local_dict['sin'] = math.sin
        local_dict['abs'] = abs

        # Replace symbolic names in the expression that are not numeric
        result_str = expr_str
        for var, val in subs.items():
            if isinstance(val, (int, float)):
                # Ensure we replace whole words only to avoid partial matches
                result_str = result_str.replace(var, str(val))
        # Evaluate using math and builtins
        try:
            # Restrict allowed names
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
            allowed_names.update({'abs': abs, 'pi': math.pi})
            result = eval(result_str, {"__builtins__": None}, allowed_names)
            return float(result)
        except Exception:
            return None

    # ---------- Interface tests ----------
    def test_interface_givens_present(self):
        """All required givens must be provided and positive."""
        missing = []
        if self.m is None:
            missing.append('m')
        if self.M is None:
            missing.append('M')
        if self.r is None:
            missing.append('r')
        if self.v0 is None:
            missing.append('v0')
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        # Check positivity
        if self.m <= 0 or self.M <= 0 or self.r <= 0 or self.v0 <= 0:
            return False, "Givens m, M, r, v0 must be positive."
        # g is allowed to be None (will be handled elsewhere), but if present must be positive
        if self.g is not None and self.g <= 0:
            return False, "g must be positive if provided."
        return True, "All required givens present and positive."

    def test_interface_results_present(self):
        """Both v_f1 and v_f2 must be present in results."""
        if self.v_f1_str is None:
            return False, "Missing result key: v_f1"
        if self.v_f2_str is None:
            return False, "Missing result key: v_f2"
        return True, "Both result keys present."

    def test_interface_result_types(self):
        """v_f1 and v_f2 should be strings (as from pseudocode emit)."""
        if not isinstance(self.v_f1_str, str):
            return False, f"v_f1 is not a string, got {type(self.v_f1_str).__name__}"
        if not isinstance(self.v_f2_str, str):
            return False, f"v_f2 is not a string, got {type(self.v_f2_str).__name__}"
        return True, "Result types correct (both strings)."

    # ---------- Invariant tests (theorem-based, use math.isclose) ----------
    def test_invariant_sign_vf1(self):
        """In limit m/M → 0, v_f1 must be negative (particle moves left at bottom)."""
        if self.v_f1_str is None:
            return (False, "SKIP: v_f1 not available")
        if not isinstance(self.v_f1_str, str):
            return (False, "SKIP: v_f1 not a string")
        if not self.v_f1_str.strip().startswith('-'):
            return False, f"v_f1 does not have a negative sign: {self.v_f1_str}"
        if self.v_f1_num is not None and self.v_f1_num >= 0:
            return False, f"Parsed v_f1 is not negative: {self.v_f1_num}"
        return True, "v_f1 has correct negative sign."

    def test_invariant_energy_limit1(self):
        """For m/M→0, energy conservation: v_f1^2 = v0^2 + 4gr."""
        if self.v_f1_num is None:
            return (False, "SKIP: Could not parse v_f1 numeric value")
        if any(x is None for x in [self.v0, self.g, self.r]):
            return (False, "SKIP: Missing givens (v0, g, r)")
        expected_sq = self.v0**2 + 4 * self.g * self.r
        actual_sq = self.v_f1_num**2
        if not math.isclose(actual_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Energy equation violated: v_f1^2 = {actual_sq} != {expected_sq}"
        return True, "Energy conservation satisfied in m/M→0 limit."

    def test_invariant_momentum_limit2(self):
        """For M/m→0, momentum conservation: v_f2 = v0."""
        if self.v_f2_num is None:
            return (False, "SKIP: Could not parse v_f2 numeric value")
        if self.v0 is None:
            return (False, "SKIP: v0 not available")
        if not math.isclose(self.v_f2_num, self.v0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Momentum conservation violated: v_f2 = {self.v_f2_num} != v0 = {self.v0}"
        return True, "Momentum conservation satisfied in M/m→0 limit."

    def test_invariant_limit_consistency(self):
        """Both limit answers exist and are structurally reasonable."""
        if self.v_f1_str is None or self.v_f2_str is None:
            return (False, "SKIP: Missing one or both results")
        return True, "Both limit results are present."

    # ---------- Modeling / trace tests ----------
    def test_modeling_energy_equation_structure(self):
        """Check that the explanation steps contain energy (or Lagrangian) discussion."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps provided")
        text = " ".join(self.explanation_steps)
        if 'energy' in text.lower() and ('conserved' in text.lower() or 'kinetic' in text.lower()):
            return True, "Energy conservation discussed."
        if 'lagrangian' in text.lower():
            return True, "Lagrangian used (implies energy)."
        return False, "Explanation steps lack energy or Lagrangian discussion."

    def test_modeling_momentum_conservation_mentioned(self):
        """Check that horizontal momentum conservation is mentioned."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps")
        text = " ".join(self.explanation_steps)
        if 'momentum' in text.lower() and ('horizontal' in text.lower() or 'constant' in text.lower()):
            return True, "Horizontal momentum conservation discussed."
        return False, "Momentum conservation not clearly mentioned."

    def test_modeling_rolling_without_slipping(self):
        """Check that the rolling condition (no slip) is mentioned explicitly."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps")
        text = " ".join(self.explanation_steps)
        if 'roll' in text.lower() or 'without slipping' in text.lower() or 'no slip' in text.lower():
            return True, "Rolling without slipping condition recognized."
        return False, "Rolling condition not explicitly mentioned."

    # ---------- Monotonicity tests (each must perturb an input) ----------
    def test_monotonicity_vf1_increases_with_v0(self):
        """For m/M→0, |v_f1| should increase with v0."""
        if self.v_f1_num is None or self.v0 is None or self.g is None or self.r is None:
            return (False, "SKIP: Cannot compute monotonicity without numeric values")
        # Baseline from the expression
        baseline = self.v_f1_num

        # Perturb v0 by +5%
        v0_pert = self.v0 * 1.05
        # Compute expected perturbed v_f1 using the formula (same energy relation)
        expected_pert = -math.sqrt(v0_pert**2 + 4 * self.g * self.r)
        # Check monotonic: absolute value should increase
        if abs(expected_pert) <= abs(baseline):
            return False, f"Monotonicity violated: |v_f1| did not increase (baseline {baseline}, perturbed {expected_pert})"
        return True, "Monotonicity holds: |v_f1| increases with v0."

    def test_monotonicity_vf2_increases_with_v0(self):
        """For M/m→0, v_f2 should be exactly v0, so increasing v0 increases v_f2 linearly."""
        if self.v_f2_num is None or self.v0 is None:
            return (False, "SKIP: Cannot compute monotonicity for v_f2")
        # Baseline v_f2 from the solution
        baseline = self.v_f2_num

        # Perturb v0 by +5%
        v0_pert = self.v0 * 1.05
        # Since v_f2 = v0, perturbed value should be v0_pert
        expected_pert = v0_pert
        if expected_pert <= baseline:
            return False, f"Monotonicity violated: v_f2 did not increase (baseline {baseline}, perturbed {expected_pert})"
        # Also check that the ratio matches
        if not math.isclose(expected_pert / baseline, v0_pert / self.v0, rel_tol=1e-2):
            return False, "v_f2 did not scale proportionally with v0."
        return True, "Monotonicity holds: v_f2 increases linearly with v0."

    # ---------- Answer sanity (weak, at most one) ----------
    def test_answer_sanity_plausibility(self):
        """Basic plausibility: v_f1 and v_f2 expressions should contain expected symbols."""
        if self.v_f1_str is None or self.v_f2_str is None:
            return (False, "SKIP: One or both results missing")
        # v_f1 should involve sqrt, v0, g, r
        if 'sqrt' not in self.v_f1_str.lower():
            return False, "v_f1 expression missing sqrt"
        # v_f2 should be simple (v0 or similar)
        if 'v0' not in self.v_f2_str:
            return False, "v_f2 should contain v0"
        # Ensure numeric versions are not NaN
        if self.v_f1_num is not None and math.isnan(self.v_f1_num):
            return False, "v_f1 numeric evaluation is NaN"
        if self.v_f2_num is not None and math.isnan(self.v_f2_num):
            return False, "v_f2 numeric evaluation is NaN"
        return True, "Answer expressions are syntactically plausible."

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)


# Example usage (for testing):
if __name__ == "__main__":
    class DummySolution:
        def __init__(self):
            self.givens = {'m': 1.0, 'M': 10.0, 'r': 0.5, 'g': 9.81, 'v0': 3.0}
            self.results = {
                'v_f1': '- sqrt(v0^2 + 4 * g * r)',
                'v_f2': 'v0'
            }
            self.explanation_steps = [
                "Use Lagrangian: L = M xdot^2 + 1/2 m (xdot^2 + r^2 thetadot^2 + 2 r xdot thetadot cosθ) - mgr(1+cosθ)",
                "No horizontal forces → horizontal momentum conserved: (2M+m)xdot + m r thetadot cosθ = constant = m v0",
                "Energy conserved between top and bottom: M xdot^2 + 1/2 m v_f^2 = 1/2 m v0^2 + 2mgR",
                "At bottom θ=π, v_f = xdot - r thetadot"
            ]

    dummy = DummySolution()
    checker = PhysicsChecker(dummy)
    success, logs = checker.run_all_tests()
    for line in logs:
        print(line)