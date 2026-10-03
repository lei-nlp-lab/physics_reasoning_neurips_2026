import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits the solution for a particle's velocity when its kinetic energy equals its rest energy.
    Primary focus: Lorentz factor consistency, relativistic energy equality, and basic physical sanity.
    """

    def __init__(self, solution_instance):
        # --- Extract results robustly: try attribute first, then results dict, fallback None ---
        self.v = None
        # Try direct attribute
        if hasattr(solution_instance, 'v'):
            self.v = solution_instance.v
        # Try results dictionary
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            if 'v' in solution_instance.results:
                self.v = solution_instance.results['v']
        # Convert to float if possible; keep None otherwise
        self.v_num = None
        if self.v is not None:
            try:
                if isinstance(self.v, sp.Basic):
                    # Check if symbol 'c' is present; if so, substitute with a placeholder (but we cannot assume c)
                    # Instead, try to evaluate without substitution; if fails, try with c=1
                    try:
                        self.v_num = float(self.v)
                    except (TypeError, ValueError, AttributeError):
                        # Attempt symbolic evaluation with c substituted to 1
                        try:
                            self.v_num = float(self.v.subs(sp.Symbol('c'), 1))
                        except:
                            self.v_num = None
                else:
                    self.v_num = float(self.v)
            except (TypeError, ValueError, AttributeError):
                self.v_num = None

        # Extract speed of light c  (NO numeric default)
        self.c = None
        if hasattr(solution_instance, 'c'):
            self.c = solution_instance.c
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            if 'c' in solution_instance.results:
                self.c = solution_instance.results['c']
        self.c_num = None
        if self.c is not None:
            try:
                self.c_num = float(self.c)
            except (TypeError, ValueError):
                self.c_num = None

        # Extract mass m (only used in tests where it cancels; still need to check if provided)
        self.m = None
        if hasattr(solution_instance, 'm'):
            self.m = solution_instance.m
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            if 'm' in solution_instance.results:
                self.m = solution_instance.results['m']
        self.m_num = None
        if self.m is not None:
            try:
                self.m_num = float(self.m)
            except (TypeError, ValueError):
                self.m_num = None

        # Extract explanation_steps (list of strings) and pseudocode (string/dict) for modeling tests
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        # If they are in results dict, also check
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            if 'explanation_steps' in solution_instance.results and self.explanation_steps is None:
                self.explanation_steps = solution_instance.results['explanation_steps']
            if 'pseudocode' in solution_instance.results and self.pseudocode is None:
                self.pseudocode = solution_instance.results['pseudocode']

    def run_all_tests(self):
        """
        Discovers all test_* methods, runs them, and collects results.
        Returns (all_pass, logs_list).
        """
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
        return all_pass, logs

    # -------------------------------------------------------------------------
    # Interface tests: existence and basic type/range checks
    # -------------------------------------------------------------------------
    def test_interface_v_exists(self):
        if self.v is None:
            return False, "Required result 'v' is missing (None)."
        return True, "v is present."

    def test_interface_v_numeric(self):
        if self.v_num is None:
            return False, "v cannot be converted to a numeric value."
        return True, "v is numeric."

    def test_interface_v_positive(self):
        if self.v_num is None:
            return (False, "SKIP: v_num not available, cannot check positivity.")
        if self.v_num <= 0:
            return False, f"v must be positive, got {self.v_num}."
        return True, "v is positive."

    def test_interface_v_finite(self):
        if self.v_num is None:
            return (False, "SKIP: v_num not available.")
        if not math.isfinite(self.v_num):
            return False, f"v is not finite: {self.v_num}."
        return True, "v is finite."

    def test_interface_v_less_than_c(self):
        if self.v_num is None:
            return (False, "SKIP: v_num not available.")
        if self.c_num is None:
            return (False, "SKIP: c not available, cannot check v < c.")
        if self.v_num >= self.c_num:
            return False, f"v ({self.v_num}) must be less than c ({self.c_num})."
        return True, "v < c satisfied."

    # -------------------------------------------------------------------------
    # Invariant tests: theorem-based consistency (use math.isclose, rel_tol=1e-2)
    # -------------------------------------------------------------------------
    def test_invariant_lorentz_factor_from_condition(self):
        """
        The condition K = E0 implies gamma = 2. Compute gamma from v and check it is close to 2.
        """
        if self.v_num is None:
            return (False, "SKIP: v_num not available for Lorentz factor calculation.")
        if self.c_num is None or self.c_num <= 0:
            return (False, "SKIP: c is missing or non-positive, cannot compute gamma.")
        beta = self.v_num / self.c_num
        if beta >= 1:
            return False, f"beta = v/c = {beta} >= 1, gamma would be infinite/complex."
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        if math.isclose(gamma, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"gamma = {gamma} ≈ 2.0 within tolerance."
        else:
            return False, f"gamma = {gamma}, expected 2.0."

    def test_invariant_kinetic_energy_equals_rest_energy(self):
        """
        Compute relativistic KE and rest energy using derived gamma and check equality.
        Since we already verified gamma ≈ 2, this is a redundant consistency check.
        """
        if self.v_num is None:
            return (False, "SKIP: v_num not available.")
        if self.c_num is None or self.c_num <= 0:
            return (False, "SKIP: c missing or non-positive.")
        beta = self.v_num / self.c_num
        if beta >= 1:
            return False, "beta >= 1, invalid."
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        # K = (gamma - 1)*m*c^2, E0 = m*c^2   (m cancels)
        ke_factor = gamma - 1.0
        rest_factor = 1.0
        if math.isclose(ke_factor, rest_factor, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"KE/rest = {ke_factor} ≈ 1.0 => condition satisfied."
        else:
            return False, f"KE/rest = {ke_factor}, expected 1.0."

    def test_invariant_velocity_derived_from_gamma(self):
        """
        Re-derive v from gamma=2: v = c * sqrt(1 - 1/gamma^2) = c * sqrt(3/4).
        Then compare with given v. This uses theorem constraints to check consistency.
        """
        if self.v_num is None:
            return (False, "SKIP: v_num not available.")
        if self.c_num is None or self.c_num <= 0:
            return (False, "SKIP: c missing or non-positive.")
        gamma_target = 2.0
        beta_expected = math.sqrt(1.0 - 1.0 / (gamma_target * gamma_target))  # sqrt(3)/2 ≈ 0.866025...
        v_expected = beta_expected * self.c_num
        if math.isclose(self.v_num, v_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"v = {self.v_num} matches derived value from gamma=2: {v_expected}."
        else:
            return False, f"v = {self.v_num}, but derived v should be {v_expected}."

    # -------------------------------------------------------------------------
    # Modeling/trace tests: check quality of usage of relativistic formulas
    # -------------------------------------------------------------------------
    def test_modeling_uses_relativistic_kinetic_energy(self):
        """
        If explanation_steps or pseudocode is available, verify that the correct formula is referenced.
        If absent, skip. Check that at least one step mentions 'gamma' or 'Lorentz factor'.
        """
        text = ""
        if self.explanation_steps is not None and isinstance(self.explanation_steps, list):
            text = " ".join(self.explanation_steps).lower()
        if self.pseudocode is not None:
            text += " " + str(self.pseudocode).lower()
        if not text:
            return (False, "SKIP: No explanation_steps or pseudocode available to verify modeling.")
        if "gamma" in text or "lorentz" in text:
            return True, "Relativistic kinetic energy (with gamma) is referenced."
        else:
            return False, "Neither 'gamma' nor 'Lorentz' found in explanation/pseudocode."

    def test_modeling_equality_condition_set_correctly(self):
        """
        Check that the condition K = E0 is explicitly mentioned in explanation steps.
        """
        if self.explanation_steps is None or not isinstance(self.explanation_steps, list):
            return (False, "SKIP: No explanation_steps available.")
        text = " ".join(self.explanation_steps).lower()
        if "rest energy" in text and "kinetic energy" in text and ("equals" in text or "equal" in text):
            return True, "Equality condition mentioned."
        else:
            return False, "Equality condition between KE and rest energy not found in explanation."

    # -------------------------------------------------------------------------
    # Monotonicity test: perturb v and check that gamma increases
    # -------------------------------------------------------------------------
    def test_monotonicity_ke_increases_with_gamma(self):
        """
        For a given v, gamma increases with v. Check that if we increase v slightly,
        (gamma-1) increases. Since we have only one v, we simulate a small perturbation.
        """
        if self.v_num is None or self.c_num is None or self.c_num <= 0:
            return (False, "SKIP: v_num or c missing.")
        beta = self.v_num / self.c_num
        if beta >= 0.99:  # avoid numerical issues near c
            return (False, "SKIP: v too close to c for safe perturbation.")
        gamma0 = 1.0 / math.sqrt(1.0 - beta * beta)
        # perturb v upward by 1%
        v1 = self.v_num * 1.01
        if v1 >= self.c_num:
            return (False, "SKIP: perturbed v would exceed c.")
        beta1 = v1 / self.c_num
        gamma1 = 1.0 / math.sqrt(1.0 - beta1 * beta1)
        if gamma1 > gamma0:
            return True, f"Increasing v increases gamma (gamma0={gamma0:.4f}, gamma1={gamma1:.4f})."
        else:
            return False, f"Monotonicity violated: gamma did not increase (gamma0={gamma0}, gamma1={gamma1})."

    # -------------------------------------------------------------------------
    # Weak answer sanity check (exactly one, secondary)
    # -------------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Only check that v is a plausible relativistic velocity: positive, finite, less than c.
        No derivation or isclose involved.
        """
        if self.v_num is None:
            return (False, "SKIP: v not numeric.")
        if self.v_num <= 0:
            return False, "v is not positive."
        if not math.isfinite(self.v_num):
            return False, "v is not finite."
        if self.c_num is not None and self.v_num >= self.c_num:
            return False, "v is not less than c."
        return True, "Answer passes basic physical sanity (positive, finite, <c)."

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Dummy solution instance with a placeholder v
    class DummySolution:
        def __init__(self, v_val):
            self.results = {"v": v_val}
            self.explanation_steps = [
                "Use relativistic kinetic energy: K = (gamma - 1)*m*c^2",
                "Set K equal to rest energy: K = m*c^2",
                "Solve gamma - 1 = 1 -> gamma = 2",
                "From gamma = 1/sqrt(1-v^2/c^2), get v = c*sqrt(3)/2"
            ]
            self.pseudocode = "results['v'] = (sqrt(3)/2)*c"

    # Test with correct v = sqrt(3)/2 * c, c=1
    c = 1.0
    correct_v = math.sqrt(3)/2 * c
    sol = DummySolution(correct_v)
    # Add c attribute so tests can use it (c is provided in the solution)
    sol.c = c
    checker = PhysicsChecker(sol)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")

    # Test with a wrong v (e.g., v=0.5c)
    sol2 = DummySolution(0.5 * c)
    sol2.c = c
    checker2 = PhysicsChecker(sol2)
    success2, logs2 = checker2.run_all_tests()
    for log in logs2:
        print(log)
    print(f"\nOverall success: {success2}")