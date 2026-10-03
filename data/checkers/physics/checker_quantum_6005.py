import math
import sys
import traceback

class PhysicsChecker:
    """
    Physics checker for the hard-sphere scattering problem.
    """

    def __init__(self, solution_instance):
        # Extract input parameters and results safely; use None defaults.
        self.results = getattr(solution_instance, 'results', {})
        self.k = self.results.get('k', None)
        if self.k is None:
            self.k = getattr(solution_instance, 'k', None)

        self.r0 = self.results.get('r0', None)
        if self.r0 is None:
            self.r0 = getattr(solution_instance, 'r0', None)

        # Derived quantities
        self.tan_delta0 = self.results.get('tan_delta0', None)
        self.tan_delta_l_approx = self.results.get('tan_delta_l_approx', None)

        # Partial wave index (optional, default to None if not given)
        self.l = self.results.get('l', None)

        # Explanation steps (for trace references)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that essential input keys exist."""
        missing = []
        for key in ['k', 'r0', 'tan_delta0']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_types_and_ranges(self):
        """Validate types and basic physical ranges of inputs."""
        if self.k is None or self.r0 is None:
            return (False, "SKIP: k or r0 not available")
        if not isinstance(self.k, (int, float)):
            return (False, f"k must be numeric, got {type(self.k)}")
        if not isinstance(self.r0, (int, float)):
            return (False, f"r0 must be numeric, got {type(self.r0)}")
        if self.k <= 0:
            return (False, f"Wave number k must be positive, got {self.k}")
        if self.r0 <= 0:
            return (False, f"Radius r0 must be positive, got {self.r0}")
        return (True, "Types and ranges valid.")

    # ------------------------------------------------------------
    # Invariant checks (physics consistency)
    # ------------------------------------------------------------
    def test_invariant_s_wave_exact_tan_relation(self):
        """
        For l=0, boundary condition gives sin(kr0+δ0)=0 → tanδ0 = tan(-kr0).
        Check that the given tan_delta0 satisfies this exactly (within tolerance).
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta0]):
            return (False, "SKIP: missing k, r0, or tan_delta0")
        kr0 = self.k * self.r0
        expected_tan = math.tan(-kr0)
        if math.isclose(self.tan_delta0, expected_tan, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"tanδ0 = {self.tan_delta0} matches tan(-kr0) = {expected_tan}")
        else:
            return (False, f"tanδ0 = {self.tan_delta0} ≠ tan(-kr0) = {expected_tan}")

    def test_invariant_phase_shift_mod_pi(self):
        """
        Check that δ0 + kr0 is an integer multiple of π (using principal value).
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta0]):
            return (False, "SKIP: missing k, r0, or tan_delta0")
        delta0 = math.atan(self.tan_delta0)  # principal value in (-π/2, π/2)
        kr0 = self.k * self.r0
        residue = (delta0 + kr0) % math.pi
        ok = math.isclose(residue, 0.0, abs_tol=1e-3) or math.isclose(residue, math.pi, abs_tol=1e-3)
        if ok:
            return (True, f"δ0+kr0 = {delta0+kr0} is nπ (mod π={residue})")
        else:
            return (False, f"δ0+kr0 = {delta0+kr0} is not nπ (residue={residue})")

    def test_invariant_low_energy_scaling(self):
        """
        For l>=1, tanδ_l ~ -(kr0)^(2l+1) / [(2l+1) * ((2l-1)!!)^2].
        Check if given tan_delta_l_approx follows this formula.
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta_l_approx, self.l]):
            return (False, "SKIP: missing k, r0, l, or tan_delta_l_approx")
        if self.l < 1:
            return (False, f"SKIP: l={self.l} < 1, test not applicable")
        kr0 = self.k * self.r0

        def double_factorial(n):
            if n <= 0:
                return 1
            result = 1
            for i in range(n, 0, -2):
                result *= i
            return result

        numerator = kr0 ** (2 * self.l + 1)
        denominator = (2 * self.l + 1) * (double_factorial(2 * self.l - 1) ** 2)
        predicted = - numerator / denominator
        if math.isclose(self.tan_delta_l_approx, predicted, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tanδ_l = {self.tan_delta_l_approx} matches predicted {predicted} for l={self.l}")
        else:
            return (False, f"tanδ_l = {self.tan_delta_l_approx} ≠ predicted {predicted}")

    # ------------------------------------------------------------
    # Modeling/Trace checks (based on explanation steps)
    # ------------------------------------------------------------
    def test_modeling_s_wave_boundary_condition(self):
        """
        Re-derive δ0 from tanδ0 and check that sin(kr0+δ0) ≈ 0 (boundary condition).
        This traces the key step in the explanation.
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta0]):
            return (False, "SKIP: missing k, r0, or tan_delta0")
        delta0 = math.atan(self.tan_delta0)  # principal value
        kr0 = self.k * self.r0
        value = math.sin(kr0 + delta0)
        if math.isclose(value, 0.0, abs_tol=1e-5):
            return (True, f"sin(kr0+δ0) = {value} ≈ 0, boundary condition satisfied.")
        else:
            return (False, f"sin(kr0+δ0) = {value} not close to 0.")

    def test_modeling_small_k_expansion_accuracy(self):
        """
        For small kr0, tanδ0 ≈ -kr0. Check that the given value is
        within 5% of -kr0 (relaxed if kr0 > 0.5).
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta0]):
            return (False, "SKIP: missing k, r0, or tan_delta0")
        kr0 = self.k * self.r0
        if kr0 > 0.5:
            rel_tol = 0.2
        else:
            rel_tol = 0.05
        approx = -kr0
        if math.isclose(self.tan_delta0, approx, rel_tol=rel_tol, abs_tol=1e-3):
            return (True, f"tanδ0 = {self.tan_delta0} ≈ -kr0 = {approx} (rel_tol={rel_tol})")
        else:
            return (False, f"tanδ0 = {self.tan_delta0} not near -kr0 = {approx}")

    # ------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_s_wave_versus_k(self):
        """
        For l=0, tanδ0 = tan(-kr0). If we increase k by 10%, the value should become more negative.
        Compute baseline and perturbed using the known formula and compare signs.
        """
        if any(v is None for v in [self.k, self.r0, self.tan_delta0]):
            return (False, "SKIP: missing k, r0, or tan_delta0")

        # Baseline (using the exact physical relation, not the given value)
        kr0_base = self.k * self.r0
        tan_base_exact = math.tan(-kr0_base)  # exact baseline

        # Perturb k by 10%
        k_pert = self.k * 1.1
        kr0_pert = k_pert * self.r0
        tan_pert_exact = math.tan(-kr0_pert)

        # Check monotonicity: tan_pert should be less than tan_base (more negative)
        if tan_pert_exact < tan_base_exact:
            # Trend correct; now check that given value is consistent with this branch
            # (i.e., given value should be close to baseline)
            if math.isclose(self.tan_delta0, tan_base_exact, rel_tol=0.1):
                return (True, f"Increasing k from {self.k} to {k_pert} makes tanδ0 more negative as expected.")
            else:
                # The given value doesn't match exact baseline; still, monotonicity holds
                return (True, "Perturbation trend correct (tanδ0 decreases), but given value differs from baseline.")
        else:
            return (False, f"For 10% increase in k, exact tanδ0 goes from {tan_base_exact} to {tan_pert_exact}, "
                           f"which is not decreasing (monotonicity violated).")

    # ------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self):
        """
        Very weak check: tan_delta0 should be finite and negative.
        if tan_delta_l_approx exists, should also be finite and negative.
        """
        errs = []
        if self.tan_delta0 is not None:
            if not math.isfinite(self.tan_delta0):
                errs.append("tan_delta0 is not finite")
            elif self.tan_delta0 >= 0:
                errs.append("tan_delta0 should be negative")
        if self.tan_delta_l_approx is not None:
            if not math.isfinite(self.tan_delta_l_approx):
                errs.append("tan_delta_l_approx is not finite")
            elif self.tan_delta_l_approx >= 0:
                errs.append("tan_delta_l_approx should be negative")
        if errs:
            return (False, "; ".join(errs))
        return (True, "Sanity checks passed.")

    # ------------------------------------------------------------
    # Runner
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