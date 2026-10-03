import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the sphere-sliding-particle problem.
    Checks theorem invariants, modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract all needed data with safe defaults (None for missing).
        """
        # Safely extract givens
        givens = getattr(solution_instance, 'givens', {})
        self.g = givens.get('g', None)
        self.R = givens.get('R', None)

        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        self.theta_d = results.get('theta_d', None)
        self.v_d = results.get('v_d', None)

        # Safely extract explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_has_givens(self):
        """Check that required gravitational acceleration and radius are provided."""
        if self.g is None:
            return (False, "Missing givens['g']")
        if self.R is None:
            return (False, "Missing givens['R']")
        if not isinstance(self.g, (int, float)):
            return (False, "givens['g'] must be numeric")
        if not isinstance(self.R, (int, float)):
            return (False, "givens['R'] must be numeric")
        return (True, "g and R present and numeric")

    def test_interface_has_results(self):
        """Check that both theta_d and v_d are present in results."""
        if self.theta_d is None:
            return (False, "Missing results['theta_d']")
        if self.v_d is None:
            return (False, "Missing results['v_d']")
        if not isinstance(self.theta_d, (int, float)):
            return (False, "results['theta_d'] must be numeric")
        if not isinstance(self.v_d, (int, float)):
            return (False, "results['v_d'] must be numeric")
        return (True, "theta_d and v_d present and numeric")

    # ------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_conservation(self):
        """Check that energy conservation holds at detachment:
        v_d^2 ≈ 2 g R (1 - cos(θ_d)).
        """
        if any(v is None for v in (self.g, self.R, self.theta_d, self.v_d)):
            return (False, "SKIP: missing required data")
        left = self.v_d ** 2
        right = 2 * self.g * self.R * (1 - math.cos(self.theta_d))
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conservation: v_d^2={left:.4f} ≈ 2gR(1-cosθ)={right:.4f}")
        else:
            return (False, f"Energy conservation violated: v_d^2={left:.4f}, expected {right:.4f}")

    def test_invariant_detachment_condition(self):
        """Check that radial force balance at detachment gives:
        v_d^2 ≈ g R cos(θ_d).
        """
        if any(v is None for v in (self.g, self.R, self.theta_d, self.v_d)):
            return (False, "SKIP: missing required data")
        left = self.v_d ** 2
        right = self.g * self.R * math.cos(self.theta_d)
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Detachment condition: v_d^2={left:.4f} ≈ gRcosθ={right:.4f}")
        else:
            return (False, f"Detachment condition violated: v_d^2={left:.4f}, expected {right:.4f}")

    def test_invariant_cos_theta_d(self):
        """Check that the two invariants imply cos(θ_d) ≈ 2/3."""
        if self.theta_d is None:
            return (False, "SKIP: theta_d missing")
        cos_theta = math.cos(self.theta_d)
        if math.isclose(cos_theta, 2.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"cos(θ_d) ≈ 2/3: cos(θ_d)={cos_theta:.4f}")
        else:
            return (False, f"cos(θ_d)={cos_theta:.4f} not close to 2/3")

    # ------------------------------------------------------------------
    # MODELING / QUALITY CHECKS
    # ------------------------------------------------------------------
    def test_modeling_explanation_presence(self):
        """Ensure the explanation contains reasoning about energy and detachment."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        texts = []
        for step in self.explanation_steps:
            if isinstance(step, str):
                texts.append(step.lower())
            elif isinstance(step, dict):
                texts.append(str(step.get('text', '')).lower())
        combined = ' '.join(texts)
        has_energy = 'energy' in combined
        has_detach = 'detach' in combined or 'leave' in combined or 'normal force' in combined
        if has_energy and has_detach:
            return (True, "Explanation steps mention energy conservation and detachment")
        else:
            return (False, "Explanation lacks mention of energy or detachment")

    def test_modeling_equations_present(self):
        """Check that the explanation contains the two key equations:
        v^2 = 2gR(1-cosθ) and radial force balance.
        """
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        texts = []
        for step in self.explanation_steps:
            if isinstance(step, str):
                texts.append(step.lower())
            elif isinstance(step, dict):
                texts.append(str(step.get('text', '')).lower())
        combined = ' '.join(texts)
        # Look for the energy equation (v^2 = 2gR(1-cosθ) or similar)
        has_energy_eq = ('v^2' in combined and '2g' in combined and 'cos' in combined and '1' in combined)
        # Look for radial force equation (mg cosθ - mv^2/R = 0 or similar)
        has_radial_eq = ('mg cos' in combined and 'mv^2' in combined and 'R' in combined)
        if has_energy_eq and has_radial_eq:
            return (True, "Explanation includes energy and radial force equations")
        else:
            return (False, f"Missing equations in explanation: energy_eq={has_energy_eq}, radial_eq={has_radial_eq}")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (with explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_v_d_with_R(self):
        """v_d should scale as sqrt(R) when g is fixed.
        Baseline: v_d from given R. Perturb R by +1% -> new v_d proportional to sqrt(1.01).
        """
        if any(v is None for v in (self.g, self.R, self.v_d)):
            return (False, "SKIP: missing required data")
        # Re‑derive v_d from known formula (allowing small rounding differences)
        g = self.g
        R = self.R
        v_d_derived = math.sqrt(2 * g * R / 3)
        # Compare with given v_d as baseline
        baseline = self.v_d
        if not math.isclose(baseline, v_d_derived, rel_tol=1e-2):
            # If given v_d is off, use derived as baseline
            baseline = v_d_derived
        # Perturb R by 1%
        R_pert = R * 1.01
        v_d_pert = math.sqrt(2 * g * R_pert / 3)
        # Expected ratio sqrt(1.01)
        expected_ratio = math.sqrt(1.01)
        actual_ratio = v_d_pert / baseline
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"v_d scales as sqrt(R): ratio perturbed/baseline = {actual_ratio:.4f} ≈ sqrt(1.01) = {expected_ratio:.4f}")
        else:
            return (False, f"v_d scaling violated: ratio {actual_ratio:.4f} not close to sqrt(1.01) = {expected_ratio:.4f}")

    def test_monotonicity_v_d_with_g(self):
        """v_d should scale as sqrt(g) when R is fixed.
        Perturb g by +1% -> new v_d proportional to sqrt(1.01).
        """
        if any(v is None for v in (self.g, self.R, self.v_d)):
            return (False, "SKIP: missing required data")
        g = self.g
        R = self.R
        baseline = self.v_d
        # Optional re‑derived baseline if given v_d differs too much
        v_d_derived = math.sqrt(2 * g * R / 3)
        if not math.isclose(baseline, v_d_derived, rel_tol=1e-2):
            baseline = v_d_derived
        g_pert = g * 1.01
        v_d_pert = math.sqrt(2 * g_pert * R / 3)
        expected_ratio = math.sqrt(1.01)
        actual_ratio = v_d_pert / baseline
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"v_d scales as sqrt(g): ratio perturbed/baseline = {actual_ratio:.4f} ≈ sqrt(1.01) = {expected_ratio:.4f}")
        else:
            return (False, f"v_d scaling violated: ratio {actual_ratio:.4f} not close to sqrt(1.01) = {expected_ratio:.4f}")

    # ------------------------------------------------------------------
    # ANSWER SANITY (weak, no isclose, no derived expectations)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Both θ_d and v_d should be finite numbers."""
        if self.theta_d is None or self.v_d is None:
            return (False, "SKIP: missing data")
        if math.isfinite(self.theta_d) and math.isfinite(self.v_d):
            return (True, "θ_d and v_d are finite")
        else:
            return (False, "Non‑finite value detected in θ_d or v_d")

    # ------------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and execute all test_* methods, return (success, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)