import math
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audits a solution to the cosmic ray proton collision problem.
    Extracts relevant variables from the solution_instance and performs
    interface, invariant, modeling, monotonicity, and weak sanity checks.
    """
    def __init__(self, solution_instance: Any):
        # --- Safer: try to get results dictionary, else fallback to empty ---
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict) and 'results' in solution_instance:
            self.results = solution_instance['results']

        # --- Extract givens safely ---
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
        elif isinstance(solution_instance, dict) and 'givens' in solution_instance:
            givens = solution_instance['givens']
        else:
            givens = {}

        self.gamma = givens.get('gamma', None)
        self.p_star = givens.get('p_star', None)   # in GeV/c
        self.m0 = givens.get('m0', None)           # rest energy in GeV (0.14)
        # Also allow m0 in MeV; convert if needed.
        if self.m0 is not None and self.m0 > 1:
            # Assume MeV -> GeV
            self.m0 = self.m0 / 1000.0

        # --- Derived quantities (compute only if givens are present) ---
        if self.gamma is not None:
            self.beta = math.sqrt(1.0 - 1.0/(self.gamma*self.gamma))
        else:
            self.beta = None

        if self.p_star is not None and self.m0 is not None:
            denom = math.sqrt(self.p_star**2 + self.m0**2)
            self.betastar = self.p_star / denom if denom > 0 else None
        else:
            self.betastar = None

        # --- Extract result keys we expect ---
        self.angle_formula = self.results.get('angle_formula', None)
        self.theta_90 = self.results.get('theta_90', None)         # degrees
        self.theta_max = self.results.get('theta_max', None)       # degrees

        # --- Optional: explanation_steps (for guidance only) ---
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps
        elif isinstance(solution_instance, dict) and 'explanation_steps' in solution_instance:
            self.explanation_steps = solution_instance['explanation_steps']
        else:
            self.explanation_steps = None

    # ==================== INTERFACE TESTS ====================
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present."""
        required = ['angle_formula', 'theta_90', 'theta_max']
        missing = [key for key in required if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_provided(self) -> Tuple[bool, str]:
        """Check that essential givens (gamma, p_star, m0) are present."""
        missing = []
        if self.gamma is None:
            missing.append('gamma')
        if self.p_star is None:
            missing.append('p_star')
        if self.m0 is None:
            missing.append('m0')
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens provided")

    # ==================== INVARIANT TESTS ====================
    def test_invariant_beta_close_to_1(self) -> Tuple[bool, str]:
        """For gamma=1000, beta should be very close to 1."""
        if self.beta is None:
            return (False, "SKIP: beta not computed (missing gamma)")
        # Expected beta ~ sqrt(1-1e-6) ~ 0.9999995
        if math.isclose(self.beta, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            return (True, f"beta = {self.beta} is acceptably close to 1")
        else:
            return (False, f"beta = {self.beta} is not close enough to 1")

    def test_invariant_betastar_calculation(self) -> Tuple[bool, str]:
        """Check that betastar is computed correctly from p_star and m0."""
        if self.betastar is None:
            return (False, "SKIP: betastar not computed (missing p_star or m0)")
        # Expected betastar from p_star=0.5 GeV, m0=0.14 GeV: 0.5/sqrt(0.25+0.0196) = 0.5/sqrt(0.2696)=0.5/0.51923=0.9629...
        expected = 0.5 / math.sqrt(0.25 + 0.14**2)  # use given values directly
        if math.isclose(self.betastar, expected, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"betastar = {self.betastar} consistent with p_star and m0")
        else:
            return (False, f"betastar = {self.betastar}, expected ~{expected}")

    def test_invariant_theta_90_tan(self) -> Tuple[bool, str]:
        """Check that tan(theta_90) = betastar / (gamma * beta)."""
        if None in (self.theta_90, self.betastar, self.gamma, self.beta):
            return (False, "SKIP: missing theta_90 or required givens")
        tan_theta = math.tan(math.radians(self.theta_90))
        expected_tan = self.betastar / (self.gamma * self.beta)
        if math.isclose(tan_theta, expected_tan, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"tan(theta_90) = {tan_theta} consistent with betastar/(gamma*beta) = {expected_tan}")
        else:
            return (False, f"tan(theta_90) = {tan_theta}, expected ~{expected_tan}")

    def test_invariant_max_theta_cos_condition(self) -> Tuple[bool, str]:
        """Check that for theta_max, the cos condition holds: cos(thetabar_max) = -betastar/beta."""
        if None in (self.theta_max, self.betastar, self.beta, self.gamma):
            return (False, "SKIP: missing theta_max or required givens")
        # Derive cos from the condition
        cos_star = -self.betastar / self.beta
        if cos_star < -1 or cos_star > 1:
            return (False, f"cos(thetabar_max) = {cos_star} out of range, degenerate case?")
        sin_star = math.sqrt(1 - cos_star**2)
        tan_theta_formula = (self.betastar * sin_star) / (self.gamma * (self.betastar * cos_star + self.beta))
        tan_theta_given = math.tan(math.radians(self.theta_max))
        if math.isclose(tan_theta_formula, tan_theta_given, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"Maximum condition holds: predicted tan(theta_max) = {tan_theta_formula}, given = {tan_theta_given}")
        else:
            return (False, f"Derived tan(theta_max) = {tan_theta_formula} does not match given {tan_theta_given}")

    # ==================== MODELING / TRACE TESTS ====================
    def test_modeling_angle_formula_variables(self) -> Tuple[bool, str]:
        """Check that the angle_formula string contains expected variables."""
        if self.angle_formula is None:
            return (False, "SKIP: angle_formula not present")
        formula = self.angle_formula.lower()
        expected_vars = ['betastar', 'thetabar', 'gamma', 'theta']
        found = [var for var in expected_vars if var in formula or var.replace('_','') in formula]
        if len(found) >= 3:
            return (True, f"Angle formula contains variables: {found}")
        else:
            return (False, f"Angle formula missing some expected variables, found: {found}")

    def test_modeling_theta_90_order_of_magnitude(self) -> Tuple[bool, str]:
        """Check that theta_90 is roughly in the range 10^-3 to 0.5 degrees."""
        if self.theta_90 is None:
            return (False, "SKIP: theta_90 not present")
        if 0.001 < self.theta_90 < 0.5:
            return (True, f"theta_90 = {self.theta_90} deg is plausible")
        else:
            return (False, f"theta_90 = {self.theta_90} deg is outside plausible range (0.001-0.5 deg)")

    # ==================== MONOTONICITY TESTS ====================
    def test_monotonicity_theta_decreases_with_decreasing_gamma(self) -> Tuple[bool, str]:
        """Perturb gamma downward (by -10%) and check that theta_max increases."""
        if None in (self.gamma, self.betastar, self.beta, self.theta_max):
            return (False, "SKIP: missing inputs for monotonicity test")
        # Baseline theta_max in radians
        baseline_theta_max_rad = math.radians(self.theta_max)
        # Perturb gamma: reduce by 10% (gamma smaller -> more relativistic beaming -> larger angle?)
        gamma_perturbed = self.gamma * 0.9
        beta_perturbed = math.sqrt(1.0 - 1.0/(gamma_perturbed*gamma_perturbed))
        # Recompute theta_max using the same betastar
        # First compute thetabar_max from condition
        cos_star = -self.betastar / beta_perturbed
        if abs(cos_star) > 1:
            # If condition yields cos > 1, then the maximum occurs at thetabar=180 deg
            thetabar_max = math.pi
        else:
            thetabar_max = math.acos(cos_star)
        sin_star = math.sin(thetabar_max)
        tan_theta_perturbed = (self.betastar * sin_star) / (gamma_perturbed * (self.betastar * cos_star + beta_perturbed))
        theta_perturbed_rad = math.atan(tan_theta_perturbed) if tan_theta_perturbed > 0 else 0.0
        # We expect that as gamma decreases, the beaming weakens, so theta_max should increase.
        if theta_perturbed_rad > baseline_theta_max_rad:
            return (True, f"Decreasing gamma from {self.gamma} to {gamma_perturbed} increased theta_max from {baseline_theta_max_rad:.6f} rad to {theta_perturbed_rad:.6f} rad")
        else:
            return (False, f"Expected theta_max increase with decreasing gamma, but got {theta_perturbed_rad:.6f} rad <= {baseline_theta_max_rad:.6f} rad")

    # ==================== WEAK ANSWER SANITY (at most one) ====================
    def test_answer_sanity_theta_90_sign_and_finite(self) -> Tuple[bool, str]:
        """Check that theta_90 is positive, finite, and not NaN."""
        if self.theta_90 is None:
            return (False, "SKIP: theta_90 not present")
        if math.isfinite(self.theta_90) and self.theta_90 > 0:
            return (True, f"theta_90 = {self.theta_90} is positive and finite")
        else:
            return (False, f"theta_90 = {self.theta_90} is not positive or finite")

    # ==================== RUN ALL TESTS ====================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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