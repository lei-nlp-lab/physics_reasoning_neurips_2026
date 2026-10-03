import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the problem of a charged insulating cylinder in an applied electric field.
    Checks focus on:
    - Interface: presence and types of required keys.
    - Invariants: physical relationships that must hold.
    - Modeling: quality of derived formulas.
    - Monotonicity: expected trends under perturbations.
    - Weak answer sanity: basic checks on final outputs (single combined sanity test).
    """
    
    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        self.results: Dict[str, Any] = (
            solution_instance.results 
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict) 
            else {}
        )
        
        # ----- Results (emitted in pseudocode) -----
        self.E_th: float = self.results.get('E_th', None)
        self.sigma_theta: float = self.results.get('sigma_theta', None)
        self.E_max: float = self.results.get('E_max', None)
        
        # ----- Givens (inputs to the problem) -----
        # Try to extract from solution_instance attributes or from results
        self.sigma0: float = getattr(solution_instance, 'sigma0', 
                                      self.results.get('sigma0', None))
        self.epsilon0: float = getattr(solution_instance, 'epsilon0', 
                                        self.results.get('epsilon0', None))
        self.E_a: float = getattr(solution_instance, 'E_a', 
                                   self.results.get('E_a', None))
        self.theta: float = getattr(solution_instance, 'theta', 
                                     self.results.get('theta', None))  # allow None, do not default to 0
        
        # Mark whether we have enough data for detailed checks
        self._has_givens = all(v is not None for v in [self.sigma0, self.epsilon0, self.E_a])
    
    # ---------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ---------------------------------------------------------------
    
    def test_interface_E_th_exists(self) -> Tuple[bool, str]:
        """Check that E_th is present in results and is numeric."""
        if self.E_th is None:
            return (False, "SKIP: E_th not in results or is None")
        if not isinstance(self.E_th, (int, float)):
            return (False, f"E_th is not numeric: {type(self.E_th)}")
        return (True, "E_th is present and numeric")
    
    def test_interface_sigma_theta_exists(self) -> Tuple[bool, str]:
        """Check that sigma_theta is present and numeric."""
        if self.sigma_theta is None:
            return (False, "SKIP: sigma_theta not in results or is None")
        if not isinstance(self.sigma_theta, (int, float)):
            return (False, f"sigma_theta is not numeric: {type(self.sigma_theta)}")
        return (True, "sigma_theta is present and numeric")
    
    def test_interface_E_max_exists(self) -> Tuple[bool, str]:
        """Check that E_max is present and numeric."""
        if self.E_max is None:
            return (False, "SKIP: E_max not in results or is None")
        if not isinstance(self.E_max, (int, float)):
            return (False, f"E_max is not numeric: {type(self.E_max)}")
        return (True, "E_max is present and numeric")
    
    # ---------------------------------------------------------------
    # Invariant tests: physical relationships that must hold
    # ---------------------------------------------------------------
    
    def test_invariant_sigma_theta_negative_if_valid(self) -> Tuple[bool, str]:
        """
        For a valid solution (E_a ≤ E_th), the total charge density sigma_theta must be ≤ 0.
        (Only negative charges present – electrons.)
        """
        if not self._has_givens:
            return (False, "SKIP: missing givens (sigma0, epsilon0, E_a)")
        if self.E_th is None:
            return (False, "SKIP: E_th not provided")
        
        # If E_a exceeds E_th, the model may break down; still check consistency if within range
        if self.E_a > self.E_th + 1e-9:
            return (True, "E_a > E_th, model not applicable; skipping charge sign check (can be positive)")
        
        # sigma_theta should be ≤ 0 for the insulating case
        if self.sigma_theta is not None and self.sigma_theta > 1e-9:
            return (False, f"Expected sigma_theta ≤ 0, got {self.sigma_theta}")
        return (True, "sigma_theta is non-positive (correct) or sigma_theta not available")
    
    def test_invariant_E_max_equals_E_th(self) -> Tuple[bool, str]:
        """
        Physics mapping says E_max = E_th (threshold field).
        Check they are approximately equal.
        """
        if self.E_max is None or self.E_th is None:
            return (False, "SKIP: E_max or E_th missing")
        if math.isclose(self.E_max, self.E_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_max = {self.E_max} ≈ E_th = {self.E_th}")
        else:
            return (False, f"E_max ({self.E_max}) differs from E_th ({self.E_th})")
    
    def test_invariant_sigma_theta_formula(self) -> Tuple[bool, str]:
        """
        Check that sigma_theta satisfies the theoretical formula:
        sigma_theta = sigma0 + 2*epsilon0*E_a*cos(theta)
        (if all givens are present). Theta may be None – skip in that case.
        """
        if not self._has_givens:
            return (False, "SKIP: missing givens (sigma0, epsilon0, E_a)")
        if self.theta is None:
            return (False, "SKIP: theta not provided")
        if self.sigma_theta is None:
            return (False, "SKIP: sigma_theta not in results")
        
        expected = self.sigma0 + 2 * self.epsilon0 * self.E_a * math.cos(self.theta)
        if math.isclose(self.sigma_theta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_theta = {self.sigma_theta} matches expected {expected}")
        else:
            return (False, f"sigma_theta = {self.sigma_theta} ≠ expected {expected}")
    
    # ---------------------------------------------------------------
    # Modeling / trace consistency tests
    # ---------------------------------------------------------------
    
    def test_modeling_E_th_derived_from_givens(self) -> Tuple[bool, str]:
        """
        Check that E_th = -sigma0/(2*epsilon0) correctly.
        This is a derived quantity: we can compute from given sigma0 and epsilon0.
        """
        if not self._has_givens:
            return (False, "SKIP: missing givens (sigma0, epsilon0)")
        if self.E_th is None:
            return (False, "SKIP: E_th not in results")
        
        expected_th = -self.sigma0 / (2 * self.epsilon0)
        if math.isclose(self.E_th, expected_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_th = {self.E_th} matches derived {expected_th}")
        else:
            return (False, f"E_th = {self.E_th} ≠ derived {expected_th}")
    
    def test_modeling_sigma0_negative(self) -> Tuple[bool, str]:
        """
        Check that sigma0 is indeed negative (electrons, initial negative charge).
        """
        if self.sigma0 is None:
            return (False, "SKIP: sigma0 missing")
        if self.sigma0 >= 0:
            return (False, f"sigma0 should be negative, got {self.sigma0}")
        return (True, "sigma0 is negative (correct)")
    
    def test_modeling_epsilon0_positive(self) -> Tuple[bool, str]:
        """
        Check that epsilon0 (vacuum permittivity) is positive.
        """
        if self.epsilon0 is None:
            return (False, "SKIP: epsilon0 missing")
        if self.epsilon0 <= 0:
            return (False, f"epsilon0 must be positive, got {self.epsilon0}")
        return (True, "epsilon0 is positive")
    
    # ---------------------------------------------------------------
    # Monotonicity tests: expected trends under perturbations
    # ---------------------------------------------------------------
    
    def test_monotonicity_sigma_theta_increases_with_E_a(self) -> Tuple[bool, str]:
        """
        For fixed theta, sigma_theta should increase (become less negative) as E_a increases.
        Perturb E_a by +10% and compare.
        Only applicable when cos(theta) > 0.
        """
        if not self._has_givens or self.sigma_theta is None:
            return (False, "SKIP: missing data to compute perturbation")
        if self.theta is None:
            return (False, "SKIP: theta not provided")
        
        theta_rad = self.theta
        base = self.sigma0 + 2 * self.epsilon0 * self.E_a * math.cos(theta_rad)
        
        # Perturb E_a upwards by 10%
        E_a_pert = self.E_a * 1.1
        perturbed = self.sigma0 + 2 * self.epsilon0 * E_a_pert * math.cos(theta_rad)
        
        # Check monotonicity only if cos(theta) > 0
        if math.cos(theta_rad) > 1e-9:
            if perturbed > base:
                return (True, f"sigma_theta increases with E_a (cosθ>0): base={base}, perturbed={perturbed}")
            else:
                return (False, f"sigma_theta does not increase: base={base}, perturbed={perturbed}")
        else:
            return (True, "SKIP: cos(theta)<=0, monotonicity not applicable for this test")
    
    def test_monotonicity_E_th_increases_with_magnitude_sigma0(self) -> Tuple[bool, str]:
        """
        E_th = -sigma0/(2*epsilon0). If sigma0 becomes more negative (larger magnitude),
        E_th should increase.
        Perturb sigma0 by -10% (more negative) and recompute.
        """
        if not self._has_givens or self.E_th is None:
            return (False, "SKIP: missing data")
        
        base_E_th = -self.sigma0 / (2 * self.epsilon0)
        # More negative sigma0: multiply by 1.1 (since sigma0 is negative, 1.1*sigma0 is more negative)
        sigma0_pert = self.sigma0 * 1.1
        perturbed_E_th = -sigma0_pert / (2 * self.epsilon0)
        
        # First check that we have the correct baseline
        if not math.isclose(self.E_th, base_E_th, rel_tol=1e-2):
            return (False, "E_th not close to derived base; cannot check monotonicity")
        
        if perturbed_E_th > base_E_th:
            return (True, f"E_th increases as sigma0 becomes more negative: base={base_E_th}, perturbed={perturbed_E_th}")
        else:
            return (False, f"E_th does not increase: base={base_E_th}, perturbed={perturbed_E_th}")
    
    # ---------------------------------------------------------------
    # Answer sanity (weak only) – combined into one method
    # ---------------------------------------------------------------
    
    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """
        Weak sanity checks on final outputs:
        - All required output quantities are finite.
        - sigma_theta is not positive when E_a <= E_th (electrons are negative).
        - E_th and sigma_theta have plausible signs and magnitudes.
        No isclose, no re-derivation of expected values.
        """
        messages = []
        all_ok = True
        
        # 1. Finiteness checks
        if self.E_th is not None and not math.isfinite(self.E_th):
            all_ok = False
            messages.append("E_th not finite")
        if self.sigma_theta is not None and not math.isfinite(self.sigma_theta):
            all_ok = False
            messages.append("sigma_theta not finite")
        if self.E_max is not None and not math.isfinite(self.E_max):
            all_ok = False
            messages.append("E_max not finite")
        
        # 2. Sign check for sigma_theta (if applicable)
        if (self.sigma_theta is not None and self.E_a is not None and self.E_th is not None
                and self.E_a <= self.E_th + 1e-9):
            if self.sigma_theta > 1e-9:
                all_ok = False
                messages.append("sigma_theta should be ≤ 0 for E_a ≤ E_th")
        
        # 3. Broad magnitude sanity: E_th should be positive (since sigma0 negative, epsilon0 positive)
        if self.E_th is not None and self.E_th < 0:
            all_ok = False
            messages.append("E_th should be positive (sigma0 negative)")
        
        if all_ok:
            return (True, "All weak sanity checks passed")
        else:
            return (False, "; ".join(messages) if messages else "Sanity check failed for unknown reason")
    
    # ---------------------------------------------------------------
    # Main test runner
    # ---------------------------------------------------------------
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them, collect results.
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
        return (all_pass, logs)