import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extracts inputs and results from the solution instance.
        """
        # Inputs (with safe defaults)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.h = getattr(solution_instance, 'h', 1.0)
        
        # Results (using .get() for dictionary or direct attribute)
        results = getattr(solution_instance, 'results', {})
        
        # Extract all required variables with proper keys
        # Note: R' in contract corresponds to R_prime in code
        self.alpha = results.get('alpha', None)
        self.y = results.get('y', None)
        self.R_prime = results.get('R_prime', None)  # Not "R'"
        self.B_direction = results.get('B_direction', None)
        self.R = results.get('R', None)
        self.R_condition = results.get('R_condition', None)
        self.B_m = results.get('B_m', None)
        
        # Convert numeric values to float (skip non-numeric)
        for attr in ['alpha', 'y', 'R_prime', 'R', 'B_m']:
            val = getattr(self, attr)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except (TypeError, ValueError):
                    # If conversion fails, keep original (e.g., symbolic)
                    pass
        
        # For R_condition, try to convert to float if possible
        if self.R_condition is not None:
            try:
                self.R_condition = float(self.R_condition)
            except (TypeError, ValueError):
                pass
        
        self.logs = []
        
    def _log(self, msg: str):
        """Helper to log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison for floats only."""
        if a is None or b is None:
            return False
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self) -> bool:
        """
        Check fundamental physical laws:
        1. Magnetic force does no work → speed constant.
        2. Lorentz force equals centripetal force.
        3. Minimum B condition.
        """
        success = True
        
        # 1. Speed is constant (magnetic force does no work) - assumed
        
        # 2. Lorentz force = centripetal force: q*v0*B = m*v0^2 / R  => R = m*v0/(q*B)
        if self.B_m is not None and self.R is not None:
            # Check that for B = B_m, R = h
            R_from_B_m = self.m * self.v0 / (self.q * self.B_m)
            if not self._is_close(R_from_B_m, self.h):
                self._log(f"Invariant failed: R from B_m is {R_from_B_m}, but h is {self.h}")
                success = False
            # Also check the formula R = m*v0/(q*B_m) gives h
            if not self._is_close(self.R, self.h):
                self._log(f"Invariant failed: R should equal h when B=B_m, but R={self.R}, h={self.h}")
                success = False
        
        # 3. For B = B_m/2, R' should be 2h
        if self.B_m is not None and self.R_prime is not None:
            B_half = self.B_m / 2
            R_prime_expected = self.m * self.v0 / (self.q * B_half)
            if not self._is_close(self.R_prime, R_prime_expected):
                self._log(f"Invariant failed: R' from B_m/2 is {R_prime_expected}, but got {self.R_prime}")
                success = False
            # Also check that R_prime equals 2h
            if not self._is_close(self.R_prime, 2 * self.h):
                self._log(f"Invariant failed: R' should be 2h, but R'={self.R_prime}, 2h={2*self.h}")
                success = False
        
        # 4. Check B_direction is "into the plane of the paper" (string check)
        if self.B_direction is not None:
            # Accept either string or numeric representation
            if isinstance(self.B_direction, str):
                if "into" not in self.B_direction.lower():
                    self._log(f"B_direction should be 'into the plane of the paper', but got {self.B_direction}")
                    success = False
            else:
                # If numeric, assume -1 represents "into"
                try:
                    if not math.isclose(float(self.B_direction), -1.0, rel_tol=1e-2):
                        self._log(f"B_direction numeric code should be -1 for 'into', but got {self.B_direction}")
                        success = False
                except (TypeError, ValueError):
                    self._log(f"B_direction should be string or numeric, but got {self.B_direction}")
                    success = False
        
        if success:
            self._log("All invariants passed.")
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Check consistency between intermediate results:
        - R' = 2h
        - α = arcsin(h/R') = π/6
        - y = R'(1 - cosα) = (2-√3)h
        """
        success = True
        
        # Check R' = 2h
        if self.R_prime is not None:
            if not self._is_close(self.R_prime, 2 * self.h):
                self._log(f"Trace consistency failed: R' should be 2h, but R'={self.R_prime}, 2h={2*self.h}")
                success = False
        
        # Check α = π/6 (or derived from geometry)
        if self.alpha is not None:
            alpha_expected = math.pi/6
            if not self._is_close(self.alpha, alpha_expected):
                # Alternatively, check sin(α) = h/R'
                if self.R_prime is not None:
                    sin_alpha = math.sin(self.alpha)
                    h_over_R_prime = self.h / self.R_prime
                    if not self._is_close(sin_alpha, h_over_R_prime):
                        self._log(f"Trace consistency failed: sin(α) should be h/R'={h_over_R_prime}, but sin(α)={sin_alpha}")
                        success = False
                else:
                    self._log(f"Trace consistency failed: α should be π/6≈{alpha_expected}, but got {self.alpha}")
                    success = False
        
        # Check y = (2-√3)h
        if self.y is not None:
            y_expected = (2 - math.sqrt(3)) * self.h
            if not self._is_close(self.y, y_expected):
                # Alternatively, check y = R'(1 - cosα)
                if self.R_prime is not None and self.alpha is not None:
                    cos_alpha = math.cos(self.alpha)
                    y_from_geometry = self.R_prime * (1 - cos_alpha)
                    if not self._is_close(self.y, y_from_geometry):
                        self._log(f"Trace consistency failed: y from geometry is {y_from_geometry}, but got {self.y}")
                        success = False
                else:
                    self._log(f"Trace consistency failed: y should be (2-√3)h≈{y_expected}, but got {self.y}")
                    success = False
        
        if success:
            self._log("Trace consistency passed.")
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input h and check trends:
        - If h increases, B_m decreases (inverse proportionality).
        - If h increases, y increases (linear proportionality).
        """
        success = True
        
        # Perturb h by 10%
        h_perturbed = self.h * 1.1
        
        # Recalculate B_m using formula: B_m = m*v0/(q*h)
        B_m_perturbed = self.m * self.v0 / (self.q * h_perturbed)
        
        # Check that B_m decreases when h increases
        if self.B_m is not None:
            if not (B_m_perturbed < self.B_m or self._is_close(B_m_perturbed, self.B_m)):
                self._log(f"Monotonicity failed: B_m should decrease when h increases. Original B_m={self.B_m}, perturbed B_m={B_m_perturbed}")
                success = False
        
        # Recalculate y using formula: y = (2-√3)h
        y_perturbed = (2 - math.sqrt(3)) * h_perturbed
        
        # Check that y increases when h increases
        if self.y is not None:
            if not (y_perturbed > self.y or self._is_close(y_perturbed, self.y)):
                self._log(f"Monotonicity failed: y should increase when h increases. Original y={self.y}, perturbed y={y_perturbed}")
                success = False
        
        if success:
            self._log("Monotonicity passed.")
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs.clear()
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for name, test in tests:
            self._log(f"Running {name}...")
            try:
                if not test():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {name} raised an exception: {e}")
                overall_success = False
        
        return overall_success, self.logs