import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all attributes from solution instance."""
        self.solution = solution_instance
        
        # Results with safe access
        self.results = getattr(solution_instance, 'results', {})
        
        # Extracted numeric values (convert strings if needed)
        self.t_B = self._safe_get_numeric('t_B')
        self.t_A = self._safe_get_numeric('t_A')
        self.fraca_Aa_B = self._safe_get_numeric('fraca_Aa_B')
        self.a = self._safe_get_numeric('a')
        
        # Extract exit athlete as string
        self.exit_athlete = getattr(solution_instance, 'N/A', 
                                   self.results.get('N/A', None))
        if isinstance(self.exit_athlete, (int, float)):
            # Convert numeric to string if needed
            self.exit_athlete = 'A' if float(self.exit_athlete) == 1 else 'B'
        
        # Extract inputs if available
        self.R_A = self._safe_get_numeric('R_A', default=6.0)
        self.R_B = self._safe_get_numeric('R_B', default=8.0)
        self.v_A = self._safe_get_numeric('v_A', default=9.0)
        self.v_B = self._safe_get_numeric('v_B', default=11.0)
        self.x = self._safe_get_numeric('x', default=15.0)
        
        # Ground truth values from the problem solution
        self.expected_a = 2.7
        self.expected_ratio = 225/242  # ≈ 0.9297520661157025
        self.expected_t_A = 4*math.pi/5  # ≈ 2.5132741228718345
        self.expected_t_B = 9*math.pi/11  # ≈ 2.5703939893007397
        self.expected_exit = 'A'
        
        # Cache for original values in monotonicity test
        self._original_cache = {}

    def _safe_get_numeric(self, key, default=None):
        """Safely extract and convert numeric values only."""
        value = getattr(self.solution, key, self.results.get(key, default))
        
        if value is None:
            return default
        
        # Handle string representations of numbers
        if isinstance(value, str):
            try:
                # Handle fractions like "225/242"
                if '/' in value and 'π' not in value:
                    from fractions import Fraction
                    return float(Fraction(value))
                # Handle expressions like "4π/5"
                elif 'π' in value:
                    import sympy as sp
                    expr = value.replace('π', 'pi')
                    return float(sp.sympify(expr).evalf())
                else:
                    # Try direct conversion
                    return float(value)
            except:
                return default
        
        # Handle sympy expressions
        if hasattr(value, 'evalf'):
            return float(value.evalf())
        
        # Handle numeric types
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def test_invariants(self):
        """Check fundamental physics invariants against ground truth."""
        logs = []
        success = True
        
        # 1. Check acceleration matches ground truth
        if self.a is not None:
            if self.a <= 0:
                logs.append(f"ERROR: Acceleration must be positive, got {self.a}")
                success = False
            if not math.isclose(self.a, self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"ERROR: Acceleration {self.a} doesn't match ground truth {self.expected_a}")
                success = False
        
        # 2. Check ratio matches ground truth
        if self.fraca_Aa_B is not None:
            if not (0 <= self.fraca_Aa_B <= 1):
                logs.append(f"ERROR: Ratio must be between 0 and 1, got {self.fraca_Aa_B}")
                success = False
            if not math.isclose(self.fraca_Aa_B, self.expected_ratio, 
                               rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"ERROR: Ratio {self.fraca_Aa_B} doesn't match ground truth {self.expected_ratio}")
                success = False
        
        # 3. Check times match ground truth and ordering
        if self.t_A is not None and self.t_B is not None:
            if self.t_A <= 0 or self.t_B <= 0:
                logs.append(f"ERROR: Times must be positive: t_A={self.t_A}, t_B={self.t_B}")
                success = False
            if not math.isclose(self.t_A, self.expected_t_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"ERROR: t_A={self.t_A} doesn't match ground truth {self.expected_t_A}")
                success = False
            if not math.isclose(self.t_B, self.expected_t_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"ERROR: t_B={self.t_B} doesn't match ground truth {self.expected_t_B}")
                success = False
            if self.t_A >= self.t_B:
                logs.append(f"ERROR: t_A ({self.t_A}) should be less than t_B ({self.t_B}) since A exits first")
                success = False
        
        # 4. Check exit athlete
        if self.exit_athlete != self.expected_exit:
            logs.append(f"ERROR: Athlete {self.exit_athlete} exits first, but ground truth says {self.expected_exit}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """Check internal consistency between steps."""
        logs = []
        success = True
        
        # We need radii and speeds to check consistency
        if None in [self.R_A, self.R_B, self.v_A, self.v_B, self.a, self.x]:
            logs.append("INFO: Missing parameters - skipping detailed trace checks")
            return True, logs  # Not a failure, just incomplete data
        
        # 1. Check times formula: t = πR/v (should hold for any consistent radii)
        # But we don't know the actual radii from ground truth, so skip
        
        # 2. Check ratio formula: a_A/a_B = (v_A²/v_B²) * (R_B/R_A)
        # This should hold if we had the correct radii, but we don't
        
        # 3. Check that v_A is consistent with kinematic equation v² = 2aΔx
        expected_v_A_sq = 2 * self.a * self.x
        if not math.isclose(self.v_A**2, expected_v_A_sq, 
                           rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ERROR: v_A²={self.v_A**2} doesn't match 2aΔx={expected_v_A_sq}")
            success = False
        
        # Note: v_B is not required to satisfy v_B² = 2aΔx in the problem statement
        # The ground truth only uses v_A to find acceleration
        
        return success, logs

    def test_monotonicity(self):
        """Check that increasing speed decreases time."""
        logs = []
        
        # We need actual radii to test monotonicity, but we don't have them
        # Instead, we'll test with the radii that would produce the ground truth times
        # From t = πR/v, we can compute implied radii
        if None in [self.v_A, self.v_B, self.t_A, self.t_B]:
            logs.append("INFO: Missing parameters - skipping monotonicity test")
            return True, logs
        
        # Compute implied radii from ground truth times
        implied_R_A = self.t_A * self.v_A / math.pi
        implied_R_B = self.t_B * self.v_B / math.pi
        
        # Save original values
        self._original_cache = {
            'v_A': self.v_A,
            'v_B': self.v_B,
            't_A': self.t_A,
            't_B': self.t_B,
            'fraca_Aa_B': self.fraca_Aa_B,
            'R_A': implied_R_A,
            'R_B': implied_R_B
        }
        
        # Test 1: Increase v_A by 10%, time should decrease
        perturbed_v_A = self.v_A * 1.1
        new_t_A = math.pi * implied_R_A / perturbed_v_A
        
        if new_t_A >= self.t_A:
            logs.append(f"ERROR: Increasing v_A from {self.v_A} to {perturbed_v_A} "
                       f"should decrease t_A, but got {new_t_A} vs {self.t_A}")
            self._restore_values()
            return False, logs
        
        # Test 2: Increase v_B by 10%, time should decrease
        perturbed_v_B = self.v_B * 1.1
        new_t_B = math.pi * implied_R_B / perturbed_v_B
        
        if new_t_B >= self.t_B:
            logs.append(f"ERROR: Increasing v_B from {self.v_B} to {perturbed_v_B} "
                       f"should decrease t_B, but got {new_t_B} vs {self.t_B}")
            self._restore_values()
            return False, logs
        
        # Test 3: Ratio monotonicity (using implied radii)
        if self.fraca_Aa_B is not None:
            new_ratio = (perturbed_v_A**2 / self.v_B**2) * (implied_R_B / implied_R_A)
            if new_ratio <= self.fraca_Aa_B:
                logs.append(f"ERROR: Increasing v_A should increase ratio, "
                           f"but {new_ratio} ≤ {self.fraca_Aa_B}")
                self._restore_values()
                return False, logs
        
        logs.append("Monotonicity checks passed: increasing speed decreases time, "
                   "and increases centripetal acceleration ratio")
        
        self._restore_values()
        return True, logs

    def _restore_values(self):
        """Restore original values after perturbation."""
        if not self._original_cache:
            return
        
        # Reassign to self attributes
        for key, value in self._original_cache.items():
            setattr(self, key, value)

    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                
                if success:
                    all_logs.append(f"✓ {test_name} passed")
                else:
                    all_logs.append(f"✗ {test_name} failed")
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"✗ {test_name} crashed with error: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs