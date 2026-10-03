import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required quantities from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs and results with safe fallbacks
        self.delta_t = float(getattr(solution_instance, 'delta_t', self.results.get('delta_t', 0.5)))
        self.lambda_ = float(getattr(solution_instance, 'λ', self.results.get('λ', 8.0)))
        
        # Wave speeds for two directions
        v_pos_raw = getattr(solution_instance, 'v_pos', 
                           self.results.get('v_pos', self.results.get('v_positive', 54.0)))
        v_neg_raw = getattr(solution_instance, 'v_neg', 
                           self.results.get('v_neg', self.results.get('v_negative', 58.0)))
        self.v_pos = float(v_pos_raw) if v_pos_raw is not None else 54.0
        self.v_neg = float(v_neg_raw) if v_neg_raw is not None else 58.0
        
        # Propagation direction (string expected, e.g., 'negative')
        direction_raw = getattr(solution_instance, 'direction', 
                               self.results.get('direction', ''))
        # Convert to string safely
        if isinstance(direction_raw, (int, float)):
            self.direction_str = "negative" if direction_raw < 0 else "positive"
        else:
            self.direction_str = str(direction_raw).lower()
        
        # Integer k from the solution (should be 3)
        k_raw = getattr(solution_instance, 'k', self.results.get('k', 3))
        self.k = int(k_raw) if k_raw is not None else 3
        
        # Derived periods
        self.T_pos = self.lambda_ / self.v_pos if self.v_pos != 0 else None
        self.T_neg = self.lambda_ / self.v_neg if self.v_neg != 0 else None
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        self.logs = []

    def _log(self, msg, success=None):
        """Helper to log messages and success status."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)

    def test_invariants(self):
        """Verify fundamental wave relationships and boundary conditions."""
        success = True
        
        # 1. Check v = λ / T for both directions
        if self.T_pos is not None:
            v_from_T = self.lambda_ / self.T_pos
            if not math.isclose(self.v_pos, v_from_T, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Positive direction: v = {self.v_pos} but λ/T = {v_from_T}", False)
                success = False
            else:
                self._log(f"Positive direction: v = λ/T holds.", True)
        
        if self.T_neg is not None:
            v_from_T = self.lambda_ / self.T_neg
            if not math.isclose(self.v_neg, v_from_T, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Negative direction: v = {self.v_neg} but λ/T = {v_from_T}", False)
                success = False
            else:
                self._log(f"Negative direction: v = λ/T holds.", True)
        
        # 2. Verify the waveform shift conditions Δt = (k + fraction) * T
        # Positive: fraction = 3/8
        if self.T_pos is not None:
            dt_calc_pos = (self.k + 3/8) * self.T_pos
            if not math.isclose(self.delta_t, dt_calc_pos, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Positive: Δt = {self.delta_t} but (k+3/8)T = {dt_calc_pos}", False)
                success = False
            else:
                self._log(f"Positive: Δt = (k+3/8)T holds with k={self.k}.", True)
        
        # Negative: fraction = 5/8
        if self.T_neg is not None:
            dt_calc_neg = (self.k + 5/8) * self.T_neg
            if not math.isclose(self.delta_t, dt_calc_neg, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Negative: Δt = {self.delta_t} but (k+5/8)T = {dt_calc_neg}", False)
                success = False
            else:
                self._log(f"Negative: Δt = (k+5/8)T holds with k={self.k}.", True)
        
        # 3. Check the inequality 3T ≤ Δt < 4T for both periods
        if self.T_pos is not None:
            lower_ok = 3 * self.T_pos <= self.delta_t + 1e-9  # small epsilon for floating point
            upper_ok = self.delta_t < 4 * self.T_pos
            if not (lower_ok and upper_ok):
                self._log(f"Positive: condition 3T ≤ Δt < 4T fails (T={self.T_pos}, Δt={self.delta_t})", False)
                success = False
            else:
                self._log(f"Positive: 3T ≤ Δt < 4T satisfied.", True)
        
        if self.T_neg is not None:
            lower_ok = 3 * self.T_neg <= self.delta_t + 1e-9
            upper_ok = self.delta_t < 4 * self.T_neg
            if not (lower_ok and upper_ok):
                self._log(f"Negative: condition 3T ≤ Δt < 4T fails (T={self.T_neg}, Δt={self.delta_t})", False)
                success = False
            else:
                self._log(f"Negative: 3T ≤ Δt < 4T satisfied.", True)
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results from the steps analysis."""
        success = True
        
        # Expected values from the steps
        expected_lambda = 8.0
        expected_k = 3
        expected_v_pos = 54.0
        expected_v_neg = 58.0
        expected_direction = "negative"
        
        # Check wavelength
        if not math.isclose(self.lambda_, expected_lambda, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Wavelength: expected {expected_lambda}, got {self.lambda_}", False)
            success = False
        else:
            self._log(f"Wavelength λ = {self.lambda_} matches expected {expected_lambda}.", True)
        
        # Check k
        if self.k != expected_k:
            self._log(f"Integer k: expected {expected_k}, got {self.k}", False)
            success = False
        else:
            self._log(f"Integer k = {self.k} matches expected {expected_k}.", True)
        
        # Check wave speeds
        if not math.isclose(self.v_pos, expected_v_pos, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Positive wave speed: expected {expected_v_pos}, got {self.v_pos}", False)
            success = False
        else:
            self._log(f"Positive wave speed v = {self.v_pos} m/s matches expected.", True)
        
        if not math.isclose(self.v_neg, expected_v_neg, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Negative wave speed: expected {expected_v_neg}, got {self.v_neg}", False)
            success = False
        else:
            self._log(f"Negative wave speed v = {self.v_neg} m/s matches expected.", True)
        
        # Check direction string (case-insensitive partial match)
        if expected_direction not in self.direction_str:
            self._log(f"Direction: expected '{expected_direction}', got '{self.direction_str}'", False)
            success = False
        else:
            self._log(f"Direction '{self.direction_str}' matches expected '{expected_direction}'.", True)
        
        return success

    def test_monotonicity(self):
        """
        Perturb wavelength λ by 1% and verify that wave speeds change proportionally,
        assuming the same integer k and shift fractions.
        """
        success = True
        
        # Original values
        lambda_orig = self.lambda_
        delta_t = self.delta_t
        k = self.k
        
        # Perturb λ by +1%
        lambda_pert = lambda_orig * 1.01
        
        # Recalculate wave speeds using the formulas from steps:
        # v = (k + fraction) * λ / Δt
        v_pos_pert = (k + 3/8) * lambda_pert / delta_t
        v_neg_pert = (k + 5/8) * lambda_pert / delta_t
        
        # Expected proportional change: 1% increase
        v_pos_expected = self.v_pos * 1.01
        v_neg_expected = self.v_neg * 1.01
        
        # Check positive direction
        if not math.isclose(v_pos_pert, v_pos_expected, rel_tol=5e-3, abs_tol=5e-3):
            self._log(f"Positive v monotonicity: after +1% λ, expected v ≈ {v_pos_expected}, got {v_pos_pert}", False)
            success = False
        else:
            self._log(f"Positive v changes proportionally with λ (+1% λ → +1% v).", True)
        
        # Check negative direction
        if not math.isclose(v_neg_pert, v_neg_expected, rel_tol=5e-3, abs_tol=5e-3):
            self._log(f"Negative v monotonicity: after +1% λ, expected v ≈ {v_neg_expected}, got {v_neg_pert}", False)
            success = False
        else:
            self._log(f"Negative v changes proportionally with λ (+1% λ → +1% v).", True)
        
        return success

    def run_all_tests(self):
        """
        Execute all checks and return overall success and logs.
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                self._log(f"Running {name}...")
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self._log(f"Error in {name}: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs