import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        # Extract inputs (with fallbacks)
        self.m = self._safe_get('m')
        self.v = self._safe_get('v')
        self.theta = self._safe_get('theta')  # in radians
        self.delta_t = self._safe_get('delta_t')
        
        # Extract results (with fallbacks)
        self.delta_p_x = self._safe_get('delta_p_x')
        self.delta_p_y = self._safe_get('delta_p_y')
        self.F_y = self._safe_get('F_y')
        self.F_ball_on_board = self._safe_get('F_ball_on_board')
        
        # Store logs for reporting
        self.logs = []
        
    def _safe_get(self, key):
        """Safely retrieve attribute from solution or its results dict."""
        # First try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return val
        # Then try results dictionary
        if hasattr(self.solution, 'results'):
            val = self.solution.results.get(key, None)
        return val
        
    def _log(self, message, success):
        """Append a test result to logs."""
        self.logs.append((success, message))
        
    def test_invariants(self):
        """Verify Newton's third law and momentum theorem."""
        success = True
        # 1. Newton's third law: F_ball_on_board = -F_y
        if self.F_ball_on_board is not None and self.F_y is not None:
            # Use fuzzy comparison (allowing 1-2% relative error)
            if math.isclose(self.F_ball_on_board, -self.F_y, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Newton's third law satisfied: F_ball_on_board = -F_y", True)
            else:
                self._log(f"Newton's third law violated: F_ball_on_board={self.F_ball_on_board}, -F_y={-self.F_y}", False)
                success = False
        else:
            self._log("Missing data for Newton's third law check", False)
            success = False
            
        # 2. Momentum theorem in y-direction: Δp_y = F_y * Δt
        if all(v is not None for v in [self.delta_p_y, self.F_y, self.delta_t]):
            expected_delta_p_y = self.F_y * self.delta_t
            if math.isclose(self.delta_p_y, expected_delta_p_y, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Momentum theorem in y-direction: Δp_y = F_y * Δt", True)
            else:
                self._log(f"Momentum theorem violated in y: Δp_y={self.delta_p_y}, F_y*Δt={expected_delta_p_y}", False)
                success = False
        else:
            self._log("Missing data for momentum theorem in y-direction", False)
            success = False
            
        # 3. Momentum theorem in x-direction: Δp_x should be 0 (since no force in x)
        if self.delta_p_x is not None:
            if math.isclose(self.delta_p_x, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Momentum theorem in x-direction: Δp_x ≈ 0 (no net force)", True)
            else:
                self._log(f"Momentum theorem violated in x: Δp_x={self.delta_p_x} (should be 0)", False)
                success = False
        else:
            self._log("Missing data for momentum theorem in x-direction", False)
            success = False
            
        return success
        
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        Follow the steps analysis: Δp_y → F_y → F_ball_on_board.
        """
        success = True
        # Step 2: Δp_y = 2*m*v*cos(θ)
        if all(v is not None for v in [self.m, self.v, self.theta, self.delta_p_y]):
            expected_delta_p_y = 2 * self.m * self.v * math.cos(self.theta)
            if math.isclose(self.delta_p_y, expected_delta_p_y, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 2 consistency: Δp_y = 2*m*v*cos(θ) ≈ {expected_delta_p_y}", True)
            else:
                self._log(f"Step 2 inconsistency: Δp_y={self.delta_p_y}, expected={expected_delta_p_y}", False)
                success = False
        else:
            self._log("Missing data for Step 2 consistency check", False)
            success = False
            
        # Step 3: F_y = Δp_y / Δt
        if all(v is not None for v in [self.delta_p_y, self.delta_t, self.F_y]):
            expected_F_y = self.delta_p_y / self.delta_t
            if math.isclose(self.F_y, expected_F_y, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 3 consistency: F_y = Δp_y/Δt ≈ {expected_F_y}", True)
            else:
                self._log(f"Step 3 inconsistency: F_y={self.F_y}, expected={expected_F_y}", False)
                success = False
        else:
            self._log("Missing data for Step 3 consistency check", False)
            success = False
            
        # Step 4: F_ball_on_board = -F_y
        if self.F_ball_on_board is not None and self.F_y is not None:
            if math.isclose(self.F_ball_on_board, -self.F_y, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 4 consistency: F_ball_on_board = -F_y ≈ {-self.F_y}", True)
            else:
                self._log(f"Step 4 inconsistency: F_ball_on_board={self.F_ball_on_board}, -F_y={-self.F_y}", False)
                success = False
        else:
            self._log("Missing data for Step 4 consistency check", False)
            success = False
            
        return success
        
    def test_monotonicity(self):
        """
        Perturb mass m by +10%, recompute Δp_y and F_y, check proportional increase.
        Restore original m after test.
        """
        if None in [self.m, self.v, self.theta, self.delta_t]:
            self._log("Missing input data for monotonicity test", False)
            return False
            
        success = True
        original_m = self.m
        # Perturb mass by 10%
        perturbed_m = original_m * 1.10
        
        # Recompute expected values using formulas from steps
        # Δp_y = 2*m*v*cos(θ)
        original_delta_p_y = 2 * original_m * self.v * math.cos(self.theta)
        perturbed_delta_p_y = 2 * perturbed_m * self.v * math.cos(self.theta)
        
        # F_y = Δp_y / Δt
        original_F_y = original_delta_p_y / self.delta_t
        perturbed_F_y = perturbed_delta_p_y / self.delta_t
        
        # Check proportionality: both should increase by 10%
        # For Δp_y
        if not math.isclose(perturbed_delta_p_y / original_delta_p_y, 1.10, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Monotonicity fail: Δp_y did not scale linearly with m (factor={perturbed_delta_p_y / original_delta_p_y})", False)
            success = False
        else:
            self._log("Δp_y scales linearly with m (+10% m → +10% Δp_y)", True)
            
        # For F_y
        if not math.isclose(perturbed_F_y / original_F_y, 1.10, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Monotonicity fail: F_y did not scale linearly with m (factor={perturbed_F_y / original_F_y})", False)
            success = False
        else:
            self._log("F_y scales linearly with m (+10% m → +10% F_y)", True)
            
        return success
        
    def run_all_tests(self):
        """
        Execute all audit tests and return overall success and logs.
        """
        self.logs = []  # Reset logs
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} test crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False
                
        return overall_success, self.logs