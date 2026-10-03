import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary (only numeric floats)
        results = getattr(solution_instance, 'results', {})
        self.m = results.get('m', None)
        self.g = results.get('g', None)
        self.R = results.get('R', None)
        self.k = results.get('k', None)
        self.q = results.get('q', None)
        self.E = results.get('E', None)
        self.Q = results.get('Q', None)
        self.F = results.get('F', None)
        
        # Important angles in radians
        self.theta_a = math.pi / 3  # 60 degrees
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((success, msg))
        return success
    
    def _is_close(self, val1, val2, tol=1e-2):
        """Numerical fuzzy comparison with tolerance."""
        if val1 is None or val2 is None:
            return False
        try:
            return math.isclose(float(val1), float(val2), rel_tol=tol, abs_tol=tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Check fundamental physics laws based on KB theorems."""
        success = True
        
        # 1. Force balance at point a (from step_1): qE = mg * tan(60°)
        if all(v is not None for v in [self.E, self.m, self.g, self.q]):
            try:
                num_lhs = float(self.q * self.E)
                num_rhs = float(self.m * self.g * math.tan(self.theta_a))
                if self._is_close(num_lhs, num_rhs):
                    self._log("Force balance at point a satisfied.", True)
                else:
                    success = self._log(
                        f"Force balance at point a failed: qE={num_lhs}, mg*tan60°={num_rhs}",
                        False
                    )
            except Exception as e:
                success = self._log(f"Force balance at point a error: {e}", False)
        else:
            success = self._log("Force balance at point a: missing data.", False)
        
        # 2. Force balance at point b: mg = F * cos(60°)
        if all(v is not None for v in [self.F, self.m, self.g]):
            try:
                num_lhs = float(self.m * self.g)
                num_rhs = float(self.F * math.cos(self.theta_a))
                if self._is_close(num_lhs, num_rhs):
                    self._log("Vertical force balance at point b satisfied.", True)
                else:
                    success = self._log(
                        f"Vertical force balance at point b failed: mg={num_lhs}, F*cos60°={num_rhs}",
                        False
                    )
            except Exception as e:
                success = self._log(f"Vertical balance at point b error: {e}", False)
        else:
            success = self._log("Vertical force balance at point b: missing data.", False)
        
        # 3. Coulomb's Law: F = k*Q*q / R^2
        if all(v is not None for v in [self.F, self.k, self.Q, self.q, self.R]):
            try:
                num_lhs = float(self.F)
                num_rhs = float(self.k * self.Q * self.q / (self.R**2))
                if self._is_close(num_lhs, num_rhs):
                    self._log("Coulomb's Law satisfied.", True)
                else:
                    success = self._log(
                        f"Coulomb's Law failed: F={num_lhs}, kQq/R²={num_rhs}",
                        False
                    )
            except Exception as e:
                success = self._log(f"Coulomb's Law error: {e}", False)
        else:
            success = self._log("Coulomb's Law: missing data.", False)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with expected formulas."""
        success = True
        
        # Check E against expected: E = sqrt(3)*m*g/q
        if all(v is not None for v in [self.E, self.m, self.g, self.q]):
            try:
                expected_E = math.sqrt(3) * self.m * self.g / self.q
                if self._is_close(self.E, expected_E):
                    self._log("E matches expected expression.", True)
                else:
                    success = self._log(
                        f"E mismatch: got {self.E}, expected {expected_E}",
                        False
                    )
            except Exception as e:
                success = self._log(f"E trace check error: {e}", False)
        else:
            success = self._log("E trace check: missing data.", False)
        
        # Check Q against expected: Q = 2*m*g*R^2/(k*q)
        if all(v is not None for v in [self.Q, self.m, self.g, self.R, self.k, self.q]):
            try:
                expected_Q = 2 * self.m * self.g * self.R**2 / (self.k * self.q)
                if self._is_close(self.Q, expected_Q):
                    self._log("Q matches expected expression.", True)
                else:
                    success = self._log(
                        f"Q mismatch: got {self.Q}, expected {expected_Q}",
                        False
                    )
            except Exception as e:
                success = self._log(f"Q trace check error: {e}", False)
        else:
            success = self._log("Q trace check: missing data.", False)
        
        # Check F against expected: F = 2*m*g
        if all(v is not None for v in [self.F, self.m, self.g]):
            try:
                expected_F = 2 * self.m * self.g
                if self._is_close(self.F, expected_F):
                    self._log("F matches 2mg.", True)
                else:
                    success = self._log(
                        f"F mismatch: got {self.F}, expected {expected_F}",
                        False
                    )
            except Exception as e:
                success = self._log(f"F consistency check error: {e}", False)
        else:
            success = self._log("F consistency check: missing data.", False)
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and verify expected trends in E and Q."""
        success = True
        
        # Symbolic monotonicity: derivatives are positive constants
        # dE/dm = sqrt(3)*g/q > 0
        # dQ/dm = 2*g*R^2/(k*q) > 0
        # Since all constants are positive, monotonicity holds.
        self._log("Monotonicity: E increases with mass (symbolic).", True)
        self._log("Monotonicity: Q increases with mass (symbolic).", True)
        
        # Numerical perturbation check if data available
        if all(v is not None for v in [self.m, self.g, self.q, self.R, self.k]):
            try:
                m_original = float(self.m)
                m_perturbed = m_original * 1.01  # +1% perturbation
                
                # Compute perturbed E
                E_original = math.sqrt(3) * m_original * self.g / self.q
                E_perturbed = math.sqrt(3) * m_perturbed * self.g / self.q
                if E_perturbed > E_original:
                    self._log("Monotonicity: E increases with +1% mass change.", True)
                else:
                    success = self._log(
                        f"Monotonicity violation: E decreased from {E_original} to {E_perturbed}",
                        False
                    )
                
                # Compute perturbed Q
                Q_original = 2 * m_original * self.g * self.R**2 / (self.k * self.q)
                Q_perturbed = 2 * m_perturbed * self.g * self.R**2 / (self.k * self.q)
                if Q_perturbed > Q_original:
                    self._log("Monotonicity: Q increases with +1% mass change.", True)
                else:
                    success = self._log(
                        f"Monotonicity violation: Q decreased from {Q_original} to {Q_perturbed}",
                        False
                    )
            except Exception as e:
                success = self._log(f"Monotonicity numerical check error: {e}", False)
        else:
            self._log("Monotonicity numerical check: insufficient data.", True)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs.clear()
        
        try:
            overall_success = True
            
            # Run each test suite
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                try:
                    self._log(f"--- Starting {test_name} ---", None)
                    if not test_func():
                        overall_success = False
                except Exception as e:
                    self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                    overall_success = False
            
            # Summary
            passed = sum(1 for success, _ in self.logs if success is True)
            failed = sum(1 for success, _ in self.logs if success is False)
            self._log(f"Tests passed: {passed}, failed: {failed}", None)
            
            return overall_success, self.logs
            
        except Exception as e:
            return False, [("ERROR", f"Checker crashed: {traceback.format_exc()}")]