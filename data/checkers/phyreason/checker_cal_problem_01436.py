import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr and results fallback
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters - extract as floats
        self.M = self._get_numeric('M', 1.0)
        self.m = self._get_numeric('m')  # could be variable or final m_max
        self.theta = self._get_numeric('theta', 0.7853981633974483)  # π/4 ≈ 45°
        self.mu = self._get_numeric('mu', math.sqrt(3)/3)
        self.g = self._get_numeric('g', 10.0)
        
        # Computed quantities
        self.N1 = self._get_numeric('N_1')
        self.N2 = self._get_numeric('N_2')
        self.N3 = self._get_numeric('N_3')
        self.f = self._get_numeric('f')
        self.m_max = self._get_numeric('m_max')
    
    def _get_numeric(self, key, default=None):
        """Safely retrieve numeric value from solution or results."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, default)
        # Convert to float if possible
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                # If it's a SymPy expression, evaluate it
                if hasattr(val, 'evalf'):
                    return float(val.evalf())
                else:
                    return None
        return None
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        # Ensure both are numeric
        try:
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def _check_equilibrium_ball_vertical(self):
        """Check: N1*cos(theta) = m*g"""
        if self.N1 is None or self.theta is None or self.m is None or self.g is None:
            return False, "Missing data for ball vertical equilibrium"
        
        lhs = self.N1 * math.cos(self.theta)
        rhs = self.m * self.g
        success = self._is_close(lhs, rhs)
        msg = f"N1*cos(θ) = {lhs:.4f}, m*g = {rhs:.4f}"
        return success, msg
    
    def _check_equilibrium_ball_horizontal(self):
        """Check: N1*sin(theta) = N2"""
        if self.N1 is None or self.theta is None or self.N2 is None:
            return False, "Missing data for ball horizontal equilibrium"
        
        lhs = self.N1 * math.sin(self.theta)
        rhs = self.N2
        success = self._is_close(lhs, rhs)
        msg = f"N1*sin(θ) = {lhs:.4f}, N2 = {rhs:.4f}"
        return success, msg
    
    def _check_equilibrium_cube_vertical(self):
        """Check: N3 = (M + m)*g"""
        if self.N3 is None or self.M is None or self.m is None or self.g is None:
            return False, "Missing data for cube vertical equilibrium"
        
        lhs = self.N3
        rhs = (self.M + self.m) * self.g
        success = self._is_close(lhs, rhs)
        msg = f"N3 = {lhs:.4f}, (M+m)*g = {rhs:.4f}"
        return success, msg
    
    def _check_equilibrium_cube_horizontal(self):
        """Check: f = N2"""
        if self.f is None or self.N2 is None:
            return False, "Missing data for cube horizontal equilibrium"
        
        success = self._is_close(self.f, self.N2)
        msg = f"f = {self.f:.4f}, N2 = {self.N2:.4f}"
        return success, msg
    
    def _check_friction_inequality(self):
        """Check: f <= mu * N3 (with equality at m_max)"""
        if self.f is None or self.mu is None or self.N3 is None:
            return False, "Missing data for friction inequality"
        
        lhs = self.f
        rhs = self.mu * self.N3
        # At m_max, equality should hold
        success = self._is_close(lhs, rhs)
        msg = f"f = {lhs:.4f}, μ*N3 = {rhs:.4f}"
        return success, msg
    
    def _check_N1_expression(self):
        """Check: N1 = m*g/cos(theta)"""
        if self.N1 is None or self.m is None or self.theta is None or self.g is None:
            return False, "Missing data for N1 expression"
        
        lhs = self.N1
        rhs = self.m * self.g / math.cos(self.theta)
        success = self._is_close(lhs, rhs)
        msg = f"N1 = {lhs:.4f}, m*g/cos(θ) = {rhs:.4f}"
        return success, msg
    
    def _check_N2_expression(self):
        """Check: N2 = m*g*tan(theta)"""
        if self.N2 is None or self.m is None or self.theta is None or self.g is None:
            return False, "Missing data for N2 expression"
        
        lhs = self.N2
        rhs = self.m * self.g * math.tan(self.theta)
        success = self._is_close(lhs, rhs)
        msg = f"N2 = {lhs:.4f}, m*g*tan(θ) = {rhs:.4f}"
        return success, msg
    
    def _check_m_max_expression(self):
        """Check: m_max = (sqrt(3)+1)/2 * M"""
        if self.m_max is None or self.M is None:
            return False, "Missing data for m_max expression"
        
        lhs = self.m_max
        rhs = (math.sqrt(3) + 1)/2 * self.M
        success = self._is_close(lhs, rhs)
        msg = f"m_max = {lhs:.4f}, (√3+1)/2 * M = {rhs:.4f}"
        return success, msg
    
    def _check_m_max_limit_condition(self):
        """Check: m_max * g * tan(theta) = mu * (M + m_max) * g"""
        if self.m_max is None or self.M is None or self.theta is None or self.g is None or self.mu is None:
            return False, "Missing data for m_max limit condition"
        
        lhs = self.m_max * self.g * math.tan(self.theta)
        rhs = self.mu * (self.M + self.m_max) * self.g
        success = self._is_close(lhs, rhs)
        msg = f"m_max*g*tan(θ) = {lhs:.4f}, μ*(M+m_max)*g = {rhs:.4f}"
        return success, msg
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        logs = []
        success = True
        
        # 1. Ball vertical equilibrium
        ok, msg = self._check_equilibrium_ball_vertical()
        logs.append(f"Ball vertical equilibrium: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 2. Ball horizontal equilibrium
        ok, msg = self._check_equilibrium_ball_horizontal()
        logs.append(f"Ball horizontal equilibrium: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 3. Cube vertical equilibrium
        ok, msg = self._check_equilibrium_cube_vertical()
        logs.append(f"Cube vertical equilibrium: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 4. Cube horizontal equilibrium
        ok, msg = self._check_equilibrium_cube_horizontal()
        logs.append(f"Cube horizontal equilibrium: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 5. Friction inequality (should be equality at maximum)
        ok, msg = self._check_friction_inequality()
        logs.append(f"Friction inequality (f <= μN3): {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        # 1. N1 expression
        ok, msg = self._check_N1_expression()
        logs.append(f"N1 = m*g/cos(θ): {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 2. N2 expression
        ok, msg = self._check_N2_expression()
        logs.append(f"N2 = m*g*tan(θ): {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 3. m_max expression
        ok, msg = self._check_m_max_expression()
        logs.append(f"m_max = (√3+1)/2 * M: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        # 4. Limit condition at m_max
        ok, msg = self._check_m_max_limit_condition()
        logs.append(f"Limit condition at m_max: {msg} -> {'PASS' if ok else 'FAIL'}")
        success &= ok
        
        return success, logs
    
    def test_monotonicity(self):
        """Test that m_max is proportional to M and inequality holds for smaller m."""
        logs = []
        success = True
        
        # 1. Proportionality of m_max to M
        if self.m_max is not None and self.M is not None:
            ratio = self.m_max / self.M
            expected_ratio = (math.sqrt(3) + 1)/2
            ok = self._is_close(ratio, expected_ratio)
            logs.append(f"m_max/M ratio: {ratio:.4f}, expected: {expected_ratio:.4f} -> {'PASS' if ok else 'FAIL'}")
            success &= ok
            
            # 2. Check that for m < m_max, the inequality holds
            if self.m is not None and self.m != self.m_max and self.theta is not None:
                # Compute tan(theta) from the limit condition if possible
                # At m_max: tanθ = μ*(M+m_max)/m_max
                tan_theta = self.mu * (self.M + self.m_max) / self.m_max
                lhs = self.m * self.g * tan_theta
                rhs = self.mu * (self.M + self.m) * self.g
                if lhs <= rhs * 1.01:  # Allow 1% tolerance for floating point
                    logs.append(f"Inequality for m={self.m:.4f}: {lhs:.4f} <= {rhs:.4f} -> PASS")
                else:
                    logs.append(f"Inequality for m={self.m:.4f}: {lhs:.4f} <= {rhs:.4f} -> FAIL")
                    success = False
            else:
                logs.append("Skipping inequality test for m < m_max (missing data)")
        else:
            logs.append("Skipping monotonicity tests (missing m_max or M)")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test suite
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            success, logs = test_method()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            overall_success &= success
        
        return overall_success, all_logs