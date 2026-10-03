import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize PhysicsChecker with safe attribute access.
        """
        self.logs = []
        
        # Extract input parameters safely
        self.p0 = getattr(solution_instance, 'p0', solution_instance.results.get('p0', 1.0e5))
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 0.01))
        self.L0 = getattr(solution_instance, 'L0', solution_instance.results.get('L0', 0.1))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.81))
        
        # Extract computed results safely
        self.p1 = getattr(solution_instance, 'p_1', solution_instance.results.get('p_1', None))
        self.p2 = getattr(solution_instance, 'p_2', solution_instance.results.get('p_2', None))
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.V_lower = getattr(solution_instance, 'V_lower', solution_instance.results.get('V_lower', None))
        
        # Constants
        self.V0 = self.S * self.L0
        self.V_upper_final = 0.5 * self.V0
        
        # Expected values
        self.p1_expected = 2.0 * self.p0
        self.p2_expected = (2.0/3.0) * self.p0
        self.m_expected = (4.0 * self.p0 * self.S) / (3.0 * self.g)
        self.V_lower_expected = 1.5 * self.V0

    def test_invariants(self):
        """
        Verify Boyle's Law and Force Equilibrium invariants.
        """
        logs = []
        success = True
        
        # 1. Boyle's Law for upper gas
        try:
            if self.p1 is not None:
                # p0 * V0 = p1 * V_upper_final
                lhs = self.p0 * self.V0
                rhs = self.p1 * self.V_upper_final
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Boyle's Law for upper gas violated. LHS={lhs:.3f}, RHS={rhs:.3f}")
                    success = False
                else:
                    logs.append("PASS: Boyle's Law for upper gas")
            else:
                logs.append("SKIP: p1 not provided")
                success = False
        except Exception as e:
            logs.append(f"ERROR in Boyle's Law (upper): {traceback.format_exc()}")
            success = False
        
        # 2. Boyle's Law for lower gas (if V_lower provided)
        try:
            if self.p2 is not None and self.V_lower is not None:
                lhs = self.p0 * self.V0
                rhs = self.p2 * self.V_lower
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Boyle's Law for lower gas violated. LHS={lhs:.3f}, RHS={rhs:.3f}")
                    success = False
                else:
                    logs.append("PASS: Boyle's Law for lower gas")
            else:
                logs.append("SKIP: p2 or V_lower not provided")
                if self.p2 is None or self.V_lower is None:
                    success = False
        except Exception as e:
            logs.append(f"ERROR in Boyle's Law (lower): {traceback.format_exc()}")
            success = False
        
        # 3. Force Equilibrium: p1*S = m*g + p2*S  (corrected)
        try:
            if self.p1 is not None and self.p2 is not None and self.m is not None:
                left_side = self.p1 * self.S
                right_side = self.m * self.g + self.p2 * self.S
                if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Force equilibrium violated. Left side (p1*S)={left_side:.3f}, Right side (m*g + p2*S)={right_side:.3f}")
                    success = False
                else:
                    logs.append("PASS: Force equilibrium")
            else:
                logs.append("SKIP: Missing p1, p2, or m for force equilibrium")
                success = False
        except Exception as e:
            logs.append(f"ERROR in force equilibrium: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        # 1. Check p1 matches expected
        try:
            if self.p1 is not None:
                if not math.isclose(self.p1, self.p1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: p1 mismatch. Expected {self.p1_expected:.3f}, got {self.p1:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: p1 matches expected value")
            else:
                logs.append("SKIP: p1 not provided")
                success = False
        except Exception as e:
            logs.append(f"ERROR in p1 check: {traceback.format_exc()}")
            success = False
        
        # 2. Check p2 matches expected
        try:
            if self.p2 is not None:
                if not math.isclose(self.p2, self.p2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: p2 mismatch. Expected {self.p2_expected:.3f}, got {self.p2:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: p2 matches expected value")
            else:
                logs.append("SKIP: p2 not provided")
                success = False
        except Exception as e:
            logs.append(f"ERROR in p2 check: {traceback.format_exc()}")
            success = False
        
        # 3. Check m matches expected
        try:
            if self.m is not None:
                if not math.isclose(self.m, self.m_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: m mismatch. Expected {self.m_expected:.3f}, got {self.m:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: m matches expected value")
            else:
                logs.append("SKIP: m not provided")
                success = False
        except Exception as e:
            logs.append(f"ERROR in m check: {traceback.format_exc()}")
            success = False
        
        # 4. Check V_lower matches geometry (if provided)
        try:
            if self.V_lower is not None:
                V_total = 2 * self.V0  # Initial total volume
                V_upper_final = 0.5 * self.V0
                V_lower_geom = V_total - V_upper_final
                if not math.isclose(self.V_lower, V_lower_geom, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: V_lower mismatch. Geometric {V_lower_geom:.3f}, got {self.V_lower:.3f}")
                    success = False
                else:
                    logs.append("PASS: V_lower matches geometric constraint")
        except Exception as e:
            logs.append(f"ERROR in V_lower check: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """
        Perturb p0 and check trends.
        """
        logs = []
        success = True
        
        try:
            # Save original
            original_p0 = self.p0
            
            # Perturb p0 by +10%
            self.p0 *= 1.10
            perturbed_p1_expected = 2.0 * self.p0
            perturbed_p2_expected = (2.0/3.0) * self.p0
            perturbed_m_expected = (4.0 * self.p0 * self.S) / (3.0 * self.g)
            
            # Check monotonicity: all should increase with p0
            if not (perturbed_p1_expected > self.p1_expected):
                logs.append("FAIL: p1 does not increase with p0")
                success = False
            else:
                logs.append("PASS: p1 increases with p0")
            
            if not (perturbed_p2_expected > self.p2_expected):
                logs.append("FAIL: p2 does not increase with p0")
                success = False
            else:
                logs.append("PASS: p2 increases with p0")
            
            if not (perturbed_m_expected > self.m_expected):
                logs.append("FAIL: m does not increase with p0")
                success = False
            else:
                logs.append("PASS: m increases with p0")
            
            # Restore original
            self.p0 = original_p0
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
            # Ensure restoration even on error
            if 'original_p0' in locals():
                self.p0 = original_p0
        
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs