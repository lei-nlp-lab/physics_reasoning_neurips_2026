import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance or its results dictionary
        # Use sensible defaults to avoid division by zero
        self.U = float(getattr(solution_instance, 'U', solution_instance.results.get('U', 1000.0)))
        self.m = float(getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0e-25)))
        self.Z = float(getattr(solution_instance, 'Z', solution_instance.results.get('Z', 1)))
        self.e = float(getattr(solution_instance, 'e', solution_instance.results.get('e', 1.60217662e-19)))
        
        # Safely extract results from solution_instance or its results dictionary
        # Note: keys in results dictionary are exactly as per contract: 'F/P', "F_1'", "F'", etc.
        self.P = float(solution_instance.results.get('P', 0.0))
        self.F_over_P = float(solution_instance.results.get('F/P', 0.0))
        self.v = float(solution_instance.results.get('v', 0.0))
        self.F_prime = float(solution_instance.results.get("F'", 0.0))
        self.N = float(solution_instance.results.get('N', 0.0))
        self.F1_prime = float(solution_instance.results.get("F_1'", 0.0))
        
        # For convenience, compute derived quantities for later checks (numeric)
        # Avoid division by zero: if m == 0, set v_expected to 0
        if self.m == 0:
            self.v_expected = 0.0
        else:
            self.v_expected = math.sqrt(2 * self.Z * self.e * self.U / self.m)
        
        # Compute expected N and F/P similarly
        denom = math.sqrt(2 * self.Z * self.e * self.m * self.U)
        if denom == 0:
            self.N_expected = 0.0
        else:
            self.N_expected = self.F_prime / denom
        
        denom2 = self.Z * self.e * self.U
        if denom2 == 0:
            self.F_over_P_expected = 0.0
        else:
            self.F_over_P_expected = math.sqrt(2 * self.m / denom2)
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Simple numeric comparison.
        """
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Verify fundamental physical laws: energy conservation, Newton's laws, power formula.
        """
        logs = []
        success = True
        
        # 1. Energy conservation: ZeU = (1/2) m v^2
        lhs = self.Z * self.e * self.U
        rhs = 0.5 * self.m * self.v**2
        if not self._isclose(lhs, rhs):
            logs.append(f"Energy conservation failed: ZeU = {lhs}, (1/2)mv^2 = {rhs}")
            success = False
        
        # 2. Newton's second law for ion beam: F1' = N * m * v
        expected_F1 = self.N * self.m * self.v
        if not self._isclose(self.F1_prime, expected_F1):
            logs.append(f"Newton's second law failed: F1' = {self.F1_prime}, expected N*m*v = {expected_F1}")
            success = False
        
        # 3. Newton's third law: F1' = F' (thrust)
        if not self._isclose(self.F1_prime, self.F_prime):
            logs.append(f"Newton's third law failed: F1' = {self.F1_prime}, but F' (thrust) = {self.F_prime}")
            success = False
        
        # 4. Power formula: P = (1/2) * F' * v
        expected_P = 0.5 * self.F_prime * self.v
        if not self._isclose(self.P, expected_P):
            logs.append(f"Power formula failed: P = {self.P}, expected (1/2)*F'*v = {expected_P}")
            success = False
        
        # 5. Formula for N: N = F' / sqrt(2 Z e m U)
        if not self._isclose(self.N, self.N_expected):
            logs.append(f"N formula failed: N = {self.N}, expected F'/sqrt(2Ze m U) = {self.N_expected}")
            success = False
        
        # 6. Formula for F/P: F/P = sqrt(2m/(ZeU))
        if not self._isclose(self.F_over_P, self.F_over_P_expected):
            logs.append(f"F/P formula failed: F/P = {self.F_over_P}, expected sqrt(2m/(ZeU)) = {self.F_over_P_expected}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        # Check that thrust from F/P ratio matches F'
        F_from_ratio = self.F_over_P * self.P
        if not self._isclose(self.F_prime, F_from_ratio):
            logs.append(f"Trace consistency: F' from F/P ratio mismatch: F' = {self.F_prime}, (F/P)*P = {F_from_ratio}")
            success = False
        
        # Check that v from energy is used consistently in force and power
        expected_F1 = self.N * self.m * self.v_expected
        if not self._isclose(self.F1_prime, expected_F1):
            logs.append(f"Trace consistency: F1' does not match N*m*v_expected, where v_expected = sqrt(2ZeU/m)")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb inputs to verify expected trends in F/P.
        """
        logs = []
        success = True
        
        # Store original values
        m_orig, Z_orig, U_orig = self.m, self.Z, self.U
        
        # Helper to compute F/P numerically
        def compute_F_over_P(m, Z, U):
            if Z * self.e * U == 0:
                return 0.0
            return math.sqrt(2 * m / (Z * self.e * U))
        
        # Check monotonicity w.r.t. mass m
        try:
            self.m = m_orig * 1.01  # 1% increase
            F_over_P_new = compute_F_over_P(self.m, self.Z, self.U)
            self.m = m_orig  # restore
            if not (F_over_P_new > self.F_over_P or self._isclose(F_over_P_new, self.F_over_P)):
                logs.append(f"Monotonicity w.r.t. m: increasing m did not increase F/P.")
                success = False
        except Exception as e:
            logs.append(f"Monotonicity test for m failed: {e}")
            success = False
        
        # Check monotonicity w.r.t. charge number Z
        try:
            self.Z = Z_orig * 1.01
            F_over_P_new = compute_F_over_P(self.m, self.Z, self.U)
            self.Z = Z_orig
            if not (F_over_P_new < self.F_over_P or self._isclose(F_over_P_new, self.F_over_P)):
                logs.append(f"Monotonicity w.r.t. Z: increasing Z did not decrease F/P.")
                success = False
        except Exception as e:
            logs.append(f"Monotonicity test for Z failed: {e}")
            success = False
        
        # Check monotonicity w.r.t. voltage U
        try:
            self.U = U_orig * 1.01
            F_over_P_new = compute_F_over_P(self.m, self.Z, self.U)
            self.U = U_orig
            if not (F_over_P_new < self.F_over_P or self._isclose(F_over_P_new, self.F_over_P)):
                logs.append(f"Monotonicity w.r.t. U: increasing U did not decrease F/P.")
                success = False
        except Exception as e:
            logs.append(f"Monotonicity test for U failed: {e}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
                if logs:
                    all_logs.append(f"{test_name}:")
                    all_logs.extend([f"  {log}" for log in logs])
            except Exception as e:
                overall_success = False
                all_logs.append(f"{test_name} raised exception: {e}")
                all_logs.append(traceback.format_exc())
        
        return overall_success, all_logs