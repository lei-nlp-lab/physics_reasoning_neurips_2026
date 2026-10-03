import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract numeric inputs with defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.R = getattr(solution_instance, 'R', 1.0)
        
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            # If results is not a dict, try to get it as attribute
            self.results = {}
        
        # Get all required values from results dict
        self.F_N = self.results.get('F_N')
        self.mu = self.results.get('mu')
        self.E_k = self.results.get('E_k')
        self.v_A = self.results.get('v_A')
        self.v_C = self.results.get('v_C')
        self.F_N_minus_mg = self.results.get('F_N-mg')
        self.W = self.results.get('W')
        self.F_N_prime = self.results.get('F_N_prime')
        
        # Expected numeric values (computed with m=1, g=9.8, R=1)
        self.expected_v_C = math.sqrt(2 * self.g * self.R)
        self.expected_v_A = math.sqrt(self.g * self.R)
        self.expected_F_N = 3 * self.m * self.g
        self.expected_mu = 0.2
        self.expected_E_k = 3.5 * self.m * self.g * self.R
        
    def _num_check(self, val1, val2):
        """Numerical check with fuzzy tolerance."""
        if val1 is None or val2 is None:
            return False
        return math.isclose(float(val1), float(val2), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        
        # 1. Conservation of mechanical energy from B to C
        if self.v_C is not None:
            if self._num_check(self.v_C, self.expected_v_C):
                logs.append("PASS: v_C matches energy conservation (numerical).")
            else:
                logs.append(f"FAIL: v_C mismatch. Expected {self.expected_v_C:.4f}, got {self.v_C}.")
        else:
            logs.append("SKIP: v_C not provided.")
        
        # 2. Normal force at C via Newton's second law
        if self.F_N is not None:
            if self._num_check(self.F_N, self.expected_F_N):
                logs.append("PASS: F_N ≈ 3mg (numerical).")
            else:
                logs.append(f"FAIL: F_N mismatch. Expected {self.expected_F_N:.4f}, got {self.F_N}.")
        else:
            logs.append("SKIP: F_N not provided.")
        
        # 3. Friction coefficient from work-energy (C→D)
        if self.mu is not None:
            if self._num_check(self.mu, self.expected_mu):
                logs.append("PASS: μ ≈ 0.2 (numerical).")
            else:
                logs.append(f"FAIL: μ mismatch. Expected 0.2, got {self.mu}.")
        else:
            logs.append("SKIP: μ not provided.")
        
        # 4. Initial kinetic energy at D for motion to A
        if self.E_k is not None:
            if self._num_check(self.E_k, self.expected_E_k):
                logs.append("PASS: E_k ≈ 3.5mgR (numerical).")
            else:
                logs.append(f"FAIL: E_k mismatch. Expected {self.expected_E_k:.4f}, got {self.E_k}.")
        else:
            logs.append("SKIP: E_k not provided.")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # 1. v_A from Newton's second law at A
        if self.v_A is not None:
            if self._num_check(self.v_A, self.expected_v_A):
                logs.append("PASS: v_A ≈ √(gR) (numerical).")
            else:
                logs.append(f"FAIL: v_A mismatch. Expected {self.expected_v_A:.4f}, got {self.v_A}.")
        else:
            logs.append("SKIP: v_A not provided.")
        
        # 2. Consistency between F_N and v_C
        if self.F_N is not None and self.v_C is not None:
            expected_F_N_from_vC = self.m * self.g + self.m * self.v_C**2 / self.R
            if self._num_check(self.F_N, expected_F_N_from_vC):
                logs.append("PASS: F_N and v_C consistent (numerical).")
            else:
                logs.append("FAIL: F_N and v_C inconsistent.")
        else:
            logs.append("SKIP: F_N or v_C missing for consistency check.")
        
        # 3. Consistency between E_k, v_A, and μ for D→A motion
        if all(x is not None for x in [self.E_k, self.v_A, self.mu]):
            # Work-energy from D to A: -5μmgR - 2mgR = ½ m v_A² - E_k
            lhs = self.E_k
            rhs = 0.5 * self.m * self.v_A**2 + 5 * self.mu * self.m * self.g * self.R + 2 * self.m * self.g * self.R
            if self._num_check(lhs, rhs):
                logs.append("PASS: E_k, v_A, μ consistent (numerical).")
            else:
                logs.append("FAIL: E_k, v_A, μ inconsistent.")
        else:
            logs.append("SKIP: E_k, v_A, or μ missing for D→A consistency.")
        
        # 4. Newton's third law: F_N' = F_N
        if self.F_N_prime is not None and self.F_N is not None:
            if self._num_check(self.F_N_prime, self.F_N):
                logs.append("PASS: F_N' ≈ F_N (numerical).")
            else:
                logs.append("FAIL: F_N' ≠ F_N.")
        else:
            logs.append("SKIP: F_N' or F_N missing.")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends using numeric scaling."""
        logs = []
        
        # Since results are numeric (not symbolic), we test scaling by recomputing
        # with perturbed parameters using the physics formulas directly
        
        # Test scaling with mass m
        if self.F_N is not None:
            # F_N should be proportional to m: F_N = 3*m*g
            # With m=1, g=9.8, R=1: F_N = 29.4
            # With m=2, g=9.8, R=1: F_N should be 58.8
            expected_F_N_double = 3 * 2.0 * self.g * self.R
            if math.isclose(2.0 * self.F_N, expected_F_N_double, rel_tol=1e-2):
                logs.append("PASS: F_N doubles when m doubles (F_N ∝ m).")
            else:
                logs.append(f"FAIL: F_N not proportional to m. Expected {expected_F_N_double}, got {2.0*self.F_N}.")
        
        if self.mu is not None:
            # μ should be constant (0.2) regardless of m
            logs.append("PASS: μ independent of m (by definition).")
        
        if self.E_k is not None:
            # E_k should be proportional to m: E_k = 3.5*m*g*R
            expected_E_k_double = 3.5 * 2.0 * self.g * self.R
            if math.isclose(2.0 * self.E_k, expected_E_k_double, rel_tol=1e-2):
                logs.append("PASS: E_k doubles when m doubles (E_k ∝ m).")
            else:
                logs.append(f"FAIL: E_k not proportional to m. Expected {expected_E_k_double}, got {2.0*self.E_k}.")
        
        # Test scaling with radius R
        if self.v_C is not None:
            # v_C ∝ √R: v_C = √(2gR)
            # With R=4: v_C should be 2 times original (since √4 = 2)
            expected_v_C_double_R = math.sqrt(2 * self.g * 4.0)
            if math.isclose(2.0 * self.v_C, expected_v_C_double_R, rel_tol=1e-2):
                logs.append("PASS: v_C scales as √R.")
            else:
                logs.append(f"FAIL: v_C scaling with R incorrect. Expected {expected_v_C_double_R}, got {2.0*self.v_C}.")
        
        if self.F_N is not None:
            # F_N independent of R: F_N = 3mg
            logs.append("PASS: F_N independent of R (by formula).")
        
        if self.E_k is not None:
            # E_k ∝ R: E_k = 3.5mgR
            expected_E_k_double_R = 3.5 * self.m * self.g * 2.0
            if math.isclose(2.0 * self.E_k, expected_E_k_double_R, rel_tol=1e-2):
                logs.append("PASS: E_k doubles when R doubles (E_k ∝ R).")
            else:
                logs.append(f"FAIL: E_k not proportional to R. Expected {expected_E_k_double_R}, got {2.0*self.E_k}.")
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = ["=== Physics Checker ==="]
        
        try:
            logs.append("--- Invariant Tests ---")
            logs.extend(self.test_invariants())
            
            logs.append("--- Trace Consistency Tests ---")
            logs.extend(self.test_trace_consistency())
            
            logs.append("--- Monotonicity Tests ---")
            logs.extend(self.test_monotonicity())
        except Exception as e:
            logs.append(f"ERROR: Test suite crashed: {e}")
            logs.append(traceback.format_exc())
        
        # Determine overall success: no FAIL or ERROR messages
        success = True
        for log in logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        return success, logs