import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from results dictionary with safe defaults
        # Use the exact keys from the problem specification
        self.Q = float(self.results.get('Q', 2e-7))
        self.q = float(self.results.get('q', 1e-7))
        self.m = float(self.results.get('m', 2e-4))
        self.l = float(self.results.get('r0', 0.5))  # initial distance
        self.k = float(self.results.get('k', 9e9))
        self.g = float(self.results.get('g', 10.0))
        
        # Extract results with safe access
        self.a = float(self.results.get('a', 0.0))
        self.L = float(self.results.get('L', 0.0))
        
        # Store for later restoration
        self._original_q = self.q
        
    def test_invariants(self):
        """Verify force balance at key points."""
        logs = []
        success = True
        
        try:
            # 1. Initial acceleration check (Newton's 2nd Law)
            # Force equation: mg - k*q*Q/l² = m*a
            F_gravity = self.m * self.g
            
            # Avoid division by zero
            if self.l == 0:
                logs.append("✗ Initial distance l cannot be zero")
                success = False
                return success, logs
            
            # Calculate electric force
            F_electric_initial = self.k * self.q * self.Q / (self.l**2)
            F_net_initial = F_gravity - F_electric_initial
            
            # Avoid division by zero for mass
            if self.m == 0:
                logs.append("✗ Mass cannot be zero")
                success = False
                return success, logs
                
            a_expected = F_net_initial / self.m
            
            # Use more generous tolerance for this specific problem
            if math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Initial acceleration check passed: a = {self.a:.4f}, expected = {a_expected:.4f}")
            else:
                logs.append(f"✗ Initial acceleration mismatch: a = {self.a:.4f}, expected = {a_expected:.4f}")
                success = False
            
            # 2. Maximum velocity condition (net force = 0)
            # Equation: mg = k*q*Q/L²
            if self.L > 0:
                F_electric_at_L = self.k * self.q * self.Q / (self.L**2)
            else:
                F_electric_at_L = float('inf')
                logs.append("✗ L must be positive for force calculation")
                success = False
            
            if success and math.isclose(F_gravity, F_electric_at_L, rel_tol=1e-2, abs_tol=1e-3):
                logs.append(f"✓ Force balance at max velocity: F_gravity = {F_gravity:.6f} N, F_electric(L) = {F_electric_at_L:.6f} N")
            elif success:
                logs.append(f"✗ Force imbalance at max velocity: F_gravity = {F_gravity:.6f} N, F_electric(L) = {F_electric_at_L:.6f} N")
                success = False
            
            # 3. Additional consistency: L should be less than initial distance
            if 0 < self.L < self.l:
                logs.append(f"✓ Distance consistency: L = {self.L:.3f} m < l = {self.l:.3f} m")
            else:
                logs.append(f"✗ Distance inconsistency: L = {self.L:.3f} m should be less than l = {self.l:.3f} m")
                success = False
                
        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # From step analysis: L should satisfy L = sqrt(k*q*Q/(m*g))
            # Avoid division by zero
            if self.m * self.g == 0:
                logs.append("✗ Cannot calculate L: m*g cannot be zero")
                success = False
                return success, logs
                
            L_calculated = math.sqrt(self.k * self.q * self.Q / (self.m * self.g))
            
            if math.isclose(self.L, L_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Distance L consistent: {self.L:.4f} m ≈ {L_calculated:.4f} m")
            else:
                logs.append(f"✗ Distance L inconsistent: {self.L:.4f} m ≠ {L_calculated:.4f} m")
                success = False
            
            # Check initial acceleration formula consistency
            # Avoid division by zero
            if self.m == 0 or self.l == 0:
                logs.append("✗ Cannot calculate a: m or l cannot be zero")
                success = False
                return success, logs
                
            a_calculated = self.g - self.k * self.q * self.Q / (self.m * self.l**2)
            
            if math.isclose(self.a, a_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Acceleration a consistent: {self.a:.4f} m/s² ≈ {a_calculated:.4f} m/s²")
            else:
                logs.append(f"✗ Acceleration a inconsistent: {self.a:.4f} m/s² ≠ {a_calculated:.4f} m/s²")
                success = False
                
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb charge q and verify expected trends."""
        logs = []
        success = True
        
        try:
            # Skip monotonicity test if basic values are zero or very small
            if abs(self.q) < 1e-10 or abs(self.Q) < 1e-10 or abs(self.m) < 1e-10 or abs(self.g) < 1e-10:
                logs.append("⚠ Skipping monotonicity test due to very small values")
                return success, logs
                
            # Increase q by 10% (still positive charge)
            q_perturbed = self.q * 1.10
            
            # Calculate new L with perturbed q
            L_perturbed = math.sqrt(self.k * q_perturbed * self.Q / (self.m * self.g))
            
            # Calculate new a with perturbed q
            a_perturbed = self.g - self.k * q_perturbed * self.Q / (self.m * self.l**2)
            
            # Expected trends:
            # 1. As q increases, electric repulsion increases → L increases
            # 2. As q increases, initial net downward force decreases → a decreases
            
            trend1_ok = L_perturbed > self.L or math.isclose(L_perturbed, self.L, rel_tol=1e-2)
            trend2_ok = a_perturbed < self.a or math.isclose(a_perturbed, self.a, rel_tol=1e-2)
            
            if trend1_ok:
                logs.append(f"✓ Monotonicity trend 1: L increases with q (L_original={self.L:.4f}, L_perturbed={L_perturbed:.4f})")
            else:
                logs.append(f"✗ Monotonicity trend 1 failed: L should increase with q")
                success = False
                
            if trend2_ok:
                logs.append(f"✓ Monotonicity trend 2: a decreases with q (a_original={self.a:.4f}, a_perturbed={a_perturbed:.4f})")
            else:
                logs.append(f"✗ Monotonicity trend 2 failed: a should decrease with q")
                success = False
                
            # Restore original value
            self.q = self._original_q
                
        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Restore original value even on error
            self.q = self._original_q
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Test 1: Physical Invariants ===")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Test 2: Trace Consistency ===")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Test 3: Monotonicity ===")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs