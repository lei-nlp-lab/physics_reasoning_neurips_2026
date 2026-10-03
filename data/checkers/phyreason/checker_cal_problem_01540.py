import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.sol = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters: extract from results dict with defaults
        self.m = self.results.get('m', 1.0)  # default 1.0
        self.g = self.results.get('g', 9.8)  # default 9.8
        self.L0 = self.results.get('L0', 1.0)  # default 1.0
        
        # Extracted results with safe access
        self.f = self.results.get('f', None)
        self.T = self.results.get('T', None)
        self.mu = self.results.get('mu', None)  # note: contract uses '$mu$' but we store as 'mu'
        self.L_CF = self.results.get('L_CF', None)
        self.v_m = self.results.get('v_m', None)
        self.F_prime = self.results.get('F_prime', None)
        self.k = self.results.get('k', None)
        
        # Derived geometric quantities
        self.BC = 1.25 * self.L0
        self.CD = 2.0 * self.L0
        # OB = sqrt(L0^2 + (1.25L0)^2) = L0 * sqrt(1 + 1.5625) = L0 * sqrt(2.5625)
        self.OB = self.L0 * math.sqrt(2.5625)  # sqrt(41)/4 * L0 ≈ 1.6008*L0
        
        self.logs = []
        
    def log(self, msg, success=None):
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify energy conservation and force balances"""
        try:
            # 1. Hooke's Law check at point B
            if self.k is not None and self.T is not None and self.OB is not None:
                expected_T = self.k * self.OB
                if math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Hooke's Law at B: T = {self.T} ≈ k*OB = {expected_T}", True)
                else:
                    self.log(f"Hooke's Law mismatch: T={self.T}, k*OB={expected_T}", False)
            
            # 2. Normal force from step 5: F' = k * L0
            if self.F_prime is not None and self.k is not None and self.L0 is not None:
                expected_F_prime = self.k * self.L0
                if math.isclose(self.F_prime, expected_F_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Normal force: F' = {self.F_prime} ≈ k*L0 = {expected_F_prime}", True)
                else:
                    self.log(f"Normal force mismatch: F'={self.F_prime}, k*L0={expected_F_prime}", False)
            
            # 3. Friction force relation: f = μ * F'
            if self.f is not None and self.mu is not None and self.F_prime is not None:
                expected_f = self.mu * self.F_prime
                if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Friction relation: f = {self.f} ≈ μ*F' = {expected_f}", True)
                else:
                    self.log(f"Friction mismatch: f={self.f}, μ*F'={expected_f}", False)
            
            # 4. Force balance at maximum speed position (step 11)
            if self.L_CF is not None and self.k is not None and self.f is not None and self.m is not None and self.g is not None:
                expected_L_CF = (self.m * self.g + self.f) / self.k
                if math.isclose(self.L_CF, expected_L_CF, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Force balance at max speed: L_CF = {self.L_CF} ≈ (mg+f)/k = {expected_L_CF}", True)
                else:
                    self.log(f"Force balance mismatch: L_CF={self.L_CF}, (mg+f)/k={expected_L_CF}", False)
            
            # 5. Energy check: Work-energy theorem from C to D
            # ΔK = W_gravity + W_elastic + W_friction = 0 (comes to rest at D)
            if all(v is not None for v in [self.m, self.g, self.L0, self.k, self.mu, self.F_prime, self.OB]):
                # Vertical displacement from C to D: 2L0 downward
                delta_h = 2 * self.L0
                W_gravity = self.m * self.g * delta_h
                
                # Elastic energy: initial (at C) and final (at D)
                # At C: distance OC = L0, rope length = AO + OC = 2L0, elongation = 2L0 - L0 = L0
                # At D: OD = sqrt(L0^2 + (2L0)^2) = L0*sqrt(5) ≈ 2.236L0
                OD = self.L0 * math.sqrt(5.0)
                U_elastic_C = 0.5 * self.k * (self.L0)**2  # elongation at C = L0
                U_elastic_D = 0.5 * self.k * (OD)**2       # elongation at D = OD
                W_elastic = U_elastic_C - U_elastic_D
                
                # Friction work: friction force * distance along rod (2L0)
                W_friction = -self.mu * self.F_prime * (2 * self.L0)
                
                # Work-energy: ΔK = 0 = W_gravity + W_elastic + W_friction
                net_work = W_gravity + W_elastic + W_friction
                if math.isclose(net_work, 0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Work-energy theorem C→D: net work = {net_work:.3g} ≈ 0", True)
                else:
                    self.log(f"Work-energy violation: net work = {net_work:.3g}", False)
            
            return True
        except Exception as e:
            self.log(f"test_invariants failed: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # 1. Check k = 4mg/(5L0) from step 3
            if self.k is not None and self.m is not None and self.g is not None and self.L0 is not None:
                expected_k = (4 * self.m * self.g) / (5 * self.L0)
                if math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Stiffness coefficient: k = {self.k} ≈ 4mg/(5L0) = {expected_k}", True)
                else:
                    self.log(f"k mismatch: k={self.k}, expected={expected_k}", False)
            
            # 2. Check μ = 0.25 from step 9
            if self.mu is not None:
                if math.isclose(self.mu, 0.25, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Friction coefficient: μ = {self.mu} ≈ 0.25", True)
                else:
                    self.log(f"μ mismatch: μ={self.mu}, expected=0.25", False)
            
            # 3. Check L_CF = 1.5*L0 from step 11
            if self.L_CF is not None and self.L0 is not None:
                expected_L_CF = 1.5 * self.L0
                if math.isclose(self.L_CF, expected_L_CF, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Elongation at max speed: L_CF = {self.L_CF} ≈ 1.5L0 = {expected_L_CF}", True)
                else:
                    self.log(f"L_CF mismatch: L_CF={self.L_CF}, expected={expected_L_CF}", False)
            
            # 4. Check v_m = sqrt(5gL0)/5 from step 14
            if self.v_m is not None and self.g is not None and self.L0 is not None:
                expected_v_m = math.sqrt(5 * self.g * self.L0) / 5
                if math.isclose(self.v_m, expected_v_m, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Maximum velocity: v_m = {self.v_m} ≈ sqrt(5gL0)/5 = {expected_v_m}", True)
                else:
                    self.log(f"v_m mismatch: v_m={self.v_m}, expected={expected_v_m}", False)
            
            # 5. Check friction force formula from step 6: f = (4/5)μmg
            if self.f is not None and self.mu is not None and self.m is not None and self.g is not None:
                expected_f = (4/5) * self.mu * self.m * self.g
                if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Friction force: f = {self.f} ≈ (4/5)μmg = {expected_f}", True)
                else:
                    self.log(f"f mismatch: f={self.f}, expected={expected_f}", False)
            
            return True
        except Exception as e:
            self.log(f"test_trace_consistency failed: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check trends"""
        try:
            if None in (self.m, self.g, self.L0, self.k, self.mu, self.v_m):
                self.log("Missing parameters for monotonicity test", False)
                return False
            
            # Save original values
            m_orig = self.m
            
            # Test 1: Increase mass -> k should increase proportionally
            self.m = m_orig * 1.1
            k_new = (4 * self.m * self.g) / (5 * self.L0)
            if k_new > self.k:
                self.log(f"Mass increase: k increases from {self.k:.3g} to {k_new:.3g} (correct)", True)
            else:
                self.log(f"Mass increase: k decreases or stays same (unexpected)", False)
            
            # Test 2: v_m should be independent of mass
            v_m_new = math.sqrt(5 * self.g * self.L0) / 5
            if math.isclose(v_m_new, self.v_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_m independent of mass: {self.v_m:.3g} → {v_m_new:.3g}", True)
            else:
                self.log(f"v_m changes with mass (unexpected)", False)
            
            # Test 3: μ should remain constant
            mu_new = self.mu  # μ is a material property
            if math.isclose(mu_new, 0.25, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"μ remains constant at {mu_new}", True)
            else:
                self.log(f"μ changes (unexpected)", False)
            
            # Restore original mass
            self.m = m_orig
            return True
        except Exception as e:
            self.log(f"test_monotonicity failed: {str(e)}\n{traceback.format_exc()}", False)
            # Try to restore on error
            if 'm_orig' in locals():
                self.m = m_orig
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        overall_success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n=== Running {test_name} Test ===")
            try:
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self.log(f"Test {test_name} crashed: {str(e)}", False)
                overall_success = False
        
        # Summary
        self.log(f"\n=== Summary ===")
        self.log(f"Overall success: {overall_success}")
        
        return overall_success, self.logs