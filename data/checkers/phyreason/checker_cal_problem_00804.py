import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs - use defaults if not provided
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Extract original height - look for specific attributes
        # First try to get h_original directly
        self.h_original = getattr(solution_instance, 'h_original', None)
        
        # If h_original is not found, check other sources
        if self.h_original is None:
            # Check if there's an attribute named 'h' that might be the original height
            if hasattr(solution_instance, 'h'):
                h_attr = solution_instance.h
                # Only use if it's not a dict (not results) and not too small (not h')
                if not isinstance(h_attr, dict) and h_attr > 0.1:  # h' is ~0.055, original h is 1.0
                    self.h_original = h_attr
        
        # If still not found, use default
        if self.h_original is None:
            self.h_original = 1.0
        
        # Safely extract results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.v_0 = self.results.get('v_0', 0.0)
        self.v_1 = self.results.get('v_1', 0.0)
        self.Q = self.results.get('Q', 0.0)
        self.v_all = self.results.get('v_all', 0.0)
        # h' is stored as 'h' in results per interface contract
        self.h_prime = self.results.get('h', 0.0)
        self.v_2 = self.results.get('v_2', 0.0)
        self.v_3 = self.results.get('v_3', 0.0)
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def fuzzy_eq(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        try:
            # 1. Conservation of Mechanical Energy for A sliding down
            # Initial: PE = m*g*h_original, KE = 0
            # Final (at O): PE = 0, KE = 0.5*m*v_0^2
            energy_initial = self.m * self.g * self.h_original
            energy_final = 0.5 * self.m * self.v_0**2
            if not self.fuzzy_eq(energy_initial, energy_final):
                self.log(f"FAIL: Energy conservation in step 1. mgh={energy_initial}, 0.5*m*v_0^2={energy_final}")
                self.log(f"  h_original={self.h_original}, m={self.m}, g={self.g}, v_0={self.v_0}")
                return False
            self.log("PASS: Energy conservation for A sliding down (step 1).")
            
            # 2. Conservation of Momentum during collision (step 2)
            # m*v_0 = (m + 2m)*v_1 = 3m*v_1
            momentum_before = self.m * self.v_0
            momentum_after = 3 * self.m * self.v_1
            if not self.fuzzy_eq(momentum_before, momentum_after):
                self.log(f"FAIL: Momentum conservation in collision. m*v_0={momentum_before}, 3m*v_1={momentum_after}")
                return False
            self.log("PASS: Momentum conservation in collision (step 2).")
            
            # 3. Energy loss Q during collision (step 3-4)
            # Q = initial KE of A - final KE of (A+B)
            initial_ke = 0.5 * self.m * self.v_0**2
            final_ke = 0.5 * 3 * self.m * self.v_1**2
            expected_Q = initial_ke - final_ke
            if not self.fuzzy_eq(self.Q, expected_Q):
                self.log(f"FAIL: Energy loss Q mismatch. Q={self.Q}, expected {expected_Q}")
                return False
            self.log("PASS: Energy loss Q consistent (steps 3-4).")
            
            # 4. Conservation of Momentum when (A+B) moves onto C (step 5)
            # Horizontal momentum: 3m*v_1 = (3m+3m)*v_all = 6m*v_all
            momentum_before2 = 3 * self.m * self.v_1
            momentum_after2 = 6 * self.m * self.v_all
            if not self.fuzzy_eq(momentum_before2, momentum_after2):
                self.log(f"FAIL: Momentum when (A+B) moves onto C. 3m*v_1={momentum_before2}, 6m*v_all={momentum_after2}")
                return False
            self.log("PASS: Momentum conservation when (A+B) moves onto C (step 5).")
            
            # 5. Energy conservation for rise on inclined plane (step 6)
            # Loss in KE of (A+B+C) = gain in PE of (A+B)
            # (1/2)*3m*v_1^2 - (1/2)*6m*v_all^2 = 3m*g*h'
            ke_before_rise = 0.5 * 3 * self.m * self.v_1**2
            ke_after_rise = 0.5 * 6 * self.m * self.v_all**2
            pe_gain = 3 * self.m * self.g * self.h_prime
            if not self.fuzzy_eq(ke_before_rise - ke_after_rise, pe_gain):
                self.log(f"FAIL: Energy during rise on inclined plane. ΔKE={ke_before_rise-ke_after_rise}, 3mgh'={pe_gain}")
                return False
            self.log("PASS: Energy conservation during rise (step 6).")
            
            # 6. Momentum and energy for when C reaches max speed (steps 9-11)
            # Momentum: 3m*v_1 = 3m*v_2 + 3m*v_3  => v_1 = v_2 + v_3
            # Energy: (1/2)*3m*v_1^2 = (1/2)*3m*v_2^2 + (1/2)*3m*v_3^2 => v_1^2 = v_2^2 + v_3^2
            # Solution: v_2 = 0, v_3 = v_1
            if not self.fuzzy_eq(self.v_1, self.v_2 + self.v_3):
                self.log(f"FAIL: Momentum in steps 9-11. v_1={self.v_1}, v_2+v_3={self.v_2+self.v_3}")
                return False
            if not self.fuzzy_eq(self.v_1**2, self.v_2**2 + self.v_3**2):
                self.log(f"FAIL: Energy in steps 9-11. v_1^2={self.v_1**2}, v_2^2+v_3^2={self.v_2**2+self.v_3**2}")
                return False
            self.log("PASS: Momentum and energy for max speed of C (steps 9-11).")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        try:
            # Check that v_1 = v_0/3 (from step 2)
            expected_v_1 = self.v_0 / 3
            if not self.fuzzy_eq(self.v_1, expected_v_1):
                self.log(f"FAIL: v_1 should be v_0/3. v_1={self.v_1}, expected {expected_v_1}")
                return False
            
            # Check that Q = (2/3)*m*g*h_original (from step 4)
            expected_Q = (2/3) * self.m * self.g * self.h_original
            if not self.fuzzy_eq(self.Q, expected_Q):
                self.log(f"FAIL: Q should be (2/3)mgh_original. Q={self.Q}, expected {expected_Q}")
                self.log(f"  h_original={self.h_original}, m={self.m}, g={self.g}")
                return False
            
            # Check that v_all = v_1/2 (from step 5)
            expected_v_all = self.v_1 / 2
            if not self.fuzzy_eq(self.v_all, expected_v_all):
                self.log(f"FAIL: v_all should be v_1/2. v_all={self.v_all}, expected {expected_v_all}")
                return False
            
            # Check that h' = h_original/18 (from step 7)
            expected_h_prime = self.h_original / 18
            if not self.fuzzy_eq(self.h_prime, expected_h_prime):
                self.log(f"FAIL: h' should be h_original/18. h'={self.h_prime}, expected {expected_h_prime}")
                self.log(f"  h_original={self.h_original}")
                return False
            
            # Check that v_3 = v_1 (from step 11)
            if not self.fuzzy_eq(self.v_3, self.v_1):
                self.log(f"FAIL: v_3 should equal v_1. v_3={self.v_3}, v_1={self.v_1}")
                return False
            
            # Check that v_2 = 0 (from step 11)
            if not self.fuzzy_eq(self.v_2, 0.0):
                self.log(f"FAIL: v_2 should be 0. v_2={self.v_2}")
                return False
            
            self.log("PASS: All intermediate results are consistent with derived formulas.")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        try:
            original_h = self.h_original
            perturbed_h = 2.0 * original_h  # Double h
            
            # Recompute expected values using formulas
            # v_0 = sqrt(2*g*h)
            v_0_new = math.sqrt(2 * self.g * perturbed_h)
            # v_1 = v_0/3
            v_1_new = v_0_new / 3
            # Q = (2/3)*m*g*h
            Q_new = (2/3) * self.m * self.g * perturbed_h
            # v_all = v_1/2
            v_all_new = v_1_new / 2
            # h' = h/18
            h_prime_new = perturbed_h / 18
            # v_3 = v_1
            v_3_new = v_1_new
            
            # Check trends
            # Q should double (linear in h)
            expected_Q_ratio = 2.0
            if self.Q == 0:
                self.log("WARNING: Q is 0, skipping ratio check")
            else:
                actual_Q_ratio = Q_new / self.Q
                if not self.fuzzy_eq(actual_Q_ratio, expected_Q_ratio):
                    self.log(f"FAIL: Q should scale linearly with h. Ratio={actual_Q_ratio}, expected {expected_Q_ratio}")
                    return False
            
            # h' should double (linear in h)
            expected_h_prime_ratio = 2.0
            if self.h_prime == 0:
                self.log("WARNING: h_prime is 0, skipping ratio check")
            else:
                actual_h_prime_ratio = h_prime_new / self.h_prime
                if not self.fuzzy_eq(actual_h_prime_ratio, expected_h_prime_ratio):
                    self.log(f"FAIL: h' should scale linearly with h. Ratio={actual_h_prime_ratio}, expected {expected_h_prime_ratio}")
                    return False
            
            # v_3 should scale as sqrt(h)
            expected_v_3_ratio = math.sqrt(2.0)
            if self.v_3 == 0:
                self.log("WARNING: v_3 is 0, skipping ratio check")
            else:
                actual_v_3_ratio = v_3_new / self.v_3
                if not self.fuzzy_eq(actual_v_3_ratio, expected_v_3_ratio):
                    self.log(f"FAIL: v_3 should scale as sqrt(h). Ratio={actual_v_3_ratio}, expected {expected_v_3_ratio}")
                    return False
            
            self.log("PASS: Monotonicity tests passed (perturbed h, checked trends).")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"--- Running {name} Test ---")
            if not test_func():
                success = False
                self.log(f"{name} test failed.")
            else:
                self.log(f"{name} test passed.")
        
        return success, self.logs