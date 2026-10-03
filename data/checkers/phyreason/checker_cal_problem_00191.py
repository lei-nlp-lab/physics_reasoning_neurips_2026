import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.S = getattr(solution_instance, 'S', 0.01)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.p0 = getattr(solution_instance, 'p0', 1e5)
        self.T0 = getattr(solution_instance, 'T0', 300.0)
        
        # Safe extraction of results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.T_2 = results.get('T_2', 0.0)
        self.V_1 = results.get('V_1', 0.0)
        self.p_1 = results.get('p_1', 0.0)
        self.l_2 = results.get('l_2', 0.0)
        self.V_2 = results.get('V_2', 0.0)
        self.k = results.get('k', 0.0)
        
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def fuzzy_eq(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Check force equilibrium and spring force consistency."""
        try:
            # 1. Check initial pressure p1 from step_1 equation
            p1_expected = self.p0 + 3 * self.m * self.g / self.S
            if not self.fuzzy_eq(self.p_1, p1_expected):
                self.log(f"Initial pressure mismatch: expected {p1_expected}, got {self.p_1}", False)
                return False
            self.log(f"Initial pressure p1 correct: {self.p_1} ≈ {p1_expected}", True)
            
            # 2. Check spring constant k from step_2 equation
            k_expected = 40 * self.m * self.g / self.l
            if not self.fuzzy_eq(self.k, k_expected):
                self.log(f"Spring constant mismatch: expected {k_expected}, got {self.k}", False)
                return False
            self.log(f"Spring constant k correct: {self.k} ≈ {k_expected}", True)
            
            # 3. Check force balance on piston I (mass 2m, area 2S) initially
            # Forces downward: weight (2mg) + atmospheric (p0*2S) + spring (k*0.1l)
            # Forces upward: gas pressure (p1*2S)
            F_down = 2*self.m*self.g + self.p0*2*self.S + self.k*0.1*self.l
            F_up = self.p_1 * 2 * self.S
            if not self.fuzzy_eq(F_down, F_up):
                self.log(f"Initial force balance on piston I fails: ΣF_down={F_down}, ΣF_up={F_up}", False)
                return False
            self.log(f"Initial force balance on piston I holds", True)
            
            # 4. Check force balance on piston II (mass m, area S) initially
            # Forces downward: weight (mg) + gas pressure (p1*S)
            # Forces upward: atmospheric (p0*S) + spring (k*0.1l)
            F_down_II = self.m*self.g + self.p_1*self.S
            F_up_II = self.p0*self.S + self.k*0.1*self.l
            if not self.fuzzy_eq(F_down_II, F_up_II):
                self.log(f"Initial force balance on piston II fails: ΣF_down={F_down_II}, ΣF_up={F_up_II}", False)
                return False
            self.log(f"Initial force balance on piston II holds", True)
            
            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check initial volume V1 from step_4
            V1_expected = (3.3 * self.l * self.S) / 2
            if not self.fuzzy_eq(self.V_1, V1_expected):
                self.log(f"Initial volume mismatch: expected {V1_expected}, got {self.V_1}", False)
                return False
            self.log(f"Initial volume V1 correct: {self.V_1} ≈ {V1_expected}", True)
            
            # 2. Check final spring length l2 from step_5
            l2_expected = 1.1 * self.l
            if not self.fuzzy_eq(self.l_2, l2_expected):
                self.log(f"Final spring length mismatch: expected {l2_expected}, got {self.l_2}", False)
                return False
            self.log(f"Final spring length l2 correct: {self.l_2} ≈ {l2_expected}", True)
            
            # 3. Check final volume V2 from step_4
            V2_expected = self.l_2 * 2 * self.S
            if not self.fuzzy_eq(self.V_2, V2_expected):
                self.log(f"Final volume mismatch: expected {V2_expected}, got {self.V_2}", False)
                return False
            self.log(f"Final volume V2 correct: {self.V_2} ≈ {V2_expected}", True)
            
            # 4. Check final temperature T2 from step_6 (isobaric process)
            # Using V1/T0 = V2/T2 => T2 = T0 * V2 / V1
            T2_expected = self.T0 * self.V_2 / self.V_1
            if not self.fuzzy_eq(self.T_2, T2_expected):
                self.log(f"Final temperature mismatch: expected {T2_expected}, got {self.T_2}", False)
                return False
            self.log(f"Final temperature T2 correct: {self.T_2} ≈ {T2_expected}", True)
            
            # 5. Check target answers consistency
            k_target = 40 * self.m * self.g / self.l
            p2_target = self.p0 + 3 * self.m * self.g / self.S
            T2_target = (4/3) * self.T0
            
            if not (self.fuzzy_eq(self.k, k_target) and 
                    self.fuzzy_eq(self.p_1, p2_target) and 
                    self.fuzzy_eq(self.T_2, T2_target)):
                self.log(f"Target answers mismatch: k={self.k} vs {k_target}, p1={self.p_1} vs {p2_target}, T2={self.T_2} vs {T2_target}", False)
                return False
            self.log("All target answers consistent", True)
            
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check trends."""
        original_m = self.m
        try:
            # Perturb m by +10%
            self.m *= 1.1
            
            # Recalculate expected values using formulas from steps
            p1_new = self.p0 + 3 * self.m * self.g / self.S
            k_new = 40 * self.m * self.g / self.l
            V1_new = (3.3 * self.l * self.S) / 2  # independent of m
            l2_new = 1.1 * self.l  # independent of m
            V2_new = l2_new * 2 * self.S  # independent of m
            T2_new = self.T0 * V2_new / V1_new  # independent of m
            
            # Check trends
            trends_ok = True
            if not (p1_new > self.p_1):  # p1 should increase with m
                self.log(f"Monotonicity: p1 did not increase with m (old={self.p_1}, new={p1_new})", False)
                trends_ok = False
            if not (k_new > self.k):  # k should increase with m
                self.log(f"Monotonicity: k did not increase with m (old={self.k}, new={k_new})", False)
                trends_ok = False
            if not self.fuzzy_eq(T2_new, self.T_2):  # T2 should remain constant
                self.log(f"Monotonicity: T2 changed with m (old={self.T_2}, new={T2_new})", False)
                trends_ok = False
            
            if trends_ok:
                self.log("All monotonicity checks passed: p1 and k increase with m, T2 constant", True)
            
            # Restore original m
            self.m = original_m
            return trends_ok
        except Exception as e:
            self.m = original_m
            self.log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            if not test_func():
                success = False
        
        self.log(f"Overall result: {'PASS' if success else 'FAIL'}")
        return success, self.logs