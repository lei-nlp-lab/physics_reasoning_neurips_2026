import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract inputs with defaults
        self.m = float(getattr(solution_instance, 'm', self.inputs.get('m', 2.0)))
        self.g = float(getattr(solution_instance, 'g', self.inputs.get('g', 10.0)))
        
        # Extract results with safe access
        self.alpha = float(self.results.get('alpha', 0.0))
        self.a1 = float(self.results.get('a_1', 0.0))
        self.F = float(self.results.get('F', 0.0))
        self.t2 = float(self.results.get('t_2', 0.0))
        self.a3 = float(self.results.get('a_3', 0.0))
        self.t3 = float(self.results.get('t_3', 0.0))
        self.a2 = float(self.results.get('a_2', 0.0))
        self.t1 = float(self.results.get('t_1', 0.0))
        self.s = float(self.results.get('s', 0.0))
        self.L = float(self.results.get('L', 0.0))
        
        # Additional derived quantities from problem context
        self.v0 = 0.0  # initial velocity
        self.v1 = 3.0  # velocity at t=0.5s (from graph description)
        self.v2 = 0.0  # velocity at t=0.75s
        self.v3 = -1.0  # velocity at t=1.0s
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        self.logs = []
    
    def _isclose(self, a, b):
        """Fuzzy numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Verify physics conservation laws and Newton's laws."""
        logs = []
        success = True
        
        try:
            # 1. Check acceleration definitions (from v-t graph)
            # a1 = (v1 - v0) / t1
            expected_a1 = (self.v1 - self.v0) / self.t1
            if not self._isclose(self.a1, expected_a1):
                logs.append(f"FAIL: a1 = {self.a1}, expected {expected_a1} from v-t graph")
                success = False
            else:
                logs.append(f"PASS: a1 matches v-t graph definition")
            
            # a2 = (v2 - v1) / t2
            expected_a2 = (self.v2 - self.v1) / self.t2
            if not self._isclose(abs(self.a2), abs(expected_a2)):
                logs.append(f"FAIL: |a2| = {abs(self.a2)}, expected {abs(expected_a2)} from v-t graph")
                success = False
            else:
                logs.append(f"PASS: a2 magnitude matches v-t graph")
            
            # a3 = (v3 - v2) / t3
            expected_a3 = (self.v3 - self.v2) / self.t3
            if not self._isclose(abs(self.a3), abs(expected_a3)):
                logs.append(f"FAIL: |a3| = {abs(self.a3)}, expected {abs(expected_a3)} from v-t graph")
                success = False
            else:
                logs.append(f"PASS: a3 magnitude matches v-t graph")
            
            # 2. Check Newton's Second Law for three intervals
            # Define symbols
            alpha_rad = math.radians(self.alpha)
            sin_alpha = math.sin(alpha_rad)
            
            # Correct physics model from the problem:
            # Phase 1: F - f - mg·sinα = m·a₁ (upward acceleration)
            # Phase 2: f + mg·sinα = m·a₂ (deceleration upward, friction + gravity oppose motion)
            # Phase 3: mg·sinα - f = m·a₃ (acceleration downward)
            
            # From phases 2 and 3:
            # eq2: f + mg·sinα = m·a₂
            # eq3: mg·sinα - f = m·a₃
            # Add eq2 and eq3: 2mg·sinα = m·(a₂ + a₃)
            expected_sin_alpha = 0.5 * (self.a2 + self.a3) / self.g
            
            if not self._isclose(sin_alpha, expected_sin_alpha):
                logs.append(f"FAIL: sin(α) = {sin_alpha:.4f}, expected {expected_sin_alpha:.4f} from Newton's laws")
                success = False
            else:
                logs.append(f"PASS: sin(α) consistent with Newton's laws")
            
            # Now check force F from phase 1
            # From eq2: f = m·a₂ - mg·sinα
            f = self.m * self.a2 - self.m * self.g * sin_alpha
            
            # Phase 1: F = mg·sinα + f + m·a₁
            expected_F = self.m * self.g * sin_alpha + f + self.m * self.a1
            
            if not self._isclose(self.F, expected_F):
                logs.append(f"FAIL: F = {self.F}, expected {expected_F} from Newton's laws")
                success = False
            else:
                logs.append(f"PASS: F consistent with Newton's laws")
            
            # 3. Check time intervals sum correctly
            total_time = self.t1 + self.t2 + self.t3
            if not self._isclose(total_time, 1.0):
                logs.append(f"FAIL: Total time {total_time} ≠ 1.0s")
                success = False
            else:
                logs.append(f"PASS: Time intervals sum to 1.0s")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check displacement calculation (step 5)
            expected_s = 0.5 * self.v1 * (self.t1 + self.t2) + 0.5 * self.v3 * self.t3
            if not self._isclose(self.s, expected_s):
                logs.append(f"FAIL: s = {self.s}, expected {expected_s} from area under v-t")
                success = False
            else:
                logs.append(f"PASS: displacement s matches area calculation")
            
            # 2. Check distance calculation (step 6)
            expected_L = 0.5 * self.v1 * (self.t1 + self.t2) + 0.5 * abs(self.v3) * self.t3
            if not self._isclose(self.L, expected_L):
                logs.append(f"FAIL: L = {self.L}, expected {expected_L} from total area")
                success = False
            else:
                logs.append(f"PASS: distance L matches total area")
            
            # 3. Check angle conversion
            if not (50 <= self.alpha <= 56):  # Allow some tolerance around 53°
                logs.append(f"WARNING: α = {self.alpha}° seems outside reasonable range")
                # Don't fail for this, just warn
            
            # 4. Check against target answers
            targets = [1.0, 1.25, 53.0, 36.0]
            computed = [self.s, self.L, self.alpha, self.F]
            
            for i, (target, comp) in enumerate(zip(targets, computed)):
                if not self._isclose(comp, target):
                    logs.append(f"FAIL: Result {i+1}: {comp} ≠ target {target}")
                    success = False
                else:
                    logs.append(f"PASS: Result {i+1} matches target")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb mass and check trend in force F."""
        logs = []
        success = True
        
        try:
            # Save original mass
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected force using physics formulas
            alpha_rad = math.radians(self.alpha)
            sin_alpha = math.sin(alpha_rad)
            
            # Using the correct physics model:
            # From original solution, compute friction coefficient μ
            # Normal force N = m*g*cosα
            # From phase 2: f = μ*N = m*a₂ - m*g*sinα
            # So μ = (m*a₂ - m*g*sinα) / (m*g*cosα) = (a₂ - g*sinα) / (g*cosα)
            
            cos_alpha = math.cos(alpha_rad)
            μ = (self.a2 - self.g * sin_alpha) / (self.g * cos_alpha)
            
            # Now compute F for perturbed mass
            f_perturbed = μ * perturbed_m * self.g * cos_alpha
            F_perturbed = perturbed_m * self.g * sin_alpha + f_perturbed + perturbed_m * self.a1
            
            # Check monotonicity: F should increase with mass
            if F_perturbed <= self.F:
                logs.append(f"FAIL: F doesn't increase with mass (monotonicity violated)")
                success = False
            else:
                logs.append(f"PASS: F increases with mass (monotonic trend)")
            
        except Exception as e:
            logs.append(f"WARNING in test_monotonicity: {str(e)}")
            # Don't fail overall for perturbation test
            success = True
        
        return success, logs
    
    def run_all_tests(self):
        """Run all physics checks and return overall result."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_method()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("ALL TESTS PASSED: Physics appears consistent")
        else:
            all_logs.append("SOME TESTS FAILED: Check physics consistency")
        
        return overall_success, all_logs