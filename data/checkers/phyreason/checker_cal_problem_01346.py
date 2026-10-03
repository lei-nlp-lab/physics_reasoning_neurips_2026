import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for all inputs and results
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with defaults from problem statement
        self.L = getattr(solution_instance, 'L', 5.0)
        self.h = getattr(solution_instance, 'h', 3.0)
        self.m = getattr(solution_instance, 'm', 20.0)
        self.f1 = getattr(solution_instance, 'f1', 88.0)
        self.f2 = getattr(solution_instance, 'f2', 100.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks
        self.x = self.results.get('x', 0.0)
        self.v = self.results.get('v', 0.0)
        self.W_f1 = self.results.get('W_f1', 0.0)
        
        # For symbolic calculations
        self.symbols = sp.symbols('L h m f1 f2 g x v W_f1', real=True, positive=True)
        L_s, h_s, m_s, f1_s, f2_s, g_s, x_s, v_s, W_f1_s = self.symbols
        
        # Symbolic equations from KB and steps analysis
        self.eq_W_f1 = sp.Eq(W_f1_s, f1_s * L_s)  # Step 1
        self.eq_incline_energy = sp.Eq(m_s * g_s * h_s - W_f1_s, sp.Rational(1, 2) * m_s * v_s**2)  # Step 2
        self.eq_horizontal_energy = sp.Eq(-f2_s * x_s, -sp.Rational(1, 2) * m_s * v_s**2)  # Step 3
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Check work done by friction on incline
            expected_W_f1 = self.f1 * self.L
            if not math.isclose(self.W_f1, expected_W_f1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: W_f1 = {self.W_f1}, expected {expected_W_f1}")
            else:
                logs.append(f"PASS: W_f1 = {self.W_f1} matches f1*L = {expected_W_f1}")
                
            # 2. Check work-energy theorem on incline (symbolic and numeric)
            lhs_incline = self.m * self.g * self.h - self.W_f1
            rhs_incline = 0.5 * self.m * self.v**2
            if not math.isclose(lhs_incline, rhs_incline, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: mgh - W_f1 = {lhs_incline}, (1/2)mv² = {rhs_incline}")
            else:
                logs.append(f"PASS: Energy balance on incline satisfied")
                
            # 3. Check work-energy theorem on horizontal (symbolic and numeric)
            lhs_horizontal = -self.f2 * self.x
            rhs_horizontal = -0.5 * self.m * self.v**2
            if not math.isclose(lhs_horizontal, rhs_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: -f2*x = {lhs_horizontal}, -(1/2)mv² = {rhs_horizontal}")
            else:
                logs.append(f"PASS: Energy balance on horizontal satisfied")
                
            # 4. Sanity check: v should be less than frictionless case
            v_frictionless = math.sqrt(2 * self.g * self.h)
            if self.v > v_frictionless + 0.1:  # Allow small tolerance
                logs.append(f"WARN: v = {self.v} exceeds frictionless case {v_frictionless:.2f}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Recompute v from W_f1 using step 2 equation
            v_recomputed = math.sqrt(2 * (self.m * self.g * self.h - self.W_f1) / self.m)
            if not math.isclose(self.v, v_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v = {self.v} inconsistent with W_f1; recomputed v = {v_recomputed}")
            else:
                logs.append(f"PASS: v consistent with W_f1")
                
            # Recompute x from v using step 3 equation
            x_recomputed = 0.5 * self.m * self.v**2 / self.f2
            if not math.isclose(self.x, x_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {self.x} inconsistent with v; recomputed x = {x_recomputed}")
            else:
                logs.append(f"PASS: x consistent with v")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_f1 = self.f1
        try:
            # Perturb f1 by +10% and observe effects
            perturbed_f1 = original_f1 * 1.1
            # Recompute W_f1
            W_f1_perturbed = perturbed_f1 * self.L
            # Recompute v using energy on incline
            v_perturbed = math.sqrt(2 * (self.m * self.g * self.h - W_f1_perturbed) / self.m)
            # Recompute x using energy on horizontal
            x_perturbed = 0.5 * self.m * v_perturbed**2 / self.f2
            
            # Check monotonic trends
            if not (v_perturbed < self.v):
                logs.append(f"FAIL: Increasing f1 did not decrease v (v_perturbed={v_perturbed}, original={self.v})")
            else:
                logs.append(f"PASS: v decreases with increased friction")
                
            if not (x_perturbed < self.x):
                logs.append(f"FAIL: Increasing f1 did not decrease x (x_perturbed={x_perturbed}, original={self.x})")
            else:
                logs.append(f"PASS: x decreases with increased friction")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original value
            self.f1 = original_f1
        return logs
    
    def run_all_tests(self):
        all_logs = []
        all_logs.append("=== PhysicsChecker Starting ===")
        
        # Run individual test suites
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        all_logs.append(f"=== PhysicsChecker Finished: {'SUCCESS' if success else 'FAILED'} ===")
        return success, all_logs

if __name__ == "__main__":
    # Example minimal test harness
    class DummySolution:
        def __init__(self):
            self.L = 5.0
            self.h = 3.0
            self.m = 20.0
            self.f1 = 88.0
            self.f2 = 100.0
            self.g = 10.0
            self.results = {
                'W_f1': 440.0,
                'v': 4.0,
                'x': 1.6
            }
    
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)