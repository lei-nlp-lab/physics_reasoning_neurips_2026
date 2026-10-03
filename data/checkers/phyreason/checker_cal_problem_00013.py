import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction using getattr with fallbacks
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 250.0))
        self.theta_deg = getattr(solution_instance, 'theta', getattr(solution_instance, 'results', {}).get('theta', 22.5))
        self.x = getattr(solution_instance, 'x', getattr(solution_instance, 'results', {}).get('x', 20.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 15.0))
        
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.f = self.results.get('f', 0.0)
        self.W = self.results.get('W', 0.0)
        self.P = self.results.get('P', 0.0)
        
        # Derived quantities
        self.theta = math.radians(self.theta_deg)
        self.cos_theta = math.cos(self.theta)  # Will compare with given 0.9
        self.v = self.x / self.t  # Constant velocity
        
        # For symbolic calculations
        self.F_sym, self.theta_sym, self.x_sym, self.t_sym, self.f_sym, self.v_sym = sp.symbols('F theta x t f v')
        
    def test_invariants(self) -> List[str]:
        logs = []
        
        # 1. Equilibrium condition (Newton's 1st Law) - Net horizontal force = 0
        try:
            # Symbolic: 2Fcosθ - f = 0
            equilibrium_eq = sp.Eq(2*self.F_sym*sp.cos(self.theta_sym) - self.f_sym, 0)
            symbolic_check = equilibrium_eq.subs({
                self.F_sym: self.F,
                self.theta_sym: self.theta,
                self.f_sym: self.f
            })
            if not math.isclose(float(symbolic_check.lhs), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Equilibrium violated. 2Fcosθ - f = {2*self.F*self.cos_theta - self.f:.2f} ≠ 0")
            else:
                logs.append("PASS: Horizontal force equilibrium maintained (2Fcosθ = f)")
        except Exception as e:
            # Numerical fallback
            net_force = 2 * self.F * self.cos_theta - self.f
            if not math.isclose(net_force, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Equilibrium violated. Net horizontal force = {net_force:.2f} N")
            else:
                logs.append("PASS: Horizontal force equilibrium maintained (2Fcosθ = f)")
        
        # 2. Work-energy consistency for constant velocity
        try:
            # Work done by tension forces should equal force × displacement
            W_expected = 2 * self.F * self.cos_theta * self.x
            if not math.isclose(self.W, W_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work mismatch. W={self.W:.1f} J, expected {W_expected:.1f} J")
            else:
                logs.append("PASS: Work calculation consistent (W = 2Fcosθ × x)")
        except Exception as e:
            logs.append(f"ERROR in work-energy check: {str(e)}")
        
        # 3. Power consistency
        try:
            P_expected = self.W / self.t
            if not math.isclose(self.P, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Power mismatch. P={self.P:.1f} W, expected {P_expected:.1f} W")
            else:
                logs.append("PASS: Power calculation consistent (P = W/t)")
        except Exception as e:
            logs.append(f"ERROR in power check: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        # Check step-by-step consistency from the steps analysis
        # Step 1: f = 2Fcosθ
        f_expected = 2 * self.F * self.cos_theta
        if not math.isclose(self.f, f_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 1 inconsistency. f={self.f:.1f} N, expected {f_expected:.1f} N")
        else:
            logs.append("PASS: Step 1 consistent (f = 2Fcosθ)")
        
        # Step 2: W = 2Fcosθ × x = f × x
        W_from_step2 = self.f * self.x
        if not math.isclose(self.W, W_from_step2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 2 inconsistency. W={self.W:.1f} J, expected from step1: {W_from_step2:.1f} J")
        else:
            logs.append("PASS: Step 2 consistent (W = f × x)")
        
        # Step 3: P = W/t
        P_from_step3 = self.W / self.t
        if not math.isclose(self.P, P_from_step3, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 3 inconsistency. P={self.P:.1f} W, expected from step2: {P_from_step3:.1f} W")
        else:
            logs.append("PASS: Step 3 consistent (P = W/t)")
        
        # Overall chain: 2Fcosθ → f → W → P
        P_from_chain = 2 * self.F * self.cos_theta * self.x / self.t
        if not math.isclose(self.P, P_from_chain, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Overall chain broken. Final P={self.P:.1f} W, expected from chain: {P_from_chain:.1f} W")
        else:
            logs.append("PASS: Overall chain consistent (2Fcosθ → f → W → P)")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        logs = []
        original_F = self.F
        
        try:
            # Test 1: Increase force by 5%
            perturbed_F = original_F * 1.05
            perturbed_f = 2 * perturbed_F * self.cos_theta
            perturbed_W = perturbed_f * self.x
            perturbed_P = perturbed_W / self.t
            
            # All quantities should increase
            if not (perturbed_f > self.f and perturbed_W > self.W and perturbed_P > self.P):
                logs.append("FAIL: Monotonicity test (increase F) - not all quantities increased")
            else:
                # Check approximate scaling (should be ~5% increase)
                f_increase = (perturbed_f - self.f) / self.f
                if math.isclose(f_increase, 0.05, rel_tol=2e-2, abs_tol=2e-2):
                    logs.append("PASS: Monotonicity test (increase F) - quantities increased proportionally")
                else:
                    logs.append(f"WARNING: Monotonicity test (increase F) - f increased by {f_increase:.1%}, expected ~5%")
            
            # Test 2: Decrease displacement by 10%
            original_x = self.x
            perturbed_x = original_x * 0.9
            perturbed_W2 = 2 * self.F * self.cos_theta * perturbed_x
            perturbed_P2 = perturbed_W2 / self.t
            
            # W should decrease, P should decrease proportionally
            if not (perturbed_W2 < self.W and perturbed_P2 < self.P):
                logs.append("FAIL: Monotonicity test (decrease x) - work or power didn't decrease")
            else:
                # Check scaling (W should decrease by ~10%)
                W_decrease = (self.W - perturbed_W2) / self.W
                if math.isclose(W_decrease, 0.10, rel_tol=2e-2, abs_tol=2e-2):
                    logs.append("PASS: Monotonicity test (decrease x) - work decreased proportionally")
                else:
                    logs.append(f"WARNING: Monotonicity test (decrease x) - W decreased by {W_decrease:.1%}, expected ~10%")
                    
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        
        all_logs.append("=== PhysicsChecker Test Results ===")
        all_logs.append(f"Inputs: F={self.F} N, θ={self.theta_deg}°, x={self.x} m, t={self.t} s")
        all_logs.append(f"Results: f={self.f} N, W={self.W} J, P={self.P} W")
        
        # Run individual tests
        all_logs.append("\n--- Invariant Tests ---")
        invariant_logs = self.test_invariants()
        all_logs.extend(invariant_logs)
        
        all_logs.append("\n--- Trace Consistency Tests ---")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        
        all_logs.append("\n--- Monotonicity Tests ---")
        monotonicity_logs = self.test_monotonicity()
        all_logs.extend(monotonicity_logs)
        
        # Determine overall success
        success = True
        for log in invariant_logs + trace_logs + monotonicity_logs:
            if log.startswith("FAIL:") or log.startswith("ERROR:"):
                success = False
                break
        
        all_logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs