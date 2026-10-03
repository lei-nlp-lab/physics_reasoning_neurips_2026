import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        
        Args:
            solution_instance: The solution object containing results.
        """
        self.solution = solution_instance
        
        # Safe extraction of results with fallbacks
        self.results = {}
        expected_keys = ['a_1', 'v_2', 't_2', 'v_0', 'h', 't_1', 's', 'v_1']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Problem constants
        self.l = 1.0  # m
        self.μ = 0.3  # friction coefficient
        self.g = 10.0  # m/s²
        self.Δl = self.l / 6.0  # plank displacement when block leaves
        
        # Derived constants
        self.m = 1.0  # mass (relative, since it cancels in equations)
        self.expected_s = 7.0 * self.l / 6.0  # expected block displacement
        
    def fuzzy_equal(self, a: float, b: float) -> bool:
        """Check if two numbers are approximately equal with physics tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants:
        1. Conservation of momentum
        2. Work-energy theorem
        3. Newton's second law for block
        4. Kinematic consistency
        """
        logs = []
        success = True
        
        try:
            # Extract values with defaults
            v0 = float(self.results.get('v_0', 0.0))
            v1 = float(self.results.get('v_1', 0.0))
            v2 = float(self.results.get('v_2', 0.0))
            a1 = float(self.results.get('a_1', 0.0))
            s = float(self.results.get('s', 0.0))
            
            # 1. Conservation of momentum: m*v0 = m*v1 + m*v2
            momentum_lhs = v0
            momentum_rhs = v1 + v2
            if not self.fuzzy_equal(momentum_lhs, momentum_rhs):
                logs.append(f"FAIL: Momentum conservation: v0={v0:.3f} ≠ v1+v2={v1+v2:.3f}")
                success = False
            else:
                logs.append(f"PASS: Momentum conservation: v0={v0:.3f} ≈ v1+v2={v1+v2:.3f}")
            
            # 2. Work-energy theorem: μmg*l = 0.5*m*v0² - 0.5*m*v1² - 0.5*m*v2²
            work_lhs = self.μ * self.g * self.l
            work_rhs = 0.5 * (v0**2 - v1**2 - v2**2)
            if not self.fuzzy_equal(work_lhs, work_rhs):
                logs.append(f"FAIL: Work-energy: μgl={work_lhs:.3f} ≠ ΔKE={work_rhs:.3f}")
                success = False
            else:
                logs.append(f"PASS: Work-energy: μgl={work_lhs:.3f} ≈ ΔKE={work_rhs:.3f}")
            
            # 3. Newton's second law for block: a1 = μ*g
            expected_a1 = self.μ * self.g
            if not self.fuzzy_equal(a1, expected_a1):
                logs.append(f"FAIL: Newton's 2nd law: a1={a1:.3f} ≠ μg={expected_a1:.3f}")
                success = False
            else:
                logs.append(f"PASS: Newton's 2nd law: a1={a1:.3f} ≈ μg={expected_a1:.3f}")
            
            # 4. Kinematic equation: v0² - v1² = 2*a1*s
            kinematic_lhs = v0**2 - v1**2
            kinematic_rhs = 2 * a1 * s
            if not self.fuzzy_equal(kinematic_lhs, kinematic_rhs):
                logs.append(f"FAIL: Kinematics: v0²-v1²={kinematic_lhs:.3f} ≠ 2*a1*s={kinematic_rhs:.3f}")
                success = False
            else:
                logs.append(f"PASS: Kinematics: v0²-v1²={kinematic_lhs:.3f} ≈ 2*a1*s={kinematic_rhs:.3f}")
            
            # 5. Check block displacement s = 7l/6
            if not self.fuzzy_equal(s, self.expected_s):
                logs.append(f"FAIL: Displacement s={s:.3f} ≠ 7l/6={self.expected_s:.3f}")
                success = False
            else:
                logs.append(f"PASS: Displacement s={s:.3f} ≈ 7l/6={self.expected_s:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Test consistency between intermediate results and final answers.
        """
        logs = []
        success = True
        
        try:
            # Extract values
            v0 = float(self.results.get('v_0', 0.0))
            v1 = float(self.results.get('v_1', 0.0))
            v2 = float(self.results.get('v_2', 0.0))
            a1 = float(self.results.get('a_1', 0.0))
            t1 = float(self.results.get('t_1', 0.0))
            t2 = float(self.results.get('t_2', 0.0))
            h = float(self.results.get('h', 0.0))
            
            # 1. Check t1 from kinematics: v0 - v1 = a1*t1
            t1_calc = (v0 - v1) / a1 if abs(a1) > 1e-9 else 0.0
            if not self.fuzzy_equal(t1, t1_calc):
                logs.append(f"FAIL: t1: {t1:.3f} ≠ (v0-v1)/a1={t1_calc:.3f}")
                success = False
            else:
                logs.append(f"PASS: t1: {t1:.3f} ≈ (v0-v1)/a1={t1_calc:.3f}")
            
            # 2. Check t2 from uniform motion: (l/2 - Δl) = v2*t2
            t2_calc = (self.l/2.0 - self.Δl) / v2 if abs(v2) > 1e-9 else 0.0
            if not self.fuzzy_equal(t2, t2_calc):
                logs.append(f"FAIL: t2: {t2:.3f} ≠ (l/2-Δl)/v2={t2_calc:.3f}")
                success = False
            else:
                logs.append(f"PASS: t2: {t2:.3f} ≈ (l/2-Δl)/v2={t2_calc:.3f}")
            
            # 3. Check h from free fall: h = 0.5*g*t2²
            h_calc = 0.5 * self.g * t2**2
            if not self.fuzzy_equal(h, h_calc):
                logs.append(f"FAIL: h: {h:.3f} ≠ 0.5*g*t2²={h_calc:.3f}")
                success = False
            else:
                logs.append(f"PASS: h: {h:.3f} ≈ 0.5*g*t2²={h_calc:.3f}")
            
            # 4. Check v0, v1, v2 from simultaneous equations
            # Solve symbolically for verification
            v0_sym, v1_sym, v2_sym = sp.symbols('v0 v1 v2', real=True)
            eq1 = sp.Eq(v0_sym, v1_sym + v2_sym)  # Momentum
            eq2 = sp.Eq(self.μ * self.g * self.l, 
                      0.5*(v0_sym**2 - v1_sym**2 - v2_sym**2))  # Work-energy
            eq3 = sp.Eq(v0_sym**2 - v1_sym**2, 
                      2 * self.μ * self.g * self.expected_s)  # Kinematics
            
            sol = sp.solve([eq1, eq2, eq3], (v0_sym, v1_sym, v2_sym), dict=True)
            if sol:
                # Take positive solutions (physics meaningful)
                for s in sol:
                    if all(val > 0 for val in [s[v0_sym], s[v1_sym], s[v2_sym]]):
                        v0_sol = float(s[v0_sym])
                        v1_sol = float(s[v1_sym])
                        v2_sol = float(s[v2_sym])
                        
                        if (self.fuzzy_equal(v0, v0_sol) and 
                            self.fuzzy_equal(v1, v1_sol) and 
                            self.fuzzy_equal(v2, v2_sol)):
                            logs.append(f"PASS: v0,v1,v2 match symbolic solution")
                        else:
                            logs.append(f"FAIL: v0({v0:.3f}),v1({v1:.3f}),v2({v2:.3f}) ≠ symbolic({v0_sol:.3f},{v1_sol:.3f},{v2_sol:.3f})")
                            success = False
                        break
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Test monotonicity by perturbing μ and checking trend.
        Expected: Increasing μ should decrease t2 and h.
        """
        logs = []
        success = True
        
        try:
            # Original parameters
            μ_orig = self.μ
            v0 = float(self.results.get('v_0', 0.0))
            v1 = float(self.results.get('v_1', 0.0))
            v2 = float(self.results.get('v_2', 0.0))
            t2_orig = float(self.results.get('t_2', 0.0))
            h_orig = float(self.results.get('h', 0.0))
            
            # Perturb μ by +10%
            μ_pert = μ_orig * 1.1
            
            # Recalculate using formulas from solution steps
            # 1. New acceleration
            a1_pert = μ_pert * self.g
            
            # 2. Solve for v2_pert from work-energy and kinematics
            # From work-energy and kinematics combined: v2² = μ*g*l/3 (derived in solution)
            v2_pert = math.sqrt(μ_pert * self.g * self.l / 3.0)
            
            # 3. Solve for v0_pert and v1_pert
            # v0 = v1 + v2 (momentum)
            # v0² - v1² = 2*μ*g*(7l/6) (kinematics)
            # Let x = v0, y = v1, then:
            # x = y + v2_pert
            # x² - y² = (14/3)*μ_pert*g*l
            # => (y + v2_pert)² - y² = (14/3)*μ_pert*g*l
            # => y² + 2*y*v2_pert + v2_pert² - y² = (14/3)*μ_pert*g*l
            # => 2*y*v2_pert = (14/3)*μ_pert*g*l - v2_pert²
            rhs = (14.0/3.0) * μ_pert * self.g * self.l - v2_pert**2
            v1_pert = rhs / (2.0 * v2_pert) if abs(v2_pert) > 1e-9 else 0.0
            v0_pert = v1_pert + v2_pert
            
            # 4. Calculate new t2 and h
            t2_pert = (self.l/2.0 - self.Δl) / v2_pert if abs(v2_pert) > 1e-9 else 0.0
            h_pert = 0.5 * self.g * t2_pert**2
            
            # Check monotonicity: increasing μ should decrease t2 and h
            if t2_pert < t2_orig and h_pert < h_orig:
                logs.append(f"PASS: Monotonicity: μ↑ ({μ_orig:.3f}→{μ_pert:.3f}) → t2↓ ({t2_orig:.3f}→{t2_pert:.3f}), h↓ ({h_orig:.3f}→{h_pert:.3f})")
            else:
                logs.append(f"FAIL: Monotonicity violated for μ perturbation")
                logs.append(f"  μ: {μ_orig:.3f} → {μ_pert:.3f}")
                logs.append(f"  t2: {t2_orig:.3f} → {t2_pert:.3f}")
                logs.append(f"  h: {h_orig:.3f} → {h_pert:.3f}")
                success = False
                
            # Also check physical bounds
            if v0_pert <= 0 or v1_pert <= 0 or v2_pert <= 0:
                logs.append(f"WARN: Perturbed velocities non-physical: v0={v0_pert:.3f}, v1={v1_pert:.3f}, v2={v2_pert:.3f}")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        all_logs = ["=== PhysicsChecker Report ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n--- Test 1: Physics Invariants ---")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n--- Test 2: Trace Consistency ---")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n--- Test 3: Monotonicity ---")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== Summary ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs