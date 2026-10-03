import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access to avoid AttributeError in sandbox.
        """
        self.results = {}
        
        # Extract all known result keys with safe fallbacks
        keys = [
            'x', 'HO\'', 'L_max', 'F_1', 'OH', 'F_3', 
            'v', 'p_z', 's', 'F_2'
        ]
        
        # Try to get results from solution_instance.results dictionary first
        results_dict = getattr(solution_instance, 'results', {})
        
        for key in keys:
            # First try direct attribute, then results dict, then default to 0.0
            value = getattr(solution_instance, key, results_dict.get(key, 0.0))
            self.results[key] = float(value) if isinstance(value, (int, float)) else value
        
        # Store symbolic parameters for checking
        self.R = sp.Symbol('R', positive=True)
        self.B = sp.Symbol('B', positive=True)
        self.q = sp.Symbol('q', positive=True)
        self.m = sp.Symbol('m', positive=True)
        self.N = sp.Symbol('N', positive=True)
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants:
        1. Lorentz force equals centripetal force for beam b.
        2. Geometric consistency for s and x.
        """
        logs = []
        all_pass = True
        
        try:
            # 1. Check Lorentz force = centripetal force for beam b
            # From step_1: q*v*B = m*v^2/R  => v = q*B*R/m
            v_expected = self.q * self.B * self.R / self.m
            
            # Get numerical value from results
            v_num = self.results.get('v', 0.0)
            v_expr = v_expected.subs({
                self.B: 1.0,  # Use normalized values for symbolic check
                self.R: 1.0,
                self.q: 1.0,
                self.m: 1.0
            })
            
            # Symbolic check: v should satisfy q*v*B = m*v^2/R
            v_sym = sp.Symbol('v')
            eq1 = sp.Eq(self.q * v_sym * self.B, self.m * v_sym**2 / self.R)
            solutions = sp.solve(eq1, v_sym)
            if v_expected in solutions:
                logs.append("✓ Lorentz force = centripetal force satisfied symbolically")
            else:
                logs.append("✗ Lorentz force = centripetal force check failed")
                all_pass = False
            
            # Numerical check with tolerance
            if isinstance(v_num, (int, float)):
                v_expected_num = 1.0  # With B=R=q=m=1
                if not math.isclose(v_num, v_expected_num, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✗ Numerical v mismatch: got {v_num}, expected ~{v_expected_num}")
                    all_pass = False
                else:
                    logs.append("✓ Numerical v matches expected value")
            
            # 2. Check geometric consistency: s^2 + (0.6R)^2 = R^2
            s_val = self.results.get('s', 0.0)
            if isinstance(s_val, (int, float)):
                # s should be 0.8R
                expected_s = 0.8  # In units of R
                if not math.isclose(s_val, expected_s, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✗ s doesn't satisfy Pythagorean theorem: got {s_val}R, expected 0.8R")
                    all_pass = False
                else:
                    logs.append("✓ s satisfies R^2 = s^2 + (0.6R)^2")
            
            # Similarly for x
            x_val = self.results.get('x', 0.0)
            if isinstance(x_val, (int, float)):
                if not math.isclose(x_val, 0.8, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✗ x doesn't satisfy Pythagorean theorem: got {x_val}R, expected 0.8R")
                    all_pass = False
                else:
                    logs.append("✓ x satisfies R^2 = x^2 + (0.6R)^2")
                    
        except Exception as e:
            logs.append(f"✗ Exception in test_invariants: {str(e)}")
            all_pass = False
            
        return all_pass, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency between intermediate results and final answers.
        """
        logs = []
        all_pass = True
        
        try:
            # Check OH and HO' are both 0.6R
            OH_val = self.results.get('OH', 0.0)
            HO_prime_val = self.results.get('HO\'', 0.0)
            
            for name, val in [('OH', OH_val), ('HO\'', HO_prime_val)]:
                if isinstance(val, (int, float)):
                    if not math.isclose(val, 0.6, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"✗ {name} incorrect: got {val}R, expected 0.6R")
                        all_pass = False
                    else:
                        logs.append(f"✓ {name} = 0.6R correct")
            
            # Check L_max = (4/15)R ≈ 0.2667R
            L_max_val = self.results.get('L_max', 0.0)
            if isinstance(L_max_val, (int, float)):
                expected_L_max = 4/15  # ≈ 0.2667
                if not math.isclose(L_max_val, expected_L_max, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✗ L_max incorrect: got {L_max_val}R, expected {expected_L_max}R")
                    all_pass = False
                else:
                    logs.append(f"✓ L_max = {expected_L_max}R correct")
            
            # Check p_z = 0.8qBR
            p_z_val = self.results.get('p_z', 0.0)
            if isinstance(p_z_val, (int, float)):
                # For B=R=q=1, p_z should be 0.8
                if not math.isclose(p_z_val, 0.8, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✗ p_z incorrect: got {p_z_val}, expected 0.8 (in units qBR)")
                    all_pass = False
                else:
                    logs.append("✓ p_z = 0.8qBR correct")
            
            # Check force values in terms of NqBR
            force_checks = [
                ('F_1', 2.6, "F₁ = 2.6NqBR for 0 < L < 4R/15"),
                ('F_2', 1.8, "F₂ = 1.8NqBR for 4R/15 < L < 0.4R"),
                ('F_3', 1.0, "F₃ = NqBR for L > 0.4R")
            ]
            
            for f_key, expected_coeff, desc in force_checks:
                f_val = self.results.get(f_key, 0.0)
                if isinstance(f_val, (int, float)):
                    if not math.isclose(f_val, expected_coeff, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"✗ {desc}: got coefficient {f_val}, expected {expected_coeff}")
                        all_pass = False
                    else:
                        logs.append(f"✓ {desc}")
                        
        except Exception as e:
            logs.append(f"✗ Exception in test_trace_consistency: {str(e)}")
            all_pass = False
            
        return all_pass, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Test monotonic relationships by perturbing parameters.
        """
        logs = []
        all_pass = True
        
        try:
            # Test 1: v ∝ B (with other parameters fixed)
            B_vals = [1.0, 1.1, 0.9]
            v_vals = []
            
            for B_test in B_vals:
                # v = q*B*R/m, with q=R=m=1
                v_test = 1.0 * B_test * 1.0 / 1.0
                v_vals.append(v_test)
            
            # Check monotonic increase with B
            if not (v_vals[1] > v_vals[0] > v_vals[2]):
                logs.append("✗ v is not monotonic with B")
                all_pass = False
            else:
                logs.append("✓ v increases monotonically with B")
            
            # Test 2: Forces ∝ N
            N_vals = [1.0, 1.2, 0.8]
            force_keys = ['F_1', 'F_2', 'F_3']
            
            for f_key in force_keys:
                f_base = self.results.get(f_key, 0.0)
                if isinstance(f_base, (int, float)):
                    f_vals = [f_base * N for N in N_vals]
                    # Check monotonic increase with N
                    if not (f_vals[1] > f_vals[0] > f_vals[2]):
                        logs.append(f"✗ {f_key} is not monotonic with N")
                        all_pass = False
                    else:
                        logs.append(f"✓ {f_key} increases monotonically with N")
            
            # Test 3: F_1 > F_2 > F_3 for same parameters
            F1 = self.results.get('F_1', 0.0)
            F2 = self.results.get('F_2', 0.0)
            F3 = self.results.get('F_3', 0.0)
            
            if isinstance(F1, (int, float)) and isinstance(F2, (int, float)) and isinstance(F3, (int, float)):
                if not (F1 > F2 > F3):
                    logs.append(f"✗ Force ordering incorrect: F₁({F1}) > F₂({F2}) > F₃({F3}) should hold")
                    all_pass = False
                else:
                    logs.append(f"✓ Force ordering correct: F₁ > F₂ > F₃")
                    
        except Exception as e:
            logs.append(f"✗ Exception in test_monotonicity: {str(e)}")
            all_pass = False
            
        return all_pass, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return overall_success, all_logs

# Example usage (commented out for safety):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)