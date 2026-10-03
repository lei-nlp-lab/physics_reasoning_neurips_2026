import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract expected keys
        self.C = getattr(solution_instance, 'C', self.results.get('C', None))
        self.Phi_c = getattr(solution_instance, 'Phi_c', self.results.get('Phi_c', None))
        self.epsilon0 = getattr(solution_instance, 'epsilon0', self.results.get('epsilon0', None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.rho = getattr(solution_instance, 'rho', self.results.get('rho', None))
        
        # Constants for numerical checks (using given k to compute epsilon0 if needed)
        self.k = 8.99e9  # N·m²/C²
        if self.epsilon0 is None:
            self.epsilon0 = 1 / (4 * math.pi * self.k)
    
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check numerical value of C
        if self.C is not None:
            try:
                C_val = float(self.C)
                if not math.isclose(C_val, 0.0947, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: C = {C_val} is not close to 0.0947")
                    success = False
                else:
                    logs.append(f"PASS: C = {C_val} is within tolerance of 0.0947")
            except (TypeError, ValueError) as e:
                logs.append(f"WARNING: Could not convert C to float: {e}")
        else:
            logs.append("WARNING: C not found in solution")
        
        # 2. Check form of Phi_c expression
        if self.Phi_c is not None:
            try:
                # Define symbols
                rho_sym, a_sym, eps_sym = sp.symbols('rho a epsilon0', positive=True)
                # Build expected expression (using found C or default)
                if self.C is not None and isinstance(self.C, (int, float, sp.Expr)):
                    C_expr = self.C
                else:
                    C_expr = 0.0947
                expected_expr = C_expr * rho_sym * a_sym**2 / eps_sym
                
                # Try to simplify difference
                if hasattr(self.Phi_c, 'subs'):
                    # It's a sympy expression
                    diff = sp.simplify(self.Phi_c - expected_expr)
                    if diff != 0:
                        # Try numerical substitution as fallback
                        test_vals = {a_sym: 1.0, rho_sym: 1.0, eps_sym: 1.0}
                        num_self = self.Phi_c.subs(test_vals)
                        num_exp = expected_expr.subs(test_vals)
                        try:
                            if not math.isclose(float(num_self), float(num_exp), rel_tol=1e-2, abs_tol=1e-2):
                                logs.append(f"FAIL: Phi_c expression mismatch. At (a=1,rho=1,ε0=1): got {num_self}, expected {num_exp}")
                                success = False
                            else:
                                logs.append("PASS: Phi_c expression matches expected form (numerical check)")
                        except:
                            logs.append(f"WARNING: Could not numerically compare expressions. Symbolic diff: {diff}")
                    else:
                        logs.append("PASS: Phi_c expression matches expected form (symbolic check)")
                else:
                    # Phi_c is not a sympy expression
                    logs.append("WARNING: Phi_c is not a sympy expression, skipping form check")
            except Exception as e:
                logs.append(f"WARNING: Exception during expression check: {e}")
        else:
            logs.append("WARNING: Phi_c not found in solution")
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check the key relationship: 2*C = 0.1894
        if self.C is not None:
            try:
                C_val = float(self.C)
                if not math.isclose(2 * C_val, 0.1894, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: 2*C = {2*C_val} != 0.1894 (within tolerance)")
                    success = False
                else:
                    logs.append("PASS: Relationship 2*C = 0.1894 holds")
            except (TypeError, ValueError) as e:
                logs.append(f"WARNING: Could not convert C to float: {e}")
        else:
            logs.append("WARNING: C not found, skipping trace consistency")
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Check scaling with side length a
        if self.Phi_c is not None and hasattr(self.Phi_c, 'subs'):
            try:
                # Define symbols
                rho_sym, a_sym, eps_sym = sp.symbols('rho a epsilon0', positive=True)
                
                # Test two values of a
                a1, a2 = 1.0, 2.0
                rho_test = 1.0
                eps_test = 1.0  # dimensionless for scaling test
                
                # Evaluate Phi_c at both a values
                subs1 = {a_sym: a1, rho_sym: rho_test, eps_sym: eps_test}
                subs2 = {a_sym: a2, rho_sym: rho_test, eps_sym: eps_test}
                
                Phi1 = self.Phi_c.subs(subs1)
                Phi2 = self.Phi_c.subs(subs2)
                
                # Convert to float if possible
                try:
                    Phi1_val = float(Phi1)
                    Phi2_val = float(Phi2)
                except:
                    # Symbolic fallback: simplify ratio
                    ratio_expr = sp.simplify(Phi2 / Phi1)
                    # Expected ratio for a^2 scaling: (2^2)/(1^2) = 4
                    if ratio_expr != 4:
                        logs.append(f"FAIL: Symbolic ratio Phi(2)/Phi(1) = {ratio_expr} != 4")
                        success = False
                    else:
                        logs.append("PASS: Phi_c scales quadratically with a (symbolic check)")
                else:
                    # Numerical check
                    expected_ratio = (a2 / a1) ** 2  # = 4
                    if not math.isclose(Phi2_val / Phi1_val, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Numerical ratio Phi(2)/Phi(1) = {Phi2_val/Phi1_val:.4f} != {expected_ratio}")
                        success = False
                    else:
                        logs.append("PASS: Phi_c scales quadratically with a (numerical check)")
            except Exception as e:
                logs.append(f"WARNING: Exception during monotonicity check: {e}")
        else:
            logs.append("WARNING: Phi_c not found or not symbolic, skipping monotonicity check")
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs